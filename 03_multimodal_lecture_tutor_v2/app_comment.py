# ============================================================
# 🧑‍🏫 멀티모달 강의자료 AI 튜터 V2
# ============================================================
# 핵심 개념:
#   ① PDF 파싱 (PyMuPDF) — 텍스트 추출 + 페이지 이미지 렌더링
#   ② 로컬 키워드 검색 — Embedding API 없이 비용 Zero
#   ③ Vision + RAG — PDF 텍스트 문맥 + 이미지를 함께 LLM에 전달
#   ④ Structured Output — JSON Schema로 요약/키워드/퀴즈 형식 고정
#
# 전체 흐름:
#   PDF 업로드
#     └─ PyMuPDF로 텍스트 추출 (extract_pdf_pages)
#     └─ PyMuPDF로 페이지 이미지 렌더링 (render_pdf_page_images)
#          ↓
#   [탭1] 요약/퀴즈: 텍스트 + 이미지 → LLM → JSON (Structured Output)
#   [탭2] 질문하기: 키워드 검색 → 관련 페이지 → LLM → 자연어 답변
#   [탭3] 텍스트 확인: 페이지별 추출 텍스트 원문 열람
# ============================================================

import os
import json
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import fitz                  # PyMuPDF: PDF 파싱 라이브러리
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image        # 이미지 처리 (현재 직접 사용하진 않지만 PIL 호환성용)

# .env 파일에서 OPENAI_API_KEY 로드
load_dotenv()

# ── 상수 설정 ──────────────────────────────────────────────
APP_TITLE = "🧑‍🏫 멀티모달 강의자료 AI 튜터 V2"
LLM_MODEL = "gpt-4o-mini"   # Vision + Structured Output 모두 지원하는 모델

st.set_page_config(page_title=APP_TITLE, page_icon="🧑‍🏫", layout="wide")


# ── ① OpenAI 클라이언트 생성 ───────────────────────────────
def get_client() -> OpenAI:
    """환경변수에서 API 키를 읽어 OpenAI 클라이언트를 반환합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. 프로젝트 루트의 .env 파일을 확인하세요.")
    return OpenAI(api_key=api_key)


# ── ② 업로드 파일 → 임시 파일 저장 ──────────────────────
def save_uploaded_file(uploaded_file, suffix: str) -> str:
    """
    Streamlit 업로드 파일을 OS 임시 폴더에 저장하고 경로를 반환합니다.

    왜 필요한가?
        PyMuPDF(fitz)는 파일 경로(str)를 받아야 합니다.
        Streamlit의 UploadedFile은 메모리 객체이므로 디스크에 저장이 필요합니다.

    Args:
        uploaded_file: Streamlit UploadedFile 객체
        suffix: 확장자 (예: ".pdf")
    Returns:
        임시 파일 경로 문자열
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


# ── ③ PDF 텍스트 추출 ─────────────────────────────────────
def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    PyMuPDF로 PDF 전체 페이지의 텍스트를 추출합니다.

    반환 형식:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    주의:
        스캔된 PDF(이미지 PDF)는 텍스트가 비어 있을 수 있습니다.
        이 경우 키워드 검색은 작동하지 않으며, Vision으로만 분석합니다.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):          # 페이지 번호를 1부터 시작
        text = page.get_text("text").strip()          # 텍스트 레이어 추출
        pages.append({"page": i, "text": text})
    doc.close()
    return pages


# ── ④ PDF → 페이지 이미지 렌더링 ─────────────────────────
def render_pdf_page_images(pdf_path: str, max_pages: int = 3) -> List[str]:
    """
    PDF 앞부분 페이지를 PNG 이미지로 렌더링합니다.

    두 가지 용도:
        1. Streamlit 미리보기 (사용자가 슬라이드 내용 확인)
        2. Vision 입력 (LLM이 슬라이드 이미지를 시각적으로 분석)

    Matrix(1.5, 1.5):
        PDF를 1.5배 해상도로 렌더링합니다. (기본 72dpi → 약 108dpi)
        너무 높으면 토큰이 많이 소모되므로 1.5배가 수업용 적정값입니다.

    alpha=False:
        투명도(알파채널) 없이 렌더링 → PNG 파일 크기 감소

    Returns:
        렌더링된 이미지 파일 경로 리스트
    """
    doc = fitz.open(pdf_path)
    image_paths = []
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        # 프로세스 ID(os.getpid())를 파일명에 포함 → 동시 접속자 파일 충돌 방지
        out_path = Path(tempfile.gettempdir()) / f"lecture_page_{os.getpid()}_{i+1}.png"
        pix.save(str(out_path))
        image_paths.append(str(out_path))
    doc.close()
    return image_paths


# ── ⑤ 로컬 키워드 검색 (Embedding API 없음) ─────────────
def keyword_search(pages: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Embedding API 없이 단순 키워드 빈도로 관련 페이지를 검색합니다.

    왜 이 방식인가?
        - Embedding API(text-embedding-3 등)는 호출 비용과 Rate Limit 발생
        - 수업 중 API 한도 초과 리스크를 없애기 위한 '수업 안정형' 구조
        - 실제 프로덕션에서는 BM25 또는 벡터 검색으로 교체 가능

    동작 방식:
        1. 질문을 공백/구두점으로 분리 → 2글자 이상 단어만 추출
        2. 각 페이지 텍스트에서 각 단어가 몇 번 등장하는지 합산 → 점수
        3. 점수 내림차순으로 top_k개 반환

    score=0.01 처리:
        텍스트가 있지만 키워드가 하나도 매칭되지 않는 페이지에
        최소 점수를 부여하여 완전히 배제되지 않도록 합니다.

    Args:
        pages:  extract_pdf_pages()의 반환값
        query:  사용자 질문 문자열
        top_k:  반환할 최대 페이지 수
    """
    # 질문을 단어로 분리 (물음표/쉼표 제거, 2글자 이상만)
    query_terms = [
        t.strip().lower()
        for t in query.replace("?", " ").replace(",", " ").split()
        if len(t.strip()) >= 2
    ]

    scored = []
    for p in pages:
        text = p["text"] or ""
        lower = text.lower()
        # 각 키워드가 페이지에 몇 번 등장하는지 합산
        score = sum(lower.count(term) for term in query_terms)
        # 텍스트는 있지만 키워드 미매칭 → 최소 점수 부여
        if score == 0 and text:
            score = 0.01
        scored.append({**p, "score": score})

    # 점수 내림차순 정렬 → 상위 top_k 반환
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ── ⑥ 이미지 → Base64 Data URL 변환 ─────────────────────
def image_to_data_url(image_path: str) -> str:
    """
    디스크의 이미지 파일을 Base64 Data URL로 변환합니다.
    (render_pdf_page_images로 저장한 PNG 파일용)

    OpenAI Vision API는 이미지를
      - 외부 URL (https://...) 또는
      - Base64 Data URL (data:image/png;base64,...) 형식으로 받습니다.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def uploaded_image_to_data_url(uploaded_image) -> str:
    """
    Streamlit UploadedFile(추가 이미지)을 Base64 Data URL로 변환합니다.
    (사용자가 직접 업로드한 슬라이드 캡처/표 이미지용)
    """
    raw = uploaded_image.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = uploaded_image.type or "image/png"
    return f"data:{mime};base64,{b64}"


# ── ⑦ Vision + RAG: 질문 답변 ───────────────────────────
def ask_llm_with_context(
    question: str,
    contexts: List[Dict[str, Any]],
    image_data_urls: List[str]
) -> str:
    """
    키워드 검색으로 찾은 PDF 문맥 + 이미지를 함께 LLM에 전달하여 답변을 생성합니다.
    이것이 이 앱의 핵심 RAG(Retrieval-Augmented Generation) 패턴입니다.

    RAG 흐름:
        [검색] keyword_search → 관련 페이지 텍스트 추출
        [증강] 텍스트 + 이미지를 프롬프트에 삽입
        [생성] LLM이 문맥 기반으로 답변 생성

    content 구성:
        content[0]  : 텍스트 (시스템 지시문 + PDF 문맥 + 질문)
        content[1~] : 이미지 Data URL (최대 3장)
                      → 텍스트와 이미지를 하나의 메시지로 묶는 것이 Vision 입력 방식

    temperature=0.2:
        0에 가까울수록 일관된 답변, 높을수록 창의적.
        사실 기반 Q&A이므로 낮게 설정합니다.

    max_tokens=900:
        비용/속도 제어. 긴 답변이 필요하면 늘리세요.
    """
    client = get_client()

    # 검색된 페이지 텍스트를 하나의 문자열로 합치기
    # text[:2500] → 페이지당 토큰 초과 방지
    context_text = "\n\n".join(
        f"[page {c['page']}]\n{c['text'][:2500]}"
        for c in contexts if c.get("text")
    )
    if not context_text:
        context_text = "PDF에서 추출된 텍스트가 없습니다. 첨부 이미지가 있다면 이미지를 중심으로 답하세요."

    # 메시지 content 구성: 텍스트 블록 + 이미지 블록들
    content = [
        {
            "type": "text",
            "text": f"""
너는 친절한 강의자료 튜터다.
아래 PDF 문맥과 이미지 자료를 근거로 한국어로 답변하라.
모르는 내용은 추측하지 말고 '자료에서 확인되지 않습니다'라고 말하라.
답변 끝에는 참고한 페이지 번호를 적어라.

[질문]
{question}

[PDF 문맥]
{context_text}
""".strip(),
        }
    ]

    # 이미지 블록 추가 (최대 3장 — 토큰/비용 제한)
    for url in image_data_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=900,
    )
    return response.choices[0].message.content


# ── ⑧ Structured Output: 요약/키워드/퀴즈 생성 ──────────
def generate_summary_and_quiz(
    pages: List[Dict[str, Any]],
    image_data_urls: List[str]
) -> Dict[str, Any]:
    """
    강의자료 전체를 분석하여 요약, 핵심 키워드, 중요 페이지, 퀴즈를 JSON으로 반환합니다.

    Structured Output 포인트:
        response_format에 json_schema를 지정하면
        LLM이 반드시 해당 형식으로만 응답합니다.
        → 파싱 오류 없이 안정적으로 dict로 변환 가능

    JSON Schema 구조:
        {
          "summary": "전체 요약 문장",
          "keywords": ["키워드1", "키워드2", ...],
          "important_pages": [{"page": 3, "reason": "핵심 개념 설명"}, ...],
          "quiz": [
            {"question": "질문", "answer": "정답", "hint": "힌트"},
            ...  (5개)
          ]
        }

    pages[:8]:
        처음 8페이지만 사용 → 토큰 수 제한 (전체 PDF가 매우 길 경우 대비)
    text[:1800]:
        페이지당 최대 1800자 → 총 토큰 예산 관리
    """
    client = get_client()

    # 텍스트 준비 (앞 8페이지, 페이지당 1800자)
    text = "\n\n".join(
        f"[page {p['page']}] {p['text'][:1800]}"
        for p in pages[:8] if p.get("text")
    )
    if not text:
        text = "PDF 텍스트가 비어 있습니다. 이미지 자료를 기반으로 요약과 퀴즈를 생성하세요."

    # ── JSON Schema 정의 ───────────────────────────────────
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "lecture_summary_quiz",
            "schema": {
                "type": "object",
                "additionalProperties": False,   # 스키마 외 키 불허
                "properties": {
                    "summary": {"type": "string"},   # 전체 요약
                    "keywords": {                    # 핵심 키워드 배열
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "important_pages": {             # 중요 페이지 목록
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "page":   {"type": "integer"},  # 페이지 번호
                                "reason": {"type": "string"}    # 중요한 이유
                            },
                            "required": ["page", "reason"]
                        }
                    },
                    "quiz": {                        # 퀴즈 5개
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},  # 문제
                                "answer":   {"type": "string"},  # 정답
                                "hint":     {"type": "string"}   # 힌트
                            },
                            "required": ["question", "answer", "hint"]
                        }
                    }
                },
                "required": ["summary", "keywords", "important_pages", "quiz"]
            }
        }
    }

    # 메시지 content: 텍스트 지시문 + 이미지 (최대 3장)
    content = [{"type": "text", "text": f"""
다음 강의자료를 분석해서 수업용 요약, 핵심 키워드, 중요 페이지, 퀴즈 5개를 JSON으로 만들어라.
한국어로 작성하라.

[PDF 텍스트]
{text}
""".strip()}]
    for url in image_data_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": content}],
        response_format=schema,   # ← Structured Output 핵심 파라미터
        temperature=0.2,
        max_tokens=1600,
    )
    # 응답 문자열 → Python dict 파싱
    return json.loads(response.choices[0].message.content)


# ════════════════════════════════════════════════════════════
# Streamlit UI
# ════════════════════════════════════════════════════════════

st.title(APP_TITLE)
st.caption("PDF 텍스트 + 슬라이드/캡처 이미지를 함께 분석하는 저녁반 최종 프로젝트형 예제입니다.")

# ── 사이드바: 수업용 안내 ──────────────────────────────────
with st.sidebar:
    st.header("수업용 설정")
    st.write(f"LLM: `{LLM_MODEL}`")
    st.info(
        "**V2 특징: 비용 제로 검색**\n"
        "- Embedding API 미사용 → Rate Limit 없음\n"
        "- 로컬 키워드 빈도로 관련 페이지 검색\n"
        "- 실습 중 API 한도 초과 리스크 제거"
    )
    st.markdown("---")
    st.write("실행: `python -m streamlit run app.py`")

# ── 파일 업로더 ───────────────────────────────────────────
pdf_file = st.file_uploader("PDF 강의자료 업로드", type=["pdf"])
image_files = st.file_uploader(
    "추가 이미지 업로드: 슬라이드 캡처, 표, 그림 등",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True   # 여러 파일 동시 업로드 허용
)

# ── PDF 업로드 후 메인 UI ──────────────────────────────────
if pdf_file:
    # 업로드 파일을 임시 경로에 저장 (PyMuPDF가 파일 경로 필요)
    pdf_path = save_uploaded_file(pdf_file, ".pdf")

    with st.spinner("PDF 텍스트와 미리보기 이미지를 추출하는 중입니다..."):
        pages = extract_pdf_pages(pdf_path)                    # 텍스트 추출
        rendered_images = render_pdf_page_images(pdf_path, max_pages=3)  # 이미지 렌더링

    st.success(f"PDF 로딩 완료: 총 {len(pages)}페이지")

    # ── 미리보기 영역 ──────────────────────────────────────
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("PDF 앞부분 미리보기")
        for img_path in rendered_images:
            st.image(img_path, use_container_width=True)
    with col2:
        st.subheader("추가 이미지 미리보기")
        if image_files:
            for img in image_files[:3]:
                st.image(img, caption=img.name, use_container_width=True)
        else:
            st.caption("추가 이미지를 올리면 Vision 모델이 함께 참고합니다.")

    # ── Vision 입력용 Data URL 준비 ────────────────────────
    # PDF 렌더링 이미지 → Data URL
    image_data_urls = [image_to_data_url(p) for p in rendered_images]
    # 추가 업로드 이미지 → Data URL (있으면 목록에 추가)
    if image_files:
        image_data_urls.extend(uploaded_image_to_data_url(img) for img in image_files)

    st.markdown("---")

    # ── 3개 탭 구성 ───────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["요약/퀴즈 생성", "질문하기", "추출 텍스트 확인"])

    # ── 탭1: 요약 + 키워드 + 퀴즈 ────────────────────────
    with tab1:
        if st.button("요약 + 키워드 + 퀴즈 생성", type="primary"):
            try:
                with st.spinner("AI가 강의자료를 분석하는 중입니다..."):
                    result = generate_summary_and_quiz(pages, image_data_urls)

                st.subheader("요약")
                st.write(result["summary"])

                st.subheader("핵심 키워드")
                st.write(", ".join(result["keywords"]))

                st.subheader("중요 페이지")
                st.table(result["important_pages"])   # dict 리스트를 표로 자동 변환

                st.subheader("퀴즈")
                for i, q in enumerate(result["quiz"], start=1):
                    # expander: 문제를 클릭하면 힌트/정답 펼침
                    with st.expander(f"문제 {i}. {q['question']}"):
                        st.write("힌트:", q["hint"])
                        st.write("정답:", q["answer"])

            except Exception as e:
                st.error("분석 중 오류가 발생했습니다.")
                st.exception(e)

    # ── 탭2: 질문하기 (RAG Q&A) ──────────────────────────
    with tab2:
        question = st.text_input(
            "질문",
            placeholder="예: 이 자료의 핵심 개념을 초보자에게 설명해줘"
        )
        top_k = st.slider(
            "참고 페이지 수",
            min_value=1, max_value=5, value=3,
            help="많을수록 더 많은 맥락을 참고하지만 토큰 비용이 늘어납니다."
        )
        if st.button("질문하기") and question:
            try:
                # ① 키워드 검색으로 관련 페이지 추출
                contexts = keyword_search(pages, question, top_k=top_k)
                # ② 문맥 + 이미지 → LLM 답변 생성
                with st.spinner("AI가 답변을 생성하는 중입니다..."):
                    answer = ask_llm_with_context(question, contexts, image_data_urls)

                st.subheader("답변")
                st.write(answer)

                # 어떤 페이지를 참고했는지 투명하게 표시
                st.subheader("참고한 페이지 후보")
                st.table([{
                    "page":    c["page"],
                    "score":   c["score"],
                    "preview": (c["text"] or "")[:120]   # 앞 120자 미리보기
                } for c in contexts])

            except Exception as e:
                st.error("질문 처리 중 오류가 발생했습니다.")
                st.exception(e)

    # ── 탭3: 추출 텍스트 원문 확인 ───────────────────────
    with tab3:
        # 페이지 번호 선택 → 해당 페이지 텍스트 표시
        selected = st.selectbox("페이지 선택", [p["page"] for p in pages])
        page = next(p for p in pages if p["page"] == selected)
        st.text_area(
            "추출 텍스트",
            page["text"],
            height=300,
            help="스캔 PDF라면 텍스트가 비어 있을 수 있습니다."
        )

else:
    # PDF 미업로드 상태 안내
    st.info("먼저 PDF를 업로드하세요. 샘플은 `data/sample_lecture.pdf`를 사용하면 됩니다.")
