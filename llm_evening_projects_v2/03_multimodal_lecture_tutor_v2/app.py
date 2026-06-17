import os
import json
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

APP_TITLE = "🧑‍🏫 멀티모달 강의자료 AI 튜터 V2"
LLM_MODEL = "gpt-4o-mini"

st.set_page_config(page_title=APP_TITLE, page_icon="🧑‍🏫", layout="wide")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. 프로젝트 루트의 .env 파일을 확인하세요.")
    return OpenAI(api_key=api_key)


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """PDF에서 페이지별 텍스트를 추출합니다. 스캔 PDF는 텍스트가 비어 있을 수 있습니다."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        pages.append({"page": i, "text": text})
    doc.close()
    return pages


def render_pdf_page_images(pdf_path: str, max_pages: int = 3) -> List[str]:
    """PDF 앞부분 몇 페이지를 이미지로 렌더링하여 Vision 입력/미리보기용으로 사용합니다."""
    doc = fitz.open(pdf_path)
    image_paths = []
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        out_path = Path(tempfile.gettempdir()) / f"lecture_page_{os.getpid()}_{i+1}.png"
        pix.save(str(out_path))
        image_paths.append(str(out_path))
    doc.close()
    return image_paths


def keyword_search(pages: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """수업 안정형 로컬 검색. Embedding API를 쓰지 않아 비용/Rate Limit을 줄입니다."""
    query_terms = [t.strip().lower() for t in query.replace("?", " ").replace(",", " ").split() if len(t.strip()) >= 2]
    scored = []
    for p in pages:
        text = p["text"] or ""
        lower = text.lower()
        score = sum(lower.count(term) for term in query_terms)
        # 질문어가 하나도 안 걸리는 PDF를 위해 텍스트 길이 보정
        if score == 0 and text:
            score = 0.01
        scored.append({**p, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def uploaded_image_to_data_url(uploaded_image) -> str:
    raw = uploaded_image.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = uploaded_image.type or "image/png"
    return f"data:{mime};base64,{b64}"


def ask_llm_with_context(question: str, contexts: List[Dict[str, Any]], image_data_urls: List[str]) -> str:
    client = get_client()
    context_text = "\n\n".join(
        f"[page {c['page']}]\n{c['text'][:2500]}" for c in contexts if c.get("text")
    )
    if not context_text:
        context_text = "PDF에서 추출된 텍스트가 없습니다. 첨부 이미지가 있다면 이미지를 중심으로 답하세요."

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
    for url in image_data_urls[:3]:
        content.append({"type": "image_url", "image_url": {"url": url}})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=900,
    )
    return response.choices[0].message.content


def generate_summary_and_quiz(pages: List[Dict[str, Any]], image_data_urls: List[str]) -> Dict[str, Any]:
    client = get_client()
    text = "\n\n".join(f"[page {p['page']}] {p['text'][:1800]}" for p in pages[:8] if p.get("text"))
    if not text:
        text = "PDF 텍스트가 비어 있습니다. 이미지 자료를 기반으로 요약과 퀴즈를 생성하세요."

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "lecture_summary_quiz",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "important_pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "page": {"type": "integer"},
                                "reason": {"type": "string"}
                            },
                            "required": ["page", "reason"]
                        }
                    },
                    "quiz": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "question": {"type": "string"},
                                "answer": {"type": "string"},
                                "hint": {"type": "string"}
                            },
                            "required": ["question", "answer", "hint"]
                        }
                    }
                },
                "required": ["summary", "keywords", "important_pages", "quiz"]
            }
        }
    }

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
        response_format=schema,
        temperature=0.2,
        max_tokens=1600,
    )
    return json.loads(response.choices[0].message.content)


st.title(APP_TITLE)
st.caption("PDF 텍스트 + 슬라이드/캡처 이미지를 함께 분석하는 저녁반 최종 프로젝트형 예제입니다.")

with st.sidebar:
    st.header("수업용 설정")
    st.write(f"LLM: `{LLM_MODEL}`")
    st.info("이 V2는 임베딩 API를 쓰지 않는 로컬 키워드 검색 기반입니다. 비용과 Rate Limit 부담을 줄이기 위한 수업 안정형 구조입니다.")
    st.markdown("---")
    st.write("실행: `python -m streamlit run 03_multimodal_lecture_tutor_v2/app.py`")

pdf_file = st.file_uploader("PDF 강의자료 업로드", type=["pdf"])
image_files = st.file_uploader("추가 이미지 업로드: 슬라이드 캡처, 표, 그림 등", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if pdf_file:
    pdf_path = save_uploaded_file(pdf_file, ".pdf")
    with st.spinner("PDF 텍스트와 미리보기 이미지를 추출하는 중입니다..."):
        pages = extract_pdf_pages(pdf_path)
        rendered_images = render_pdf_page_images(pdf_path, max_pages=3)

    st.success(f"PDF 로딩 완료: 총 {len(pages)}페이지")

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

    image_data_urls = [image_to_data_url(p) for p in rendered_images]
    if image_files:
        image_data_urls.extend(uploaded_image_to_data_url(img) for img in image_files)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["요약/퀴즈 생성", "질문하기", "추출 텍스트 확인"])

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
                st.table(result["important_pages"])
                st.subheader("퀴즈")
                for i, q in enumerate(result["quiz"], start=1):
                    with st.expander(f"문제 {i}. {q['question']}"):
                        st.write("힌트:", q["hint"])
                        st.write("정답:", q["answer"])
            except Exception as e:
                st.error("분석 중 오류가 발생했습니다.")
                st.exception(e)

    with tab2:
        question = st.text_input("질문", placeholder="예: 이 자료의 핵심 개념을 초보자에게 설명해줘")
        top_k = st.slider("참고 페이지 수", 1, 5, 3)
        if st.button("질문하기") and question:
            try:
                contexts = keyword_search(pages, question, top_k=top_k)
                with st.spinner("AI가 답변을 생성하는 중입니다..."):
                    answer = ask_llm_with_context(question, contexts, image_data_urls)
                st.subheader("답변")
                st.write(answer)
                st.subheader("참고한 페이지 후보")
                st.table([{ "page": c["page"], "score": c["score"], "preview": (c["text"] or "")[:120] } for c in contexts])
            except Exception as e:
                st.error("질문 처리 중 오류가 발생했습니다.")
                st.exception(e)

    with tab3:
        selected = st.selectbox("페이지 선택", [p["page"] for p in pages])
        page = next(p for p in pages if p["page"] == selected)
        st.text_area("추출 텍스트", page["text"], height=300)
else:
    st.info("먼저 PDF를 업로드하세요. 샘플은 `data/sample_lecture.pdf`를 사용하면 됩니다.")
