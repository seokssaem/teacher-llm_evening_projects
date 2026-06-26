"""
╔══════════════════════════════════════════════════════════════════╗
║              04. AI Research Agent  (수업 안정형 V1)              ║
╠══════════════════════════════════════════════════════════════════╣
║  목적: 실시간 웹검색 API 없이도 동작하는 워크플로우형 Agent 입문 예제  ║
║                                                                  ║
║  Agent 단계 (순서대로 실행):                                       ║
║   ① 조사 계획 생성  →  ② 자료 요약  →  ③ 최종 보고서              ║
║   →  ④ 발표 스크립트  →  ⑤ 예상 Q&A                              ║
║                                                                  ║
║  실행 방법:                                                        ║
║   python -m streamlit run 04_research_agent/app.py               ║
╚══════════════════════════════════════════════════════════════════╝

[수업 포인트]
  - 단순 챗봇 vs. 워크플로우형 Agent 차이를 직접 확인합니다.
  - 복잡한 LangChain / LangGraph 없이도 "Agent 사고방식"을 구현할 수 있습니다.
  - 각 단계(노드)가 독립된 함수로 분리되어 있어 유지보수가 쉽습니다.
"""

# ── from __future__ import annotations ──────────────────────────────────────
# Python 3.9 이하에서도 `list[str]`, `dict[str, Any]` 같은 타입 힌트를
# 그대로 쓸 수 있게 해 주는 호환성 임포트입니다.
# Python 3.10+ 에서는 없어도 되지만, 수업 환경(3.11)에서도 습관적으로 넣으면 안전합니다.
from __future__ import annotations

import json                      # JSON 파싱 (모델 응답을 딕셔너리로 변환할 때 사용)
import os                        # 환경변수 읽기 (API 키 등)
from datetime import datetime    # 보고서 생성 시각 표시용
from typing import Any           # 타입 힌트: "어떤 타입이든 가능" 표현

import streamlit as st           # 웹 UI 프레임워크
from dotenv import load_dotenv   # .env 파일의 KEY=VALUE 를 환경변수로 읽어옵니다
from openai import OpenAI        # OpenAI Python SDK

# ── .env 파일 로드 ────────────────────────────────────────────────────────────
# 프로젝트 루트의 .env 에서 OPENAI_API_KEY 등을 읽습니다.
# .env 파일이 없으면 조용히 무시하고 넘어갑니다(에러 없음).
load_dotenv()

# ── 앱 전역 상수 ──────────────────────────────────────────────────────────────
APP_TITLE = "04. AI Research Agent"

# 기본 모델: gpt-4o-mini (속도 ↑, 비용 ↓, 수업 환경에 적합)
# gpt-4o 로 바꾸면 품질은 올라가지만 요금도 올라갑니다.
DEFAULT_MODEL = "gpt-4o-mini"


# ╔══════════════════════════════════════════════════════════════════╗
# ║                     섹션 1: 기본 유틸 함수                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def get_client() -> OpenAI | None:
    """
    ① OpenAI 클라이언트를 생성해서 반환합니다.

    반환값:
        - OpenAI 객체 : API 키가 있을 때
        - None        : API 키가 없을 때 (UI에서 별도 안내)

    [수업 포인트]
    API 키는 코드에 직접 넣지 말고, 반드시 .env + python-dotenv 조합으로
    환경변수에서 읽어야 합니다. 깃허브에 올렸다가 키가 유출되면 큰일납니다!
    """
    api_key = os.getenv("OPENAI_API_KEY")   # 환경변수에서 키 읽기
    if not api_key:
        return None                          # 키 없으면 None 반환
    return OpenAI(api_key=api_key)           # 클라이언트 생성 후 반환


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    ② OpenAI Chat Completions API를 호출하는 핵심 함수입니다.

    매개변수:
        system_prompt : AI 역할·행동 지침 (예: "당신은 보고서 작성 전문가입니다")
        user_prompt   : 실제 작업 지시 (예: "이 주제로 보고서를 써 주세요")
        temperature   : 창의성 조절 (0.0 = 일관성 최대, 1.0 = 창의성 최대)
                        → 0.2: 사실 위주의 안정적 출력

    반환값:
        모델이 생성한 텍스트 (str)

    [수업 포인트] system_prompt vs. user_prompt 역할 분리
        - system : "페르소나(persona)" 설정 → Agent의 성격·규칙
        - user   : "태스크(task)" 지시    → 실제로 할 일
        두 가지를 분리하면 같은 user_prompt에 다른 페르소나를 쉽게 바꿔 끼울 수 있습니다.
    """
    client = get_client()
    if client is None:
        # API 키가 없으면 실행을 즉시 중단하고 오류 메시지를 냅니다.
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,        # 사용할 모델 지정
        temperature=temperature,    # 창의성 수준
        messages=[
            # ── 메시지 배열 구조 ──────────────────────────────────────────
            # Chat Completions는 role(역할) + content(내용) 딕셔너리 목록을 받습니다.
            # "system" → AI 행동 지침 / "user" → 사용자 요청 / "assistant" → AI 응답
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    # response.choices[0].message.content: 첫 번째 응답 텍스트
    # 응답이 비어있을 경우를 대비해 or "" 처리
    return response.choices[0].message.content or ""


def safe_json_loads(text: str) -> dict[str, Any]:
    """
    ③ 모델 응답에서 JSON을 안전하게 파싱합니다.

    모델이 JSON을 반환할 때 ```json ... ``` 마크다운 코드블록을 감싸서 주는 경우가 많습니다.
    이 함수는 그 코드블록 기호를 먼저 제거한 뒤 파싱합니다.

    [수업 포인트]
    실제 프로덕션에서도 LLM 응답을 구조화(JSON)해서 받을 때
    이런 전처리 함수가 꼭 필요합니다. 모델이 항상 깔끔한 JSON만 주지는 않거든요!
    """
    cleaned = text.strip()

    # 마크다운 코드블록 제거 (순서 중요: ```json 먼저, 그 다음 ```)
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    try:
        return json.loads(cleaned)              # 정상 파싱 성공
    except json.JSONDecodeError:
        # 파싱 실패 시: 원문을 "raw_response" 키에 그대로 담아 반환
        # 앱이 죽지 않도록 graceful degradation 처리
        return {"raw_response": text}


def make_download_text(topic: str, report: str, script: str, qa: str) -> str:
    """
    ④ 보고서·스크립트·Q&A를 하나의 Markdown 파일로 합칩니다.

    [수업 포인트]
    생성된 결과물을 파일로 내려받을 수 있게 하면 포트폴리오에 바로 활용 가능합니다.
    datetime.now().strftime()으로 생성 시각을 기록해 버전 관리에도 도움이 됩니다.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")   # 예: 2025-06-26 14:30
    return f"""# AI Research Agent 보고서

- 주제: {topic}
- 생성 시각: {now}

---

## 1. 최종 보고서

{report}

---

## 2. 발표 스크립트

{script}

---

## 3. 예상 질문과 답변

{qa}
"""


# ╔══════════════════════════════════════════════════════════════════╗
# ║               섹션 2: 프롬프트 빌더 함수 (5개 단계)               ║
# ║                                                                  ║
# ║  [설계 원칙]                                                      ║
# ║  각 단계마다 독립된 함수로 프롬프트를 만들어 (system, user) 튜플로  ║
# ║  반환합니다. call_llm()과 분리하면:                                ║
# ║   - 프롬프트만 단독으로 수정·테스트 가능                            ║
# ║   - 여러 단계가 같은 call_llm()을 재사용 → DRY 원칙 준수           ║
# ║   - 나중에 LangChain PromptTemplate으로 쉽게 교체 가능             ║
# ╚══════════════════════════════════════════════════════════════════╝

def build_plan_prompt(topic: str, audience: str, goal: str) -> tuple[str, str]:
    """
    [단계 ①] 조사 계획 생성 프롬프트

    역할: 보고서를 쓰기 전, 무엇을 어떻게 조사할지 계획을 세웁니다.
    → Agent 관점: '계획(Planning)' 단계에 해당합니다.

    반환: (system_prompt, user_prompt) 튜플
    """
    system = """당신은 AI/IT 수업용 리서치 코치입니다.
학생이 조사 주제를 정하면, 실제 보고서 작성을 위한 조사 계획을 명확하고 실용적으로 제안합니다.
출력은 반드시 한국어로 작성하세요."""

    user = f"""
조사 주제: {topic}
대상 독자/청중: {audience}
보고서 목적: {goal}

아래 형식으로 조사 계획을 작성해 주세요.
1. 핵심 질문 5개
2. 찾아야 할 자료 유형 5개
3. 보고서 목차 초안
4. 좋은 자료인지 판단하는 기준
5. 학생이 바로 검색할 수 있는 검색어 8개
"""
    # f-string으로 사용자 입력값(topic, audience, goal)을 프롬프트에 동적으로 삽입합니다.
    return system, user


def build_summary_prompt(topic: str, pasted_sources: str) -> tuple[str, str]:
    """
    [단계 ②] 붙여넣은 자료 요약 프롬프트

    역할: 사용자가 직접 붙여넣은 자료만 분석하여 핵심을 추출합니다.
    → Agent 관점: '정보 수집·분석(Observation)' 단계에 해당합니다.

    [수업 포인트] Grounding (근거 기반 생성)
    system 프롬프트에 "자료에 없는 내용은 추측하지 말라"고 명시합니다.
    이처럼 AI의 '환각(hallucination)'을 줄이는 지시를 프롬프트에 넣는 것이
    RAG(Retrieval-Augmented Generation) 설계의 핵심입니다.
    """
    system = """당신은 자료 분석가입니다.
사용자가 붙여넣은 자료만 근거로 핵심 내용을 요약합니다.
자료에 없는 내용은 추측하지 말고 '제공 자료에서 확인되지 않음'이라고 표시하세요.
출력은 한국어로 작성하세요."""

    user = f"""
주제: {topic}

[붙여넣은 자료]
{pasted_sources}

아래 형식으로 정리해 주세요.
1. 핵심 요약 5줄
2. 중요한 사실/수치/근거
3. 서로 다른 관점 또는 쟁점
4. 보고서에 반드시 넣을 포인트
5. 추가로 확인하면 좋은 내용
"""
    return system, user


def build_report_prompt(topic: str, audience: str, goal: str, pasted_sources: str) -> tuple[str, str]:
    """
    [단계 ③] 최종 보고서 생성 프롬프트

    역할: 수집한 자료를 바탕으로 포트폴리오용 완성 보고서를 작성합니다.
    → Agent 관점: '실행(Action)' 단계 — 가장 핵심적인 산출물을 생성합니다.

    [수업 포인트] Markdown 형식 지정
    "출력은 한국어 Markdown으로 작성하세요"라고 명시하면
    st.markdown()으로 바로 렌더링할 수 있는 형태로 출력됩니다.
    이처럼 출력 형식을 프롬프트에서 제어하는 것이 프롬프트 엔지니어링의 기초입니다.
    """
    system = """당신은 수업용 보고서를 작성하는 AI 리서치 에이전트입니다.
사용자가 제공한 자료를 중심으로, 포트폴리오에 넣을 수 있는 깔끔한 보고서를 작성합니다.
근거 없는 과장은 피하고, 자료에 없는 내용은 명확히 구분하세요.
출력은 한국어 Markdown으로 작성하세요."""

    user = f"""
조사 주제: {topic}
대상 독자/청중: {audience}
보고서 목적: {goal}

[붙여넣은 자료]
{pasted_sources}

다음 구조로 보고서를 작성해 주세요.
# 제목
## 1. 한 줄 요약
## 2. 배경
## 3. 핵심 내용
## 4. 활용 사례
## 5. 한계와 주의점
## 6. 결론
## 7. 포트폴리오 확장 아이디어
"""
    return system, user


def build_script_prompt(topic: str, report: str) -> tuple[str, str]:
    """
    [단계 ④] 발표 스크립트 생성 프롬프트

    역할: 작성된 보고서를 3분 발표 스크립트로 변환합니다.
    → Agent 관점: '변환(Transform)' 단계 — 같은 내용을 다른 형식으로 재가공합니다.

    [수업 포인트] 체이닝(Chaining)
    이전 단계의 출력(report)을 다음 단계의 입력으로 연결합니다.
    이것이 바로 워크플로우형 Agent의 핵심 패턴 — '체인(Chain)'입니다.
    LangChain의 이름도 여기서 유래했습니다.
    """
    system = """당신은 발표 코치입니다.
보고서를 3분 발표용 스크립트로 바꾸고, 발표자가 자연스럽게 말할 수 있게 작성합니다.
출력은 한국어로 작성하세요."""

    user = f"""
주제: {topic}

[보고서]
{report}

아래 형식으로 작성해 주세요.
1. 30초 오프닝
2. 2분 핵심 발표 스크립트
3. 30초 마무리
4. 발표 슬라이드 제목 5개
"""
    return system, user


def build_qa_prompt(topic: str, report: str) -> tuple[str, str]:
    """
    [단계 ⑤] 예상 Q&A 생성 프롬프트

    역할: 발표 후 나올 수 있는 예상 질문과 모범 답변을 준비합니다.
    → Agent 관점: '검증(Verification)' 단계 — 결과물의 빈틈을 미리 점검합니다.

    [수업 포인트] 다양성 지시
    "쉬운 질문, 비판적 질문, 기술적 질문이 섞이게"라고 명시하면
    모델이 특정 유형에 편중되지 않고 균형 있는 Q&A를 생성합니다.
    """
    system = """당신은 발표 후 질의응답을 준비하는 코치입니다.
예상 질문과 답변을 현실적으로 만듭니다.
출력은 한국어로 작성하세요."""

    user = f"""
주제: {topic}

[보고서]
{report}

예상 질문 7개와 모범 답변을 만들어 주세요.
질문은 쉬운 질문, 비판적 질문, 기술적 질문이 섞이게 작성하세요.
"""
    return system, user


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    섹션 3: Streamlit UI                          ║
# ║                                                                  ║
# ║  [실행 순서]                                                      ║
# ║  Streamlit은 파일을 위→아래로 순서대로 실행합니다.                  ║
# ║  버튼을 누르거나 입력값이 바뀌면 전체 스크립트가 다시 실행됩니다.    ║
# ║  → st.session_state로 결과값을 저장해 재실행 시에도 유지합니다.    ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 페이지 기본 설정 ─────────────────────────────────────────────────────────
# 반드시 다른 st. 명령보다 먼저 호출해야 합니다.
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🕵️",
    layout="wide",          # 화면 전체 너비 사용 (기본값: "centered")
)

st.title("🕵️ AI Research Agent")
st.caption("수업 안정형 버전: 실시간 웹검색 없이, 붙여넣은 자료 기반으로 보고서를 생성합니다.")


# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("실행 상태")

    # API 키 유무에 따라 초록색 성공 / 빨간색 에러 배지 표시
    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY 감지됨")
    else:
        st.error("OPENAI_API_KEY 없음")

    st.markdown("---")
    st.subheader("수업 포인트")
    st.markdown(
        """
- Agent를 꼭 복잡한 프레임워크로 만들 필요는 없습니다.
- `계획 → 자료 분석 → 보고서 → 발표 → Q&A`처럼 단계를 나누면 워크플로우형 Agent가 됩니다.
- V1은 안정성을 위해 웹검색 API를 쓰지 않습니다.
"""
    )


# ── 2컬럼 레이아웃 ────────────────────────────────────────────────────────────
# [1, 1] = 왼쪽:오른쪽 비율. [2, 1]이면 왼쪽이 2배 넓어집니다.
col1, col2 = st.columns([1, 1])

with col1:
    # ── 입력 영역 ────────────────────────────────────────────────────────────
    st.subheader("1. 조사 설정")

    # value= : 앱을 처음 열었을 때 보여줄 기본값 (데모 시 편리)
    topic    = st.text_input("조사 주제",       value="생성형 AI가 교육 분야에 미치는 영향")
    audience = st.text_input("대상 독자/청중",   value="AI 입문 수강생")
    goal     = st.text_input("보고서 목적",      value="수업 발표와 포트폴리오 정리")

    st.subheader("2. 자료 붙여넣기")
    pasted_sources = st.text_area(
        "뉴스, 블로그, 논문 초록, 회사 자료 등을 붙여넣으세요",
        height=300,
        placeholder="여기에 조사 자료를 붙여넣으세요.\n예: 기사 요약, 공식 문서 일부, 통계 자료, 블로그 내용 등",
    )

with col2:
    # ── 버튼 영역 ────────────────────────────────────────────────────────────
    st.subheader("3. Agent 실행")
    st.info("처음에는 '조사 계획 생성'만 눌러도 수업 흐름 설명이 가능합니다.")

    # use_container_width=True : 버튼이 컬럼 너비를 꽉 채웁니다.
    # [참고] Streamlit 2026 이후 버전에서는 width='stretch' 로 변경될 예정입니다.
    plan_btn    = st.button("① 조사 계획 생성",                 use_container_width=True)
    summary_btn = st.button("② 붙여넣은 자료 요약",             use_container_width=True)
    report_btn  = st.button("③ 최종 보고서 생성",               use_container_width=True)
    full_btn    = st.button("④ 보고서 + 발표 + Q&A 한번에 생성", use_container_width=True)


# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
# [핵심 개념] st.session_state
#
# Streamlit은 버튼을 누를 때마다 스크립트 전체를 위→아래로 재실행합니다.
# 일반 변수는 재실행마다 초기화되지만,
# st.session_state에 저장한 값은 재실행 후에도 유지됩니다.
#
# if "key" not in st.session_state:  →  첫 실행 시에만 초기화 (중요!)
# 이미 존재하면 덮어쓰지 않고 넘어갑니다.
if "plan"    not in st.session_state: st.session_state.plan    = ""
if "summary" not in st.session_state: st.session_state.summary = ""
if "report"  not in st.session_state: st.session_state.report  = ""
if "script"  not in st.session_state: st.session_state.script  = ""
if "qa"      not in st.session_state: st.session_state.qa      = ""


# ── 버튼 클릭 처리 ───────────────────────────────────────────────────────────
# try/except로 전체를 감싸면 어느 단계에서 오류가 나도 앱이 죽지 않습니다.
try:

    # ── ① 조사 계획 생성 ─────────────────────────────────────────────────────
    if plan_btn:
        # st.spinner: API 호출 중 로딩 아이콘과 안내 문구를 표시합니다.
        with st.spinner("조사 계획 생성 중..."):
            system, user = build_plan_prompt(topic, audience, goal)
            st.session_state.plan = call_llm(system, user)
            # 결과를 session_state에 저장 → 재실행 후에도 결과 유지

    # ── ② 자료 요약 ──────────────────────────────────────────────────────────
    if summary_btn:
        if not pasted_sources.strip():
            # 자료 없이 요약을 누르면 경고만 표시하고 API는 호출하지 않습니다.
            st.warning("먼저 자료를 붙여넣어 주세요.")
        else:
            with st.spinner("자료 요약 중..."):
                system, user = build_summary_prompt(topic, pasted_sources)
                st.session_state.summary = call_llm(system, user)

    # ── ③ 최종 보고서 생성 ───────────────────────────────────────────────────
    if report_btn:
        if not pasted_sources.strip():
            st.warning("먼저 자료를 붙여넣어 주세요.")
        else:
            with st.spinner("보고서 생성 중..."):
                system, user = build_report_prompt(topic, audience, goal, pasted_sources)
                st.session_state.report = call_llm(system, user)

    # ── ④ 보고서 + 발표 스크립트 + Q&A 한 번에 생성 ────────────────────────
    if full_btn:
        if not pasted_sources.strip():
            st.warning("먼저 자료를 붙여넣어 주세요.")
        else:
            # [수업 포인트] 순차 실행 체이닝 패턴
            # 보고서 → 스크립트 → Q&A 순서로 실행하며,
            # 앞 단계의 출력(st.session_state.report)을 다음 단계 입력으로 넘깁니다.

            # Step 1: 보고서 생성
            with st.spinner("보고서 생성 중..."):
                system, user = build_report_prompt(topic, audience, goal, pasted_sources)
                st.session_state.report = call_llm(system, user)

            # Step 2: 보고서 → 발표 스크립트 (이전 단계 결과를 입력으로 사용)
            with st.spinner("발표 스크립트 생성 중..."):
                system, user = build_script_prompt(topic, st.session_state.report)
                st.session_state.script = call_llm(system, user)

            # Step 3: 보고서 → 예상 Q&A (이전 단계 결과를 입력으로 사용)
            with st.spinner("예상 Q&A 생성 중..."):
                system, user = build_qa_prompt(topic, st.session_state.report)
                st.session_state.qa = call_llm(system, user)

except Exception as e:
    # 어떤 오류든 사용자에게 친절하게 표시하고 상세 내용도 보여줍니다.
    st.error("실행 중 오류가 발생했습니다.")
    st.exception(e)    # 스택 트레이스까지 출력 (디버깅에 유용)


# ── 결과 출력 탭 ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("결과")

# st.tabs: 탭 UI를 만듭니다. 반환값은 각 탭 컨텍스트 객체의 튜플입니다.
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["조사 계획", "자료 요약", "최종 보고서", "발표 스크립트", "예상 Q&A"]
)

with tab1:
    if st.session_state.plan:
        # st.markdown: Markdown 문자열을 HTML로 렌더링합니다.
        # 모델이 **볼드**, ## 헤더 등을 사용했다면 그대로 예쁘게 표시됩니다.
        st.markdown(st.session_state.plan)
    else:
        st.caption("조사 계획이 아직 없습니다.")    # 회색 작은 글씨

with tab2:
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.caption("자료 요약이 아직 없습니다.")

with tab3:
    if st.session_state.report:
        st.markdown(st.session_state.report)
    else:
        st.caption("최종 보고서가 아직 없습니다.")

with tab4:
    if st.session_state.script:
        st.markdown(st.session_state.script)
    else:
        st.caption("발표 스크립트가 아직 없습니다.")

with tab5:
    if st.session_state.qa:
        st.markdown(st.session_state.qa)
    else:
        st.caption("예상 Q&A가 아직 없습니다.")


# ── 다운로드 버튼 ─────────────────────────────────────────────────────────────
# 보고서·스크립트·Q&A 중 하나라도 있으면 다운로드 버튼을 표시합니다.
if st.session_state.report or st.session_state.script or st.session_state.qa:
    download_text = make_download_text(
        topic=topic,
        report=st.session_state.report,
        script=st.session_state.script,
        qa=st.session_state.qa,
    )
    st.download_button(
        label="📥 Markdown 보고서 다운로드",
        data=download_text,
        file_name="research_agent_report.md",
        mime="text/markdown",        # 브라우저에 파일 타입 알림
        use_container_width=True,
    )


# ── 수업용 설명 접기/펼치기 ──────────────────────────────────────────────────
# st.expander: 클릭하면 펼쳐지는 섹션입니다. 부가 설명을 숨겨두기 좋습니다.
with st.expander("📚 수업용 설명: 이게 왜 Agent인가요?"):
    st.markdown(
        """
### 단순 챗봇 vs. 워크플로우형 Agent

| 구분 | 단순 챗봇 | 이 예제 (워크플로우형 Agent) |
|------|-----------|---------------------------|
| 처리 방식 | 질문 1개 → 답변 1개 | 목표를 여러 단계로 분해하여 순서대로 실행 |
| 상태 관리 | 없음 | `st.session_state`로 단계별 결과 보존 |
| 출력 체이닝 | 없음 | 보고서 → 스크립트 → Q&A 로 연결 |

### 이 앱의 Agent 흐름

```
사용자 입력 (주제, 자료)
        │
        ▼
① 조사 계획 생성    ← Planning 단계
        │
        ▼
② 자료 요약         ← Observation 단계
        │
        ▼
③ 보고서 작성       ← Action 단계 (핵심 산출물)
        │
        ▼
④ 발표 스크립트     ← Transform 단계
        │
        ▼
⑤ 예상 Q&A         ← Verification 단계
```

> 💡 **핵심**: 복잡한 프레임워크 없이도 "단계 분리 + 결과 체이닝"만으로
> 워크플로우형 Agent를 구현할 수 있습니다.
> LangGraph를 배우면 이 흐름을 그래프로 시각화하고 자동화할 수 있습니다.
"""
    )
