"""
04 Research Agent
수업용 안정 버전: 실시간 웹검색 API 없이, 사용자가 붙여넣은 자료를 기반으로
조사 계획 → 핵심 요약 → 보고서 → 발표 스크립트 → 예상 Q&A를 생성합니다.

실행:
python -m streamlit run 04_research_agent/app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "04. AI Research Agent"
DEFAULT_MODEL = "gpt-4o-mini"


# -----------------------------
# 기본 유틸
# -----------------------------
def get_client() -> OpenAI | None:
    """OPENAI_API_KEY가 있으면 OpenAI 클라이언트를 반환합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """OpenAI Chat Completions API 호출. 수업 안정성을 위해 한 함수로 모았습니다."""
    client = get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def safe_json_loads(text: str) -> dict[str, Any]:
    """모델 응답에서 JSON 부분만 최대한 안전하게 파싱합니다."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": text}


def make_download_text(topic: str, report: str, script: str, qa: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
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


# -----------------------------
# 프롬프트
# -----------------------------
def build_plan_prompt(topic: str, audience: str, goal: str) -> tuple[str, str]:
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
    return system, user


def build_summary_prompt(topic: str, pasted_sources: str) -> tuple[str, str]:
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


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🕵️", layout="wide")
st.title("🕵️ AI Research Agent")
st.caption("수업 안정형 버전: 실시간 웹검색 없이, 붙여넣은 자료 기반으로 보고서를 생성합니다.")

with st.sidebar:
    st.header("실행 상태")
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 조사 설정")
    topic = st.text_input("조사 주제", value="생성형 AI가 교육 분야에 미치는 영향")
    audience = st.text_input("대상 독자/청중", value="AI 입문 수강생")
    goal = st.text_input("보고서 목적", value="수업 발표와 포트폴리오 정리")

    st.subheader("2. 자료 붙여넣기")
    pasted_sources = st.text_area(
        "뉴스, 블로그, 논문 초록, 회사 자료 등을 붙여넣으세요",
        height=300,
        placeholder="여기에 조사 자료를 붙여넣으세요.\n예: 기사 요약, 공식 문서 일부, 통계 자료, 블로그 내용 등",
    )

with col2:
    st.subheader("3. Agent 실행")
    st.info("처음에는 '조사 계획 생성'만 눌러도 수업 흐름 설명이 가능합니다.")

    plan_btn = st.button("① 조사 계획 생성", use_container_width=True)
    summary_btn = st.button("② 붙여넣은 자료 요약", use_container_width=True)
    report_btn = st.button("③ 최종 보고서 생성", use_container_width=True)
    full_btn = st.button("④ 보고서 + 발표 + Q&A 한번에 생성", use_container_width=True)

if "plan" not in st.session_state:
    st.session_state.plan = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "report" not in st.session_state:
    st.session_state.report = ""
if "script" not in st.session_state:
    st.session_state.script = ""
if "qa" not in st.session_state:
    st.session_state.qa = ""

try:
    if plan_btn:
        with st.spinner("조사 계획 생성 중..."):
            system, user = build_plan_prompt(topic, audience, goal)
            st.session_state.plan = call_llm(system, user)

    if summary_btn:
        if not pasted_sources.strip():
            st.warning("먼저 자료를 붙여넣어 주세요.")
        else:
            with st.spinner("자료 요약 중..."):
                system, user = build_summary_prompt(topic, pasted_sources)
                st.session_state.summary = call_llm(system, user)

    if report_btn:
        if not pasted_sources.strip():
            st.warning("먼저 자료를 붙여넣어 주세요.")
        else:
            with st.spinner("보고서 생성 중..."):
                system, user = build_report_prompt(topic, audience, goal, pasted_sources)
                st.session_state.report = call_llm(system, user)

    if full_btn:
        if not pasted_sources.strip():
            st.warning("먼저 자료를 붙여넣어 주세요.")
        else:
            with st.spinner("보고서 생성 중..."):
                system, user = build_report_prompt(topic, audience, goal, pasted_sources)
                st.session_state.report = call_llm(system, user)
            with st.spinner("발표 스크립트 생성 중..."):
                system, user = build_script_prompt(topic, st.session_state.report)
                st.session_state.script = call_llm(system, user)
            with st.spinner("예상 Q&A 생성 중..."):
                system, user = build_qa_prompt(topic, st.session_state.report)
                st.session_state.qa = call_llm(system, user)

except Exception as e:
    st.error("실행 중 오류가 발생했습니다.")
    st.exception(e)

st.markdown("---")
st.subheader("결과")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["조사 계획", "자료 요약", "최종 보고서", "발표 스크립트", "예상 Q&A"])

with tab1:
    if st.session_state.plan:
        st.markdown(st.session_state.plan)
    else:
        st.caption("조사 계획이 아직 없습니다.")

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

if st.session_state.report or st.session_state.script or st.session_state.qa:
    download_text = make_download_text(
        topic=topic,
        report=st.session_state.report,
        script=st.session_state.script,
        qa=st.session_state.qa,
    )
    st.download_button(
        "Markdown 보고서 다운로드",
        data=download_text,
        file_name="research_agent_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

with st.expander("수업용 설명: 이게 왜 Agent인가요?"):
    st.markdown(
        """
이 예제는 브라우저를 직접 조작하거나 복잡한 Agent 프레임워크를 쓰지는 않습니다.  
하지만 하나의 요청을 아래 단계로 나눠 처리합니다.

1. 조사 계획 생성  
2. 자료 요약  
3. 보고서 작성  
4. 발표 스크립트 작성  
5. 예상 질문 생성  

즉, 단순 챗봇이 아니라 **업무 흐름을 단계별로 수행하는 워크플로우형 Agent 입문 예제**입니다.
"""
    )
