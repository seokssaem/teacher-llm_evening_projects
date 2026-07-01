"""
[Step 05] Vision | 이미지 설명 + 해시태그 자동 생성기
================================================
핵심 개념:
  ① Vision  - 이미지를 Base64로 인코딩해 LLM에 전달
  ② Prompt Engineering - 톤/언어/개수를 프롬프트로 제어
  ③ Streamlit - 파일 업로드 + 결과 복사 버튼 UI

실행:
  python -m streamlit run image_tagger.py
"""

from __future__ import annotations

import base64
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

APP_TITLE = "📸 이미지 설명 + 해시태그 자동 생성기"
DEFAULT_MODEL = "gpt-4o-mini"   # Vision 지원 모델


# ── 유틸 함수 ──────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """OpenAI 클라이언트 반환. API 키 없으면 즉시 오류."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    return OpenAI(api_key=api_key)


def image_to_base64(file_bytes: bytes, mime_type: str) -> str:
    """
    이미지 바이트 → Base64 Data URL 변환

    [수업 포인트]
    OpenAI Vision API는 이미지를 직접 받지 않습니다.
    Base64로 인코딩한 뒤 "data:image/jpeg;base64,..." 형태의
    Data URL로 전달해야 합니다. (영수증 분석기와 동일한 패턴!)
    """
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def analyze_image(
    data_url: str,
    tone: str,
    language: str,
    tag_count: int,
) -> dict[str, str]:
    """
    이미지를 분석해 설명문과 해시태그를 반환합니다.

    매개변수:
        data_url  : Base64 Data URL
        tone      : 설명 톤 (예: "친근하고 유머러스하게", "전문적으로")
        language  : 출력 언어 (한국어 / English)
        tag_count : 생성할 해시태그 개수

    반환: {"description": "...", "hashtags": "..."}

    [수업 포인트] 프롬프트로 출력 스타일 제어
    tone, language, tag_count 를 f-string으로 주입하면
    같은 함수로 완전히 다른 느낌의 결과를 만들 수 있습니다.
    """
    client = get_client()

    system_prompt = f"""당신은 SNS 마케팅 카피라이터입니다.
이미지를 보고 {tone} 스타일로 설명문과 해시태그를 {language}로 작성합니다.
반드시 아래 형식으로만 출력하세요. 다른 말은 하지 마세요.

[설명]
(2~3문장 설명)

[해시태그]
(해시태그 {tag_count}개를 #으로 시작하여 스페이스로 구분해서 나열)"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0.7,        # 창의성 살짝 올림 (마케팅 카피니까)
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    # ① 텍스트 지시
                    {"type": "text", "text": "이 이미지를 분석해 주세요."},
                    # ② 이미지 (Base64 Data URL)
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content or ""

    # ── 응답 파싱 ────────────────────────────────────────────────────────
    # "[설명]"과 "[해시태그]" 구분자로 텍스트를 분리합니다.
    description, hashtags = "", ""
    if "[설명]" in raw and "[해시태그]" in raw:
        parts = raw.split("[해시태그]")
        description = parts[0].replace("[설명]", "").strip()
        hashtags = parts[1].strip()
    else:
        # 파싱 실패 시 전체를 설명란에 표시
        description = raw

    return {"description": description, "hashtags": hashtags}


# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title=APP_TITLE, page_icon="📸", layout="wide")
st.title(APP_TITLE)
st.caption("이미지를 업로드하면 SNS용 설명문과 해시태그를 자동으로 생성합니다.")

# ── 사이드바: 옵션 설정 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 생성 옵션")

    tone = st.selectbox(
        "설명 톤",
        options=[
            "친근하고 유머러스하게",
            "전문적이고 신뢰감 있게",
            "감성적이고 따뜻하게",
            "짧고 임팩트 있게",
        ],
    )

    language = st.radio(
        "출력 언어",
        options=["한국어", "English"],
        horizontal=True,
    )

    tag_count = st.slider(
        "해시태그 개수",
        min_value=3,
        max_value=15,
        value=8,
    )

    st.markdown("---")
    st.subheader("📚 수업 포인트")
    st.markdown(
        """
- **Vision**: 이미지 → Base64 → LLM
- **Prompt 제어**: 톤·언어·개수를 변수로 주입
- **파싱**: 구분자로 결과 분리
"""
    )

# ── 메인: 파일 업로드 + 결과 ─────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 이미지 업로드")
    uploaded = st.file_uploader(
        "JPG / PNG / WEBP 이미지를 올려주세요",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        # 업로드된 이미지 미리보기
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

with col2:
    st.subheader("2. 결과")

    if uploaded is None:
        st.info("왼쪽에서 이미지를 업로드하세요.")
    else:
        if st.button("✨ 설명 + 해시태그 생성", type="primary", use_container_width=True):
            # MIME 타입 결정 (확장자 기반)
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/jpeg")

            try:
                with st.spinner("이미지 분석 중..."):
                    file_bytes = uploaded.getvalue()
                    data_url = image_to_base64(file_bytes, mime_type)
                    result = analyze_image(data_url, tone, language, tag_count)

                # ── 결과 출력 ─────────────────────────────────────────────
                st.markdown("#### 📝 설명문")
                st.write(result["description"])

                # 클립보드 복사용 text_area
                st.text_area(
                    "복사용",
                    value=result["description"],
                    height=100,
                    label_visibility="collapsed",
                )

                st.markdown("#### #️⃣ 해시태그")
                st.write(result["hashtags"])
                st.text_area(
                    "복사용",
                    value=result["hashtags"],
                    height=80,
                    label_visibility="collapsed",
                )

            except Exception as e:
                st.error("오류가 발생했습니다.")
                st.exception(e)

# ── 수업용 설명 ────────────────────────────────────────────────────────────
with st.expander("📚 수업용 설명: Vision API 흐름"):
    st.markdown(
        """
### 이미지가 LLM에 전달되는 과정

```
이미지 파일 (JPG/PNG)
      │
      ▼
① 바이트 읽기 (uploaded.getvalue())
      │
      ▼
② Base64 인코딩 (base64.b64encode)
      │
      ▼
③ Data URL 조합 ("data:image/jpeg;base64,...")
      │
      ▼
④ messages 배열에 image_url 타입으로 삽입
      │
      ▼
⑤ GPT-4o-mini Vision이 이미지 + 텍스트 동시 처리
      │
      ▼
⑥ 설명문 + 해시태그 텍스트 반환
```

> 💡 영수증 분석기(Step 02)와 **완전히 동일한 Vision 패턴**입니다.
> 차이점은 Structured Output 대신 구분자 파싱을 사용한 것뿐입니다.
"""
    )
