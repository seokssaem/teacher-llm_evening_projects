import os
import json
import base64
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

APP_TITLE = "🧾 Invoice / Receipt Analyzer"
MODEL = "gpt-4o-mini"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    return OpenAI(api_key=api_key)


def image_to_data_url(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = uploaded_file.type or "image/png"
    return f"data:{mime};base64,{b64}"


def analyze_invoice_image(uploaded_file) -> dict:
    client = get_client()
    data_url = image_to_data_url(uploaded_file)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "invoice_receipt_result",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "document_type": {"type": "string", "description": "invoice or receipt or unknown"},
                    "vendor": {"type": "string"},
                    "invoice_number": {"type": "string"},
                    "invoice_date": {"type": "string"},
                    "due_date": {"type": "string"},
                    "currency": {"type": "string"},
                    "subtotal": {"type": "number"},
                    "tax": {"type": "number"},
                    "total": {"type": "number"},
                    "payment_method": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "description": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price": {"type": "number"},
                                "amount": {"type": "number"}
                            },
                            "required": ["description", "quantity", "unit_price", "amount"]
                        }
                    },
                    "notes": {"type": "string"}
                },
                "required": ["document_type", "vendor", "invoice_number", "invoice_date", "due_date", "currency", "subtotal", "tax", "total", "payment_method", "items", "notes"]
            }
        }
    }
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이미지 속 영수증 또는 인보이스를 읽고 정확한 JSON으로 추출해줘. 보이지 않는 값은 빈 문자열 또는 0으로 둬."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        response_format=schema,
        temperature=0,
        max_tokens=1600,
    )
    return json.loads(response.choices[0].message.content)


st.title(APP_TITLE)
st.caption("영수증/인보이스 이미지를 JSON과 CSV로 정리하는 멀티모달 구조화 출력 프로젝트입니다.")

with st.sidebar:
    st.header("수업용 설정")
    st.write(f"LLM: `{MODEL}`")
    st.info("핵심 개념: Vision + Structured Output + Pandas 저장")

uploaded = st.file_uploader("영수증 또는 인보이스 이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("업로드 이미지")
        st.image(uploaded, use_container_width=True)

    with col2:
        st.subheader("분석")
        if st.button("JSON 추출하기", type="primary"):
            try:
                with st.spinner("이미지를 분석하는 중입니다..."):
                    result = analyze_invoice_image(uploaded)
                st.success("분석 완료")
                st.json(result)

                summary_df = pd.DataFrame([{
                    "document_type": result.get("document_type"),
                    "vendor": result.get("vendor"),
                    "invoice_number": result.get("invoice_number"),
                    "invoice_date": result.get("invoice_date"),
                    "due_date": result.get("due_date"),
                    "currency": result.get("currency"),
                    "subtotal": result.get("subtotal"),
                    "tax": result.get("tax"),
                    "total": result.get("total"),
                    "payment_method": result.get("payment_method"),
                }])
                items_df = pd.DataFrame(result.get("items", []))

                summary_path = OUTPUT_DIR / "invoice_summary.csv"
                items_path = OUTPUT_DIR / "invoice_items.csv"
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
                items_df.to_csv(items_path, index=False, encoding="utf-8-sig")

                st.subheader("요약 표")
                st.dataframe(summary_df, use_container_width=True)
                st.subheader("항목 표")
                st.dataframe(items_df, use_container_width=True)
                st.caption(f"CSV 저장 완료: {summary_path}, {items_path}")
            except Exception as e:
                st.error("이미지 분석 중 오류가 발생했습니다.")
                st.exception(e)
else:
    st.info("샘플 이미지는 `data/sample_invoice.png`를 사용하세요.")
