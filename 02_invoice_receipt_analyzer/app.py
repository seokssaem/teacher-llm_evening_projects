"""
02_invoice_receipt_analyzer/app.py

수업용 프로젝트 2: Invoice / Receipt Analyzer
- 영수증/인보이스 이미지 업로드
- OpenAI Vision으로 구조화된 JSON 추출
- 표 표시
- CSV 저장

실행:
    streamlit run 02_invoice_receipt_analyzer/app.py
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

APP_TITLE = "Invoice / Receipt Analyzer"
DEFAULT_MODEL = "gpt-4o-mini"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    return OpenAI(api_key=api_key)


def encode_image(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def extract_json_from_text(text: str) -> dict[str, Any]:
    """모델 응답에서 JSON 객체를 안전하게 파싱합니다."""
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")

    return json.loads(text[start : end + 1])


def analyze_invoice_image(file_bytes: bytes, mime_type: str) -> dict[str, Any]:
    """이미지에서 인보이스/영수증 정보를 JSON으로 추출합니다."""
    client = get_client()
    image_base64 = encode_image(file_bytes)

    prompt = """
You are an accounting document extraction assistant.
Extract key information from the invoice or receipt image.
Return only valid JSON. Do not include markdown.

JSON schema:
{
  "document_type": "invoice | receipt | unknown",
  "vendor": "string or null",
  "invoice_number": "string or null",
  "document_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "currency": "USD | KRW | EUR | other | null",
  "subtotal": number or null,
  "tax": number or null,
  "total": number or null,
  "payment_method": "string or null",
  "line_items": [
    {
      "description": "string",
      "quantity": number or null,
      "unit_price": number or null,
      "amount": number or null
    }
  ],
  "accounting_notes": ["short note in Korean"]
}

Rules:
- If a field is not visible, use null.
- Use numbers without currency symbols.
- accounting_notes should explain possible bookkeeping use in Korean.
"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        temperature=0,
        max_tokens=900,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
    )

    content = response.choices[0].message.content or "{}"
    return extract_json_from_text(content)


def flatten_for_csv(data: dict[str, Any]) -> pd.DataFrame:
    """문서 단위 정보와 line_items를 CSV 저장용 표로 펼칩니다."""
    base = {
        "document_type": data.get("document_type"),
        "vendor": data.get("vendor"),
        "invoice_number": data.get("invoice_number"),
        "document_date": data.get("document_date"),
        "due_date": data.get("due_date"),
        "currency": data.get("currency"),
        "subtotal": data.get("subtotal"),
        "tax": data.get("tax"),
        "total": data.get("total"),
        "payment_method": data.get("payment_method"),
    }

    line_items = data.get("line_items") or []
    if not line_items:
        return pd.DataFrame([base])

    rows = []
    for item in line_items:
        row = base.copy()
        row.update(
            {
                "item_description": item.get("description"),
                "item_quantity": item.get("quantity"),
                "item_unit_price": item.get("unit_price"),
                "item_amount": item.get("amount"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")
    st.title("🧾 Invoice / Receipt Analyzer")
    st.caption("영수증/인보이스 이미지를 JSON과 CSV로 정리하는 멀티모달 구조화 출력 프로젝트입니다.")

    with st.sidebar:
        st.header("수업용 설정")
        st.write(f"Vision model: `{DEFAULT_MODEL}`")
        st.info("비용 절약을 위해 이미지 detail은 low로 설정했습니다.")

    uploaded_file = st.file_uploader("영수증 또는 인보이스 이미지 업로드", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_file is None:
        st.warning("이미지를 업로드하세요. 샘플은 `data/sample_invoice.png`를 사용하면 됩니다.")
        return

    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type or "image/png"

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("업로드 이미지")
        st.image(file_bytes, use_container_width=True)

    with col2:
        st.subheader("분석")
        analyze = st.button("JSON 추출하기", type="primary")

        if analyze:
            try:
                with st.spinner("이미지를 분석하고 JSON으로 변환하는 중입니다."):
                    data = analyze_invoice_image(file_bytes, mime_type)
            except Exception as exc:
                st.error("이미지 분석 중 오류가 발생했습니다.")
                st.exception(exc)
                return

            st.success("분석 완료")
            st.json(data)

            df = flatten_for_csv(data)
            st.subheader("표 형태")
            st.dataframe(df, use_container_width=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = OUTPUT_DIR / f"invoice_result_{timestamp}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            st.download_button(
                "CSV 다운로드",
                data=df.to_csv(index=False, encoding="utf-8-sig"),
                file_name=csv_path.name,
                mime="text/csv",
            )
            st.caption(f"서버 저장 위치: `{csv_path}`")


if __name__ == "__main__":
    main()
