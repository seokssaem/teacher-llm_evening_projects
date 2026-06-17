"""
01_ai_doc_tutor/app.py

수업용 프로젝트 1: AI 문서/강의자료 튜터
- PDF 업로드
- LlamaIndex로 RAG 인덱스 생성
- 질문/답변
- 근거 페이지 표시

실행:
    streamlit run 01_ai_doc_tutor/app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


load_dotenv()


APP_TITLE = "AI 문서/강의자료 튜터"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@st.cache_resource(show_spinner=False)
def configure_llama_index() -> None:
    """LlamaIndex 전역 설정을 한 번만 구성합니다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")

    Settings.llm = OpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        max_tokens=700,
        api_key=api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=DEFAULT_EMBED_MODEL,
        api_key=api_key,
    )


@st.cache_resource(show_spinner=False)
def build_index_from_pdf(file_bytes: bytes, filename: str) -> VectorStoreIndex:
    """업로드된 PDF 바이트를 임시 파일로 저장한 뒤 RAG 인덱스를 생성합니다."""
    configure_llama_index()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / filename
        pdf_path.write_bytes(file_bytes)

        documents = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()

        if not documents:
            raise ValueError("PDF에서 텍스트를 읽지 못했습니다. 스캔본 PDF라면 OCR/LlamaParse가 필요합니다.")

        index = VectorStoreIndex.from_documents(documents)
        return index


def format_source_info(response) -> list[dict]:
    """LlamaIndex 응답에서 근거 정보를 화면 표시용으로 정리합니다."""
    results = []
    source_nodes = getattr(response, "source_nodes", []) or []

    for i, node in enumerate(source_nodes, start=1):
        metadata = node.metadata or {}
        page_label = metadata.get("page_label") or metadata.get("page") or "페이지 정보 없음"
        file_name = metadata.get("file_name") or metadata.get("filename") or "업로드 문서"
        score = getattr(node, "score", None)
        text = node.get_content() if hasattr(node, "get_content") else str(node)

        results.append(
            {
                "rank": i,
                "file_name": file_name,
                "page": page_label,
                "score": round(score, 4) if isinstance(score, float) else None,
                "preview": text[:500].replace("\n", " "),
            }
        )
    return results


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
    st.title("📚 AI 문서/강의자료 튜터")
    st.caption("PDF 강의자료를 업로드하고 문서 기반 질문을 해보는 RAG 입문 프로젝트입니다.")

    with st.sidebar:
        st.header("수업용 설정")
        st.write(f"LLM: `{DEFAULT_MODEL}`")
        st.write(f"Embedding: `{DEFAULT_EMBED_MODEL}`")
        st.info("스캔본 PDF는 텍스트 추출이 안 될 수 있습니다. 이 프로젝트는 우선 텍스트 PDF 기준입니다.")

    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])

    if uploaded_file is None:
        st.warning("먼저 PDF를 업로드하세요. 샘플은 `data/sample_lecture.pdf`를 사용하면 됩니다.")
        return

    file_bytes = uploaded_file.getvalue()

    try:
        with st.spinner("문서를 읽고 인덱스를 만드는 중입니다. 업로드 후 첫 실행에만 시간이 걸립니다."):
            index = build_index_from_pdf(file_bytes, uploaded_file.name)
    except Exception as exc:
        st.error("문서 인덱스 생성 중 오류가 발생했습니다.")
        st.exception(exc)
        return

    st.success("문서 인덱스 생성 완료")

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "질문 입력",
            value="이 문서의 핵심 내용을 초보자도 이해할 수 있게 요약해줘.",
            height=100,
        )
        ask = st.button("질문하기", type="primary")

    with col2:
        top_k = st.slider("검색할 근거 수", min_value=1, max_value=5, value=3)
        st.caption("값이 클수록 근거는 많아지지만 토큰 사용량도 늘어납니다.")

    if ask:
        if not question.strip():
            st.warning("질문을 입력하세요.")
            return

        query_engine = index.as_query_engine(similarity_top_k=top_k)

        with st.spinner("문서에서 근거를 찾고 답변을 생성하는 중입니다."):
            response = query_engine.query(question)

        st.subheader("답변")
        st.write(str(response))

        st.subheader("근거 문서")
        sources = format_source_info(response)

        if not sources:
            st.info("표시할 근거 정보가 없습니다.")
        else:
            for source in sources:
                with st.expander(f"근거 {source['rank']} - {source['file_name']} / page: {source['page']}"):
                    if source["score"] is not None:
                        st.write(f"유사도 점수: `{source['score']}`")
                    st.write(source["preview"])


if __name__ == "__main__":
    main()
