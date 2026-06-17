import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

APP_TITLE = "📚 AI 문서/강의자료 튜터"
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


def configure_llama_index():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0.2, api_key=api_key)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL, api_key=api_key)


@st.cache_resource(show_spinner=False)
def build_index_from_pdf(file_bytes: bytes, uploaded_name: str):
    configure_llama_index()
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / uploaded_name
        pdf_path.write_bytes(file_bytes)
        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index


st.title(APP_TITLE)
st.caption("PDF 강의자료를 업로드하고 문서 기반 질문을 해보는 RAG 입문 프로젝트입니다.")

with st.sidebar:
    st.header("수업용 설정")
    st.write(f"LLM: `{LLM_MODEL}`")
    st.write(f"Embedding: `{EMBED_MODEL}`")
    st.info("스캔본 PDF는 텍스트 추출이 안 될 수 있습니다. 이 프로젝트는 우선 텍스트 PDF 기준입니다.")

uploaded_pdf = st.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded_pdf:
    try:
        with st.spinner("문서를 읽고 인덱스를 만드는 중입니다. 업로드 후 첫 실행에만 시간이 걸립니다."):
            index = build_index_from_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
        st.success("문서 인덱스 생성 완료")

        query_engine = index.as_query_engine(similarity_top_k=3)
        question = st.text_input("질문", placeholder="예: 이 문서의 핵심 내용을 요약해줘")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("질문하기", type="primary") and question:
                with st.spinner("답변 생성 중..."):
                    response = query_engine.query(question)
                st.subheader("답변")
                st.write(str(response))
                if getattr(response, "source_nodes", None):
                    st.subheader("참고 문맥")
                    for i, node in enumerate(response.source_nodes, start=1):
                        st.markdown(f"**참고 {i}**")
                        st.write(node.node.get_content()[:500])
        with col2:
            if st.button("문서 요약 생성"):
                with st.spinner("요약 생성 중..."):
                    response = query_engine.query("이 문서를 초보자도 이해할 수 있게 핵심만 요약해줘.")
                st.subheader("문서 요약")
                st.write(str(response))
    except Exception as e:
        st.error("문서 인덱스 생성 중 오류가 발생했습니다.")
        st.exception(e)
else:
    st.info("먼저 PDF를 업로드하세요. 샘플은 `data/sample_lecture.pdf`를 사용하면 됩니다.")
