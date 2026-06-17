# LLM 멀티모달 저녁반 검증 프로젝트 2종

## 프로젝트 구성

```text
llm_evening_projects/
├─ pyproject.toml
├─ .env.example
├─ data/
│  ├─ sample_lecture.pdf
│  └─ sample_invoice.png
├─ outputs/
├─ 01_ai_doc_tutor/
│  └─ app.py
└─ 02_invoice_receipt_analyzer/
   └─ app.py
```

## 설치

```bash
uv python pin 3.11
uv venv --python 3.11
source .venv/Scripts/activate
uv sync
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## API 키 설정

`.env.example` 파일을 복사해서 `.env`로 이름을 바꾸고 API 키를 입력합니다.

```bash
cp .env.example .env
```

Windows에서 Git Bash 사용 시:

```bash
cp .env.example .env
```

## 실행 1: AI 문서/강의자료 튜터

```bash
streamlit run 01_ai_doc_tutor/app.py
```

샘플 파일: `data/sample_lecture.pdf`

## 실행 2: Invoice / Receipt Analyzer

```bash
streamlit run 02_invoice_receipt_analyzer/app.py
```

샘플 파일: `data/sample_invoice.png`

## 수업 운영 원칙

- Python 3.11 고정
- uv 사용
- OpenAI API만 사용
- Streamlit으로 화면 구성
- LlamaIndex는 PDF RAG에만 최소 사용
- Chroma, FAISS, Qdrant, CLIP, Ollama, Gemini, Claude, Cohere는 이번 검증 프로젝트에서 제외

## 비용 절약 설정

- 기본 모델: `gpt-4o-mini`
- 답변 길이 제한
- 질문 버튼을 눌렀을 때만 API 호출
- PDF 인덱스는 업로드 후 한 번만 생성
