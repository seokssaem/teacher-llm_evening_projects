# LLM 저녁반 수업용 프로젝트 예제 v2

석쌤 저녁반에서 바로 검증하고 커스텀할 수 있도록 만든 최소 의존성 프로젝트입니다.

## 포함 프로젝트

| 폴더 | 프로젝트 | 핵심 |
|---|---|---|
| `01_ai_doc_tutor` | AI 문서/강의자료 튜터 | PDF RAG 입문, LlamaIndex 사용 |
| `02_invoice_receipt_analyzer` | Invoice / Receipt Analyzer | 이미지 → JSON → CSV, Structured Output |
| `03_multimodal_lecture_tutor_v2` | 멀티모달 강의자료 AI 튜터 V2 | PDF + 이미지 + 요약/퀴즈/Q&A |

## 설치

```bash
uv sync
cp .env.example .env
```

`.env` 파일에 OpenAI API 키를 넣습니다.

```env
OPENAI_API_KEY=sk-...
```

## 실행

Windows/Git Bash/VS Code 터미널에서는 아래 방식이 가장 안전합니다.

```bash
python -m streamlit run 01_ai_doc_tutor/app.py
python -m streamlit run 02_invoice_receipt_analyzer/app.py
python -m streamlit run 03_multimodal_lecture_tutor_v2/app.py
```

## 수업 운영 팁

- `01_ai_doc_tutor`: RAG 흐름 설명용입니다. OpenAI embedding API를 사용하므로 잔액/Rate Limit 영향을 받습니다.
- `02_invoice_receipt_analyzer`: 멀티모달 + 구조화 출력 실습용입니다.
- `03_multimodal_lecture_tutor_v2`: 수업 안정형 프로젝트입니다. PDF 검색은 로컬 키워드 방식으로 먼저 처리하고, 최종 요약/답변/퀴즈 생성에만 OpenAI를 호출합니다.

## 권장 수업 순서

1. `02_invoice_receipt_analyzer`로 이미지 → JSON의 재미를 먼저 보여줍니다.
2. `01_ai_doc_tutor`로 RAG 기본 구조를 설명합니다.
3. `03_multimodal_lecture_tutor_v2`로 최종 프로젝트형 앱을 보여줍니다.
