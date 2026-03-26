# RAG Engine Backend

A lightweight Retrieval-Augmented Generation (RAG) backend for ingesting PDFs and answering questions over their contents.

## Overview

The service:

1. Parses uploaded PDFs with Docling
2. Splits extracted text into chunks
3. Embeds chunks with OpenAI `text-embedding-3-small`
4. Stores embeddings in Chroma
5. Retrieves relevant chunks for a query
6. Generates answers with OpenAI `gpt-4o-mini`

The HTTP API is built with FastAPI and supports both standard and streaming responses.

## Architecture

The codebase follows a layered structure:

- `app/domain`: core entities and service contracts
- `app/application/use_cases`: ingestion and question-answering workflows
- `app/infrastructure/services`: concrete adapters for parsing, chunking, embeddings, vector storage, and LLM calls
- `app/infrastructure/web`: FastAPI app, routes, and API models
- `app/core/dependencies.py`: runtime wiring of services and use cases

## Requirements

- Python 3.12+
- OpenAI API key

Set the environment variable:

```bash
export OPENAI_API_KEY=your_api_key
```

## Installation

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -r requirements.txt
```

## Running Tests

Run the full test suite with `uv`:

```bash
uv run pytest -q tests
```

Or with `pytest` directly:

```bash
pytest -q tests
```

Current test coverage includes:

- unit tests for `IngestPDF`
- unit tests for `AnswerQueryUseCase`
- API route tests for `/upload`, `/ask`, and `/ask/stream`

Test doubles are centralized in `tests/fakes.py`, and `pytest` is configured in `pyproject.toml` to resolve imports from the project root.

## Running The API

```bash
uvicorn app.infrastructure.web.main:app --reload
```



## API

### `POST /upload`

Uploads and ingests a PDF file.

Multipart form field:

- `file`: PDF document

Example:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@document.pdf"
```

Response:

```json
{
  "message": "PDF ingested successfully."
}
```

### `POST /ask`

Returns a completed answer for a question.

Request body:

```json
{
  "question": "What is this document about?",
  "top_k": 5,
  "history": []
}
```

Response body:

```json
{
  "question": "What is this document about?",
  "answer": "...",
  "context": ["..."],
  "history": [
    {"role": "user", "content": "What is this document about?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

### `POST /ask/stream`

Streams answer tokens as Server-Sent Events and finishes with a `done` payload containing the same shape as `/ask`.

Event types:

- `token`: partial answer text
- `error`: terminal error message
- `done`: final structured response payload

## Request Flow

### Ingestion

`/upload` -> `IngestPDF.execute()` -> parse -> chunk -> embed -> store

### Question Answering

`/ask` or `/ask/stream` -> optional question rewrite from chat history -> embed query -> retrieve top-k chunks -> generate answer

## Key Implementation Details

- PDF parsing: Docling
- Chunking: LangChain `RecursiveCharacterTextSplitter`
- Embeddings model: OpenAI `text-embedding-3-small`
- Answer model: OpenAI `gpt-4o-mini`
- Vector store: Chroma collection `documents`
- Follow-up questions can be rewritten into standalone questions using prior chat history

## Development Notes

- Dependency wiring happens at import time in `app/core/dependencies.py`
- The API entry point is `app/infrastructure/web/main.py`
- Tests live under `tests/`
