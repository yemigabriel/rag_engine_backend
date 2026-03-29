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

Run the full test suite with the project virtualenv:

```bash
./.venv/bin/pytest -q tests
```

Or with `uv` if you have it installed:

```bash
uv run pytest -q tests
```

Current test coverage includes:

- unit tests for `IngestPDF`
- unit tests for `AnswerQueryUseCase`
- API route tests for `/upload`, `/ask`, and `/ask/stream`

Test doubles are centralized in `tests/fakes.py`.

## Running The API

```bash
uvicorn app.infrastructure.web.main:app --reload
```

## Running Redis And The Worker Locally

The queue defaults to `redis://127.0.0.1:6379/0`, so local development works without adding `REDIS_URL` as long as Redis is running on your machine.

Start the full local stack with Docker Compose:

```bash
docker compose up --build
```

This starts:

- `redis` on port `6379`
- `web` on port `8000`
- `worker` for background ingestion jobs

If you prefer running app processes outside Docker, you can still start only Redis with:

```bash
docker compose up -d redis
```

Then run the API in one terminal:

```bash
./.venv/bin/uvicorn app.infrastructure.web.main:app --reload
```

And the RQ worker in another terminal:

```bash
./.venv/bin/python worker.py
```

On macOS, the worker defaults to RQ's `SimpleWorker` to avoid native-library `fork()` crashes during PDF ingestion and embedding. You can override that behavior with `RQ_WORKER_CLASS=worker`, `simple`, or `spawn`.

If Redis is down, `/upload` and `/jobs/{job_id}` now return `503` with a clear queue error instead of a generic `500`.

## Railway Deployment

Use two Railway services from this same repo:

- `web` service: use the default `Dockerfile`
- `worker` service: set `RAILWAY_DOCKERFILE_PATH=Dockerfile.worker`

Both services should share the same app environment variables, especially:

- `OPENAI_API_KEY`
- `REDIS_URL`

The worker service does not need a public port. The web service should expose Railway's `$PORT` through the existing uvicorn command.



## API

### `POST /upload`

Uploads a PDF and enqueues background ingestion.

Multipart form field:

- `file`: PDF document

Example:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@document.pdf"
```

Success response: `202 Accepted`

```json
{
  "job_id": "9e39aa29-5679-40d1-b840-53a0945a1aae",
  "status": "Uploaded document successfully",
  "filename": "document.pdf"
}
```

Possible upload errors:

- `400`: non-PDF file or empty upload
- `413`: file exceeds `UPLOAD_MAX_BYTES`
- `503`: background queue is unavailable

### `GET /jobs/{job_id}`

Returns the ingestion job status.

Success response:

```json
{
  "job_id": "9e39aa29-5679-40d1-b840-53a0945a1aae",
  "status": "Document ready",
  "filename": "document.pdf",
  "result": {
    "filename": "document.pdf",
    "message": "PDF ingested successfully."
  },
  "error": null
}
```

If the job fails, `error` contains the last exception line:

```json
{
  "job_id": "9e39aa29-5679-40d1-b840-53a0945a1aae",
  "status": "Failed",
  "filename": null,
  "result": null,
  "error": "ValueError: parse failed"
}
```

Job status labels returned by the API:

- `Uploaded document successfully`: job is queued, deferred, or scheduled
- `Analysing document`: job is currently running
- `Document ready`: job finished successfully
- `Failed`: job failed, was stopped, or was canceled

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

Streams answer tokens as Server-Sent Events and finishes with a `done` payload containing the same schema as `POST /ask`.

Event types:

- `token`: partial answer text
- `error`: terminal error message
- `done`: final structured response payload

Example stream events:

```text
data: {"type":"token","content":"Hello "}

data: {"type":"token","content":"world"}

data: {"type":"done","payload":{"question":"Stream this","answer":"Hello world","context":["Chunk A","Chunk B"],"history":[{"role":"user","content":"Rewritten question"},{"role":"assistant","content":"Hello world"}]}}
```

## Request Flow

### Ingestion

`POST /upload` -> RQ enqueue -> worker runs `ingest_pdf_job()` -> parse -> chunk -> embed -> store -> poll with `GET /jobs/{job_id}`

### Question Answering

`POST /ask` or `POST /ask/stream` -> optional question rewrite from chat history -> embed query -> retrieve top-k chunks -> generate answer

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
