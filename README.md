# RAG Engine Backend

A FastAPI backend for uploading PDFs, processing them asynchronously, and answering questions over their contents with RAG (retrieval-augmented generation).

## Client App

The companion Swift iOS app for this backend is [DocAsk](https://github.com/yemigabriel/DocAsk).

## What It Does

- accepts PDF uploads
- queues ingestion with Redis + RQ
- parses documents with Docling
- chunks and embeds content with OpenAI
- stores vectors in Chroma or Pinecone
- answers questions with standard and streaming responses

## Stack

- Python
- FastAPI
- Redis + RQ
- Docling
- OpenAI
- Chroma / Pinecone
- Docker Compose

## Run

Add `OPENAI_API_KEY` to your `.env`, then start the app with:

```bash
docker compose up --build
```

The API will be available at `http://127.0.0.1:8000`.

## API

### `POST /upload`

Uploads a PDF and returns a background job id.

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@document.pdf"
```

Example response:

```json
{
  "job_id": "9e39aa29-5679-40d1-b840-53a0945a1aae",
  "status": "Uploaded document successfully",
  "filename": "document.pdf"
}
```

### `GET /jobs/{job_id}`

Returns the ingestion job status.

Example response:

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

### `POST /ask`

Returns a complete answer for a question.

```json
{
  "question": "What is this document about?",
  "top_k": 5,
  "history": []
}
```

Example response:

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

Streams answer tokens as Server-Sent Events and finishes with a `done` event.

```text
data: {"type":"token","content":"Hello "}

data: {"type":"token","content":"world"}

data: {"type":"done","payload":{"question":"Stream this","answer":"Hello world","context":["Chunk A","Chunk B"],"history":[{"role":"user","content":"Rewritten question"},{"role":"assistant","content":"Hello world"}]}}
```

## Notes

- background ingestion runs through a separate worker
- follow-up questions can be rewritten into standalone queries
- the main API entry point is `app/infrastructure/web/main.py`
- the worker entry point is `worker.py`
