import importlib
import json
import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.infrastructure.web.api_models import JobStatus
from tests.fakes import FakeAnswerQueryUseCase, FakeQueue, FakeQueuedJob


def build_test_client(queue_error=None, fetch_job_error=None):
    fake_dependencies = types.ModuleType("app.core.dependencies")
    fake_answer_query_use_case = FakeAnswerQueryUseCase()
    fake_dependencies.get_answer_query_use_case = lambda: fake_answer_query_use_case

    fake_queue_module = types.ModuleType("app.infrastructure.queue.rq")
    fake_queue_module.queue = FakeQueue()
    fake_queue_module.jobs = {}
    if queue_error is None:
        fake_queue_module.get_ingestion_queue = lambda: fake_queue_module.queue
    else:
        def get_ingestion_queue():
            raise queue_error

        fake_queue_module.get_ingestion_queue = get_ingestion_queue

    if fetch_job_error is None:
        fake_queue_module.fetch_job = lambda job_id: fake_queue_module.jobs.get(job_id)
    else:
        def fetch_job(_job_id):
            raise fetch_job_error

        fake_queue_module.fetch_job = fetch_job

    sys.modules.pop("app.infrastructure.web.routes", None)
    sys.modules["app.core.dependencies"] = fake_dependencies
    sys.modules["app.infrastructure.queue.rq"] = fake_queue_module

    routes_module = importlib.import_module("app.infrastructure.web.routes")

    app = FastAPI()
    app.include_router(routes_module.router)
    client = TestClient(app)

    return client, fake_answer_query_use_case, fake_queue_module


def test_upload_pdf_enqueues_ingestion_job():
    client, _, fake_queue_module = build_test_client()

    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "job-123",
        "status": JobStatus.UPLOADED_DOCUMENT,
        "filename": "sample.pdf",
    }
    assert fake_queue_module.queue.calls == [
        {
            "job_path": "app.infrastructure.jobs.ingest_pdf_job.ingest_pdf_job",
            "file_bytes": b"%PDF-1.4 test",
            "filename": "sample.pdf",
        }
    ]


def test_upload_rejects_non_pdf_files():
    client, _, _ = build_test_client()

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are allowed."}


def test_upload_returns_503_when_queue_is_unavailable():
    client, _, _ = build_test_client(queue_error=ValueError("redis offline"))

    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Background worker queue is unavailable: redis offline"
    }


def test_ask_returns_answer_payload():
    client, answer_query_use_case, _ = build_test_client()

    response = client.post(
        "/ask",
        json={
            "question": "What is the summary?",
            "top_k": 3,
            "history": [{"role": "user", "content": "Earlier question"}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is the summary?",
        "answer": "Synthesized answer",
        "context": ["Chunk A", "Chunk B"],
        "history": [
            {"role": "user", "content": "Earlier question"},
            {"role": "user", "content": "What is the summary?"},
            {"role": "assistant", "content": "Synthesized answer"},
        ],
    }
    assert answer_query_use_case.answer_calls == [
        {
            "question": "What is the summary?",
            "top_k": 3,
            "history": [{"role": "user", "content": "Earlier question"}],
        }
    ]


def test_get_job_status_returns_finished_job():
    client, _, fake_queue_module = build_test_client()
    fake_queue_module.jobs["job-123"] = FakeQueuedJob(
        status="finished",
        result={
            "filename": "sample.pdf",
            "message": "PDF ingested successfully.",
        },
    )

    response = client.get("/jobs/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "status": JobStatus.DOCUMENT_READY,
        "filename": "sample.pdf",
        "result": {
            "filename": "sample.pdf",
            "message": "PDF ingested successfully.",
        },
        "error": None,
    }


def test_get_job_status_returns_started_label():
    client, _, fake_queue_module = build_test_client()
    fake_queue_module.jobs["job-123"] = FakeQueuedJob(status="started")

    response = client.get("/jobs/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "status": JobStatus.INGESTING_DOC,
        "filename": None,
        "result": None,
        "error": None,
    }


def test_get_job_status_returns_failed_label():
    client, _, fake_queue_module = build_test_client()
    fake_queue_module.jobs["job-123"] = FakeQueuedJob(
        status="failed",
        exc_info="Traceback...\nValueError: parse failed",
    )

    response = client.get("/jobs/job-123")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "status": JobStatus.FAILED,
        "filename": None,
        "result": None,
        "error": "ValueError: parse failed",
    }


def test_get_job_status_returns_503_when_queue_is_unavailable():
    client, _, _ = build_test_client(fetch_job_error=ValueError("redis offline"))

    response = client.get("/jobs/job-123")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Background worker queue is unavailable: redis offline"
    }


def test_ask_stream_emits_tokens_and_done_payload():
    client, answer_query_use_case, _ = build_test_client()

    with client.stream(
        "POST",
        "/ask/stream",
        json={"question": "Stream this", "top_k": 2, "history": []},
    ) as response:
        body = "".join(
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in response.iter_text()
        )

    assert response.status_code == 200
    events = [event for event in body.strip().split("\n\n") if event]
    payloads = [json.loads(event.removeprefix("data: ")) for event in events]

    assert payloads == [
        {"type": "token", "content": "Hello "},
        {"type": "token", "content": "world"},
        {
            "type": "done",
            "payload": {
                "question": "Stream this",
                "answer": "Hello world",
                "context": [],
                "history": [
                    {"role": "user", "content": "Stream this"},
                    {"role": "assistant", "content": "Hello world"},
                ],
            },
        },
    ]
    assert answer_query_use_case.stream_calls == [
        {
            "question": "Stream this",
            "top_k": 2,
            "history": [],
        }
    ]
