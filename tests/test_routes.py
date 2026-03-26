import importlib
import json
import sys
import types
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fakes import FakeAnswerQueryUseCase, FakeIngestPDFUseCase


def build_test_client():
    fake_dependencies = types.ModuleType("app.core.dependencies")
    fake_dependencies.ingest_pdf_use_case = FakeIngestPDFUseCase()
    fake_dependencies.answer_query_use_case = FakeAnswerQueryUseCase()

    sys.modules.pop("app.infrastructure.web.routes", None)
    sys.modules["app.core.dependencies"] = fake_dependencies

    routes_module = importlib.import_module("app.infrastructure.web.routes")

    app = FastAPI()
    app.include_router(routes_module.router)
    client = TestClient(app)

    return client, fake_dependencies


def test_upload_pdf_calls_ingest_use_case():
    client, dependencies = build_test_client()

    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "PDF ingested successfully."}
    assert len(dependencies.ingest_pdf_use_case.calls) == 1
    assert dependencies.ingest_pdf_use_case.calls[0].endswith(".pdf")


def test_upload_rejects_non_pdf_files():
    client, _ = build_test_client()

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are allowed."}


def test_ask_returns_answer_payload():
    client, dependencies = build_test_client()

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
    assert dependencies.answer_query_use_case.answer_calls == [
        {
            "question": "What is the summary?",
            "top_k": 3,
            "history": [{"role": "user", "content": "Earlier question"}],
        }
    ]


def test_ask_stream_emits_tokens_and_done_payload():
    client, dependencies = build_test_client()

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
                "context": ["Chunk A", "Chunk B"],
                "history": [
                    {"role": "user", "content": "Rewritten question"},
                    {"role": "assistant", "content": "Hello world"},
                ],
            },
        },
    ]
    assert dependencies.answer_query_use_case.prepare_calls == [
        {"question": "Stream this", "top_k": 2, "history": []}
    ]
    assert dependencies.answer_query_use_case.stream_calls == [
        {
            "question": "Stream this",
            "context": ["Chunk A", "Chunk B"],
            "top_k": 2,
            "history": [],
        }
    ]
