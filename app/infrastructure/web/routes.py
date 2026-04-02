import json

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError

from app.core.dependencies import get_answer_query_use_case
from app.core.settings import settings
from app.infrastructure.queue.rq import fetch_job, get_ingestion_queue
from app.infrastructure.web.api_models import (
    AskRequest,
    AskResponse,
    JobStatus,
    JobStatusResponse,
    UploadJobResponse,
)

router = APIRouter()

JOB_STATUS_MAP = {
    "queued": JobStatus.UPLOADED_DOCUMENT,
    "deferred": JobStatus.UPLOADED_DOCUMENT,
    "scheduled": JobStatus.UPLOADED_DOCUMENT,
    "started": JobStatus.INGESTING_DOC,
    "finished": JobStatus.DOCUMENT_READY,
    "failed": JobStatus.FAILED,
    "stopped": JobStatus.FAILED,
    "canceled": JobStatus.FAILED,
}


def _queue_unavailable_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"Background worker queue is unavailable: {exc}",
    )


def _map_job_status(status: str) -> JobStatus:
    return JOB_STATUS_MAP.get(status, JobStatus.INGESTING_DOC)

@router.post("/upload", response_model=UploadJobResponse, status_code=202)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and enqueue it for background ingestion."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
        if len(file_bytes) > settings.upload_max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"PDF exceeds max upload size of {settings.upload_max_bytes} bytes.",
            )

        try:
            job = get_ingestion_queue().enqueue(
                "app.infrastructure.jobs.ingest_pdf_job.ingest_pdf_job",
                file_bytes,
                file.filename,
            )
        except (RedisError, ValueError) as exc:
            raise _queue_unavailable_error(exc) from exc

        return UploadJobResponse(
            job_id=job.id,
            status=_map_job_status(job.get_status(refresh=True)),
            filename=file.filename,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    try:
        job = fetch_job(job_id)
    except (RedisError, ValueError) as exc:
        raise _queue_unavailable_error(exc) from exc

    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    error = None
    if job.is_failed and job.exc_info:
        error = job.exc_info.strip().splitlines()[-1]

    result = job.result if isinstance(job.result, dict) else None
    filename = None
    if result:
        filename = result.get("filename")

    return JobStatusResponse(
        job_id=job.id,
        status=_map_job_status(job.get_status(refresh=True)),
        filename=filename,
        result=result,
        error=error,
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Endpoint to ask a question based on ingested PDFs."""
    try:
        result = get_answer_query_use_case().answer(
            question=request.question,
            top_k=request.top_k,
            history=request.history
        )

        updated_history = (request.history or []) + [
            {"role": "user", "content": result.question},
            {"role": "assistant", "content": result.answer},
        ]

        return AskResponse(
            question=result.question,
            answer=result.answer,
            context=result.context,
            history=updated_history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/ask/stream")
async def ask_question_stream(request: AskRequest):
    """Endpoint to ask a question and yield a streaming response."""
    def generate():
        full_answer = ""
        try:
            answer_query_use_case = get_answer_query_use_case()
            print("Preparing answer...")
            context, rewritten_question = answer_query_use_case.prepare_answer(
                question=request.question,
                top_k=request.top_k,
                history=request.history
            )

            print(f"Rewritten question: {rewritten_question}, Context chunks: {len(context)}")

            for chunk in answer_query_use_case.stream_answer(
                question=request.question,
                context=context,
                top_k=request.top_k,
                history=request.history
            ):
                full_answer += chunk

                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        updated_history = (request.history or []) + [
            {"role": "user", "content": rewritten_question},
            {"role": "assistant", "content": full_answer},
        ]

        final_payload = AskResponse(
            question=request.question,
            answer=full_answer,
            context=context,
            history=updated_history
        ).model_dump()

        yield f"data: {json.dumps({'type': 'done', 'payload': final_payload})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
