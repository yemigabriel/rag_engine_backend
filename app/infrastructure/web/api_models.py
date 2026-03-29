from enum import StrEnum
from typing import Dict, List
from pydantic import BaseModel


class JobStatus(StrEnum):
    UPLOADED_DOCUMENT = "Uploaded document successfully"
    INGESTING_DOC = "Analysing document"
    DOCUMENT_READY = "Document ready"
    FAILED = "Failed"


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    history: List[Dict[str, str]] = []


class AskResponse(BaseModel):
    question: str
    answer: str
    context: List[str]
    history: List[dict]


class UploadJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str | None = None
    result: dict | None = None
    error: str | None = None
