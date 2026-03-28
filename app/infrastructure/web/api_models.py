from typing import Dict, List
from pydantic import BaseModel


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
    status: str
    filename: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str | None = None
    result: dict | None = None
    error: str | None = None
