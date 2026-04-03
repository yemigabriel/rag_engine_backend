from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field


@dataclass
class Document:
    id: str
    content: str

@dataclass
class Chunk:
    id: str
    text: str
    source_document_id: str

@dataclass
class EmbeddedChunk:
    id: str
    text: str
    vector: List[float]
    source_document_id: str

@dataclass
class RetreivedChunk:
    id: str
    text: str
    source_document_id: str
    score: float

@dataclass
class Answer:
    answer: str
    question: str
    context: List[str]
    history: List[dict]

@dataclass
class Query:
    question: str

class QueryRewrite(BaseModel):
    needs_retrieval: bool = Field(
        description="Whether a database search is required to answer the question."
    )
    standalone_query: str = Field(
        description="The rewritten query, or the original query if no rewrite is needed."
    )