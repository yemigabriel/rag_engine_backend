from dataclasses import dataclass
from typing import List


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
    source_documents: List[Document]

@dataclass
class Query:
    question: str
