from typing import List, Protocol
from app.domain.entities import Chunk, EmbeddedChunk, RetreivedChunk


class DocumentParser(Protocol):
    def parse(self, document_path: str) -> str:
        """Parse a document and return its content as a string."""
        ...

class Chunker(Protocol):
    def chunk(self, text: str) -> List[Chunk]:
        """Chunk the input text into smaller pieces."""
        ...

class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert texts into a vector representation."""
        ...

class VectorStore(Protocol):
    def add(self, chunks: List[EmbeddedChunk]) -> None:
        """Add embedded chunks."""
        ...
    
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[RetreivedChunk]:
        """Retrieve relevant documents based on the query embedding."""
        ...

class LLMService(Protocol):
    def generate_answer(self, question: str, context: List[str]) -> str:
        """Generate an answer based on the question and provided context."""
        ...
        