
from app.domain.entities import EmbeddedChunk
from app.domain.interfaces import Chunker, DocumentParser, Embedder, VectorStore


class IngestPDF:
    """Use case for ingesting a PDF document, parsing it, chunking the content, 
    embedding the chunks, and storing them in a vector store."""
    def __init__(self, 
                 parser: DocumentParser, 
                 chunker: Chunker, 
                 embedder: Embedder, 
                 vector_store: VectorStore):
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
    
    def execute(self, document_path: str) -> None:
        content = self.parser.parse(document_path)
        chunks = self.chunker.chunk(content)
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        embedded_chunks = [
            EmbeddedChunk(id=chunk.id, text=chunk.text, vector=embedding, source_document_id=chunk.source_document_id)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.vector_store.add(embedded_chunks)
        