from typing import List
import chromadb
from app.domain.entities import EmbeddedChunk, RetreivedChunk

class ChromaVectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
    
    def add(self, chunks: List[EmbeddedChunk]) -> None:
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.vector for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [{"source_document_id": chunk.source_document_id} for chunk in chunks]
        self.collection.add(
            ids=ids, 
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[RetreivedChunk]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"])
        
        retrieved_chunks = [
            RetreivedChunk(
                id=chunk_id,
                text=text,
                source_document_id=metadata.get("source_document_id", "unknown"),
                score=1 / (1 + distance)
            )
            for chunk_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
        
        ranked_retrieved_chunks = self._rerank(retrieved_chunks)
        return ranked_retrieved_chunks
    
    def _rerank(self, chunks: List[RetreivedChunk]) -> List[RetreivedChunk]:
        return sorted(chunks, key=lambda x: x.score, reverse=True)
