from typing import List
import os

import chromadb

from app.domain.entities import EmbeddedChunk, RetreivedChunk


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        backend: str = "persistent",
        persist_path: str = "./data/chroma",
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
    ):
        if backend == "memory":
            self.client = chromadb.Client()
        elif backend == "persistent":
            os.makedirs(persist_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_path)
        elif backend == "http":
            if not host:
                raise ValueError("host is required for Chroma HTTP mode")
            self.client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
        else:
            raise ValueError(f"Unsupported Chroma backend: {backend}")

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


class PineconeVectorStore:
    def __init__(self, api_key: str, index_host: str, namespace: str = "default"):
        try:
            from pinecone import Pinecone
        except ImportError as exc:
            raise ImportError(
                "pinecone is required when VECTOR_STORE_BACKEND=pinecone."
            ) from exc

        self.namespace = namespace
        self.index = Pinecone(api_key=api_key).Index(host=index_host)

    def add(self, chunks: List[EmbeddedChunk]) -> None:
        vectors = [
            {
                "id": chunk.id,
                "values": chunk.vector,
                "metadata": {
                    "text": chunk.text,
                    "source_document_id": chunk.source_document_id,
                },
            }
            for chunk in chunks
        ]
        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)

    def retrieve(self, query_embedding: List[float], top_k: int) -> List[RetreivedChunk]:
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )
        matches = getattr(results, "matches", None)
        if matches is None and isinstance(results, dict):
            matches = results.get("matches", [])
        matches = matches or []

        retrieved_chunks = [
            RetreivedChunk(
                id=self._read_match_field(match, "id"),
                text=self._read_match_metadata(match).get("text", ""),
                source_document_id=self._read_match_metadata(match).get(
                    "source_document_id",
                    "unknown",
                ),
                score=float(self._read_match_field(match, "score", 0.0)),
            )
            for match in matches
        ]
        return self._rerank(retrieved_chunks)

    def _rerank(self, chunks: List[RetreivedChunk]) -> List[RetreivedChunk]:
        return sorted(chunks, key=lambda x: x.score, reverse=True)

    @staticmethod
    def _read_match_field(match, field: str, default=None):
        if isinstance(match, dict):
            return match.get(field, default)
        return getattr(match, field, default)

    @classmethod
    def _read_match_metadata(cls, match) -> dict:
        metadata = cls._read_match_field(match, "metadata", {})
        return metadata if isinstance(metadata, dict) else {}
