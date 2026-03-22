from typing import Dict, List
from app.domain.entities import Answer, EmbeddedChunk
from app.domain.interfaces import Embedder, LLMService, VectorStore


class AnswerQueryUseCase:
    def __init__(
        self, 
        embedder: Embedder, 
        vector_store: VectorStore, 
        llm_service: LLMService
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_service = llm_service
    
    def execute(self, question: str, top_k: int = 3, history = None) -> Answer:
        question = self._rewrite_question(question, history)
        
        query_embedding = self.embedder.embed([question])[0]
        retrieved_chunks: List[EmbeddedChunk] = self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=top_k
        )
        context = [chunk.text for chunk in retrieved_chunks]
        
        response = self.llm_service.generate_answer(
            question=question,
            context=context,
            history=history or []
        )
        
        updated_history = (history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]

        return Answer(
            question=question,
            answer=response,
            context=context,
            history=updated_history
        )
        
    def _rewrite_question(self, question: str, history) -> str:
        if not history:
            return question

        rewritten = self.llm_service.rewrite_question(
            question=question,
            history=history or []
        )
        return rewritten.strip()