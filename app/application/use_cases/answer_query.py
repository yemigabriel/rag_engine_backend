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
    
    def prepare_answer(self, question: str, top_k: int = 10, history = None) -> Answer:
        question = self._rewrite_question(question, history)
        
        query_embedding = self.embedder.embed([question])[0]
        retrieved_chunks: List[EmbeddedChunk] = self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=top_k
        )
        context = [chunk.text for chunk in retrieved_chunks]
        
        return context, question
        
    def answer(self, question: str, top_k: int = 10, history = None) -> Answer:
        context, rewritten_question = self.prepare_answer(question, top_k, history)
        
        answer = self.llm_service.generate_answer(rewritten_question, context, history)
        
        updated_history = (history or []) + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        
        return Answer(
            question=rewritten_question,
            answer=answer,
            context=context,
            history=updated_history
        )
           
    def stream_answer(self, question: str, context: List[str], top_k: int = 10, history = None):
        return self.llm_service.generate_answer_stream(question, context, history or [])
            
    def _rewrite_question(self, question: str, history) -> str:
        if not history:
            return question

        rewritten = self.llm_service.rewrite_question(
            question=question,
            history=history or []
        )
        return rewritten.strip()