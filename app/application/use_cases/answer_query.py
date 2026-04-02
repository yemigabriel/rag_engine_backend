from typing import Dict, List
from app.domain.entities import Answer, EmbeddedChunk, QueryRewrite
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
    
    def prepare_answer(self, question: str, top_k: int = 10, history = None):
        rewrite_result = self._rewrite_question(question, history)
        
        rewritten_question = rewrite_result.standalone_query
        needs_retrieval = rewrite_result.needs_retrieval
        
        if not needs_retrieval:
            return [], rewritten_question, False
        
        
        query_embedding = self.embedder.embed([question])[0]
        retrieved_chunks: List[EmbeddedChunk] = self.vector_store.retrieve(
            query_embedding=query_embedding,
            top_k=top_k
        )
        context = [chunk.text for chunk in retrieved_chunks]
        
        return context, rewritten_question, True
        
    def answer(self, question: str, top_k: int = 10, history = None) -> Answer:
        context, rewritten_question, needs_retrieval = self.prepare_answer(question, top_k, history)
        if needs_retrieval:
            answer = self.llm_service.generate_answer(rewritten_question, context, history)
        else:
            answer = self.llm_service.generate_conversational_answer(rewritten_question, [], history)
        
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
           
    def stream_answer(self, question: str, top_k: int = 10, history = None):
        context, rewritten_question, needs_retrieval = self.prepare_answer(question, top_k, history)
        if needs_retrieval:
            answer = self.llm_service.generate_answer_stream(rewritten_question, context, history or [])
        else:
            answer = self.llm_service.generate_conversational_answer_stream(rewritten_question, [], history or [])
            
        return answer
            
    def _rewrite_question(self, question: str, history) -> QueryRewrite:
        rewrite_result = self.llm_service.rewrite_question(
            question=question,
            history=history or []
        )
        
        return rewrite_result