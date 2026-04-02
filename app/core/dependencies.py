from functools import lru_cache

from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_pdf import IngestPDF
from app.core.settings import settings
from app.infrastructure.services.llm_service import OpenAILLMService
from app.infrastructure.services.chunker import LangChainChunker
from app.infrastructure.services.openai_embedder import OpenAIEmbedder
from app.infrastructure.services.pdf_parser import DoclingParser
from app.infrastructure.services.vectore_store import ChromaVectorStore, PineconeVectorStore

settings.validate()


@lru_cache
def get_parser() -> DoclingParser:
    return DoclingParser()


@lru_cache
def get_chunker() -> LangChainChunker:
    return LangChainChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model_name=settings.openai_embedding_model,
    )


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    if settings.vector_store_backend == "pinecone":
        return PineconeVectorStore(
            api_key=settings.pinecone_api_key,
            index_host=settings.pinecone_index_host,
            namespace=settings.pinecone_namespace,
        )
    return ChromaVectorStore(
        collection_name=settings.chroma_collection_name,
        backend=settings.vector_store_backend,
        persist_path=settings.chroma_persist_path,
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
    )

@lru_cache
def get_llm_service() -> OpenAILLMService:
    return OpenAILLMService(
        api_key=settings.openai_api_key,
        model_name=settings.openai_chat_model,
    )


@lru_cache
def get_ingest_pdf_use_case() -> IngestPDF:
    return IngestPDF(
        parser=get_parser(),
        chunker=get_chunker(),
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_answer_query_use_case() -> AnswerQueryUseCase:
    return AnswerQueryUseCase(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
    )
