from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_pdf import IngestPDF
from app.core.settings import settings
from app.infrastructure.services.llm_service import OpenAILLMService
from app.infrastructure.services.chunker import LangChainChunker
from app.infrastructure.services.openai_embedder import OpenAIEmbedder
from app.infrastructure.services.pdf_parser import DoclingParser
from app.infrastructure.services.vectore_store import ChromaVectorStore

settings.validate()

parser = DoclingParser()
chunker = LangChainChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
embedder = OpenAIEmbedder(
    api_key=settings.openai_api_key,
    model_name=settings.openai_embedding_model,
)
vector_store = ChromaVectorStore(collection_name=settings.chroma_collection_name)
llm_service = OpenAILLMService(
    api_key=settings.openai_api_key,
    model_name=settings.openai_chat_model,
)

ingest_pdf_use_case = IngestPDF(
    parser=parser,
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store
)

answer_query_use_case = AnswerQueryUseCase(
    embedder=embedder, 
    vector_store=vector_store,
    llm_service=llm_service
)
