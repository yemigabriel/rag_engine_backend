import os
from dotenv import load_dotenv
from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.application.use_cases.ingest_pdf import IngestPDF
from app.infrastructure.services.llm_service import OpenAILLMService
from app.infrastructure.services.chunker import LangChainChunker
from app.infrastructure.services.openai_embedder import OpenAIEmbedder
from app.infrastructure.services.pdf_parser import DoclingParser
from app.infrastructure.services.vectore_store import ChromaVectorStore

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

parser = DoclingParser()
chunker = LangChainChunker()
embedder = OpenAIEmbedder(api_key=api_key)
vector_store = ChromaVectorStore()
llm_service = OpenAILLMService(api_key=api_key)

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