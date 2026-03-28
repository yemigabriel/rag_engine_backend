import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.openai_embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "documents")
        self.vector_store_backend = os.getenv("VECTOR_STORE_BACKEND", "persistent")
        self.chroma_persist_path = os.getenv("CHROMA_PERSIST_PATH", "./data/chroma")
        self.chroma_host = os.getenv("CHROMA_HOST")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.chroma_ssl = os.getenv("CHROMA_SSL", "false").lower() == "true"
        self.redis_url = os.getenv("REDIS_URL")
        self.rq_queue_name = os.getenv("RQ_QUEUE_NAME", "pdf_ingestion")
        self.rq_job_timeout = int(os.getenv("RQ_JOB_TIMEOUT", "1800"))
        self.upload_max_bytes = int(os.getenv("UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))

    def validate(self) -> None:
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.vector_store_backend not in {"memory", "persistent", "http"}:
            raise ValueError("VECTOR_STORE_BACKEND must be memory, persistent, or http.")
        if self.vector_store_backend == "http" and not self.chroma_host:
            raise ValueError("CHROMA_HOST is required when VECTOR_STORE_BACKEND=http.")
        if self.rq_job_timeout <= 0:
            raise ValueError("RQ_JOB_TIMEOUT must be greater than 0.")
        if self.upload_max_bytes <= 0:
            raise ValueError("UPLOAD_MAX_BYTES must be greater than 0.")

    def validate_queue(self) -> None:
        if not self.redis_url:
            raise ValueError("REDIS_URL is required for background ingestion.")


settings = Settings()
