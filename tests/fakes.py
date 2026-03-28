class FakeEmbedder:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return self.responses


class FakeParser:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def parse(self, document_path: str) -> str:
        self.calls.append(document_path)
        return self.content


class FakeChunker:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def chunk(self, text: str):
        self.calls.append(text)
        return self.chunks


class FakeIngestVectorStore:
    def __init__(self):
        self.added_chunks = None

    def add(self, chunks):
        self.added_chunks = chunks


class FakeRetrieveVectorStore:
    def __init__(self, retrieved_chunks):
        self.retrieved_chunks = retrieved_chunks
        self.calls = []

    def retrieve(self, query_embedding, top_k):
        self.calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self.retrieved_chunks


class FakeLLMService:
    def __init__(self, answer="final answer", rewritten_question="rewritten question"):
        self.answer = answer
        self.rewritten_question = rewritten_question
        self.generate_answer_calls = []
        self.rewrite_question_calls = []

    def generate_answer(self, question, context, history):
        self.generate_answer_calls.append(
            {"question": question, "context": context, "history": history}
        )
        return self.answer

    def generate_answer_stream(self, question, context, history):
        yield "partial"

    def rewrite_question(self, question, history):
        self.rewrite_question_calls.append({"question": question, "history": history})
        return self.rewritten_question


class FakeIngestPDFUseCase:
    def __init__(self):
        self.calls = []

    def execute(self, document_path: str) -> None:
        self.calls.append(document_path)


class FakeAnswerQueryUseCase:
    def __init__(self):
        self.answer_calls = []
        self.prepare_calls = []
        self.stream_calls = []

    def answer(self, question: str, top_k: int = 10, history=None):
        from app.domain.entities import Answer

        self.answer_calls.append(
            {"question": question, "top_k": top_k, "history": history}
        )
        return Answer(
            question=question,
            answer="Synthesized answer",
            context=["Chunk A", "Chunk B"],
            history=(history or [])
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Synthesized answer"},
            ],
        )

    def prepare_answer(self, question: str, top_k: int = 10, history=None):
        self.prepare_calls.append(
            {"question": question, "top_k": top_k, "history": history}
        )
        return ["Chunk A", "Chunk B"], "Rewritten question"

    def stream_answer(self, question: str, context, top_k: int = 10, history=None):
        self.stream_calls.append(
            {
                "question": question,
                "context": context,
                "top_k": top_k,
                "history": history,
            }
        )
        yield "Hello "
        yield "world"


class FakeQueuedJob:
    def __init__(self, job_id="job-123", status="queued", result=None, exc_info=None):
        self.id = job_id
        self._status = status
        self.result = result
        self.exc_info = exc_info

    @property
    def is_failed(self):
        return self._status == "failed"

    def get_status(self, refresh=False):
        return self._status


class FakeQueue:
    def __init__(self):
        self.calls = []
        self.job = FakeQueuedJob()

    def enqueue(self, job_path, file_bytes, filename):
        self.calls.append(
            {
                "job_path": job_path,
                "file_bytes": file_bytes,
                "filename": filename,
            }
        )
        return self.job
