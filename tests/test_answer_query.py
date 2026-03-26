from app.application.use_cases.answer_query import AnswerQueryUseCase
from app.domain.entities import RetreivedChunk
from tests.fakes import FakeEmbedder, FakeLLMService, FakeRetrieveVectorStore


def test_answer_uses_retrieved_context_and_updates_history():
    embedder = FakeEmbedder([[0.11, 0.22]])
    vector_store = FakeRetrieveVectorStore(
        [
            RetreivedChunk(
                id="1",
                text="Relevant paragraph",
                source_document_id="doc-1",
                score=0.9,
            )
        ]
    )
    llm_service = FakeLLMService(answer="Grounded answer")

    use_case = AnswerQueryUseCase(
        embedder=embedder,
        vector_store=vector_store,
        llm_service=llm_service,
    )

    result = use_case.answer("What does the document say?")

    assert embedder.calls == [["What does the document say?"]]
    assert vector_store.calls == [{"query_embedding": [0.11, 0.22], "top_k": 10}]
    assert llm_service.generate_answer_calls == [
        {
            "question": "What does the document say?",
            "context": ["Relevant paragraph"],
            "history": None,
        }
    ]
    assert result.question == "What does the document say?"
    assert result.answer == "Grounded answer"
    assert result.context == ["Relevant paragraph"]
    assert result.history == [
        {"role": "user", "content": "What does the document say?"},
        {"role": "assistant", "content": "Grounded answer"},
    ]

def test_answer_rewrites_follow_up_questions_when_history_exists():
    history = [{"role": "user", "content": "Tell me about the document"}]
    embedder = FakeEmbedder([[0.33, 0.44]])
    vector_store = FakeRetrieveVectorStore([])
    llm_service = FakeLLMService(
        answer="Rewritten answer",
        rewritten_question="What does the document say about pricing?",
    )

    use_case = AnswerQueryUseCase(
        embedder=embedder,
        vector_store=vector_store,
        llm_service=llm_service,
    )

    result = use_case.answer("What about pricing?", history=history)

    assert llm_service.rewrite_question_calls == [
        {"question": "What about pricing?", "history": history}
    ]
    assert embedder.calls == [["What does the document say about pricing?"]]
    assert llm_service.generate_answer_calls == [
        {
            "question": "What does the document say about pricing?",
            "context": [],
            "history": history,
        }
    ]
    assert result.question == "What does the document say about pricing?"
    assert result.history == [
        {"role": "user", "content": "Tell me about the document"},
        {"role": "user", "content": "What about pricing?"},
        {"role": "assistant", "content": "Rewritten answer"},
    ]
