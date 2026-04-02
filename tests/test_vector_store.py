import sys
import types

from app.domain.entities import EmbeddedChunk
from app.infrastructure.services.vectore_store import PineconeVectorStore


class FakePineconeIndex:
    def __init__(self):
        self.upsert_calls = []
        self.query_calls = []
        self.query_response = {
            "matches": [
                {
                    "id": "chunk-2",
                    "score": 0.72,
                    "metadata": {
                        "text": "Second chunk",
                        "source_document_id": "doc-2",
                    },
                },
                {
                    "id": "chunk-1",
                    "score": 0.91,
                    "metadata": {
                        "text": "First chunk",
                        "source_document_id": "doc-1",
                    },
                },
            ]
        }

    def upsert(self, vectors, namespace):
        self.upsert_calls.append({"vectors": vectors, "namespace": namespace})

    def query(self, vector, top_k, namespace, include_metadata):
        self.query_calls.append(
            {
                "vector": vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": include_metadata,
            }
        )
        return self.query_response


class FakePineconeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.indexes = []

    def Index(self, host):
        index = FakePineconeIndex()
        self.indexes.append({"host": host, "index": index})
        return index


def install_fake_pinecone_module():
    fake_module = types.ModuleType("pinecone")
    fake_module.Pinecone = FakePineconeClient
    sys.modules["pinecone"] = fake_module


def test_pinecone_vector_store_adds_vectors_with_metadata():
    install_fake_pinecone_module()
    vector_store = PineconeVectorStore(
        api_key="test-key",
        index_host="index-host",
        namespace="documents",
    )

    vector_store.add(
        [
            EmbeddedChunk(
                id="chunk-1",
                text="First chunk",
                vector=[0.1, 0.2],
                source_document_id="doc-1",
            )
        ]
    )

    assert vector_store.index.upsert_calls == [
        {
            "vectors": [
                {
                    "id": "chunk-1",
                    "values": [0.1, 0.2],
                    "metadata": {
                        "text": "First chunk",
                        "source_document_id": "doc-1",
                    },
                }
            ],
            "namespace": "documents",
        }
    ]


def test_pinecone_vector_store_retrieves_ranked_chunks():
    install_fake_pinecone_module()
    vector_store = PineconeVectorStore(
        api_key="test-key",
        index_host="index-host",
        namespace="documents",
    )

    results = vector_store.retrieve(query_embedding=[0.3, 0.4], top_k=2)

    assert vector_store.index.query_calls == [
        {
            "vector": [0.3, 0.4],
            "top_k": 2,
            "namespace": "documents",
            "include_metadata": True,
        }
    ]
    assert [chunk.id for chunk in results] == ["chunk-1", "chunk-2"]
    assert [chunk.text for chunk in results] == ["First chunk", "Second chunk"]
