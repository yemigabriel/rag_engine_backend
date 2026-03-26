from app.application.use_cases.ingest_pdf import IngestPDF
from app.domain.entities import Chunk
from tests.fakes import FakeChunker, FakeEmbedder, FakeIngestVectorStore, FakeParser


def test_ingest_pdf_parses_chunks_embeds_and_stores():
    chunks = [
        Chunk(id="chunk-1", text="First chunk", source_document_id="doc-1"),
        Chunk(id="chunk-2", text="Second chunk", source_document_id="doc-1"),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    parser = FakeParser("parsed markdown")
    chunker = FakeChunker(chunks)
    embedder = FakeEmbedder(embeddings)
    vector_store = FakeIngestVectorStore()

    use_case = IngestPDF(
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    use_case.execute("/tmp/sample.pdf")

    assert parser.calls == ["/tmp/sample.pdf"]
    assert chunker.calls == ["parsed markdown"]
    assert embedder.calls == [["First chunk", "Second chunk"]]

    assert vector_store.added_chunks is not None
    assert [chunk.id for chunk in vector_store.added_chunks] == ["chunk-1", "chunk-2"]
    assert [chunk.text for chunk in vector_store.added_chunks] == ["First chunk", "Second chunk"]
    assert [chunk.vector for chunk in vector_store.added_chunks] == embeddings
    assert [chunk.source_document_id for chunk in vector_store.added_chunks] == ["doc-1", "doc-1"]
