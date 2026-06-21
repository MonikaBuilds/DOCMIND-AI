import pytest

from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk
from app.services.vector_service import VectorService


class FakeVectorStore:
    def __init__(self) -> None:
        self.added_chunks: list[DocumentChunk] = []
        self.added_embeddings: list[list[float]] = []
        self.search_calls: list[tuple[list[float], int]] = []

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        self.search_calls.append((query_embedding, top_k))
        return [
            RetrievedChunk(
                chunk_id="doc:p1:c0",
                document_id="doc",
                filename="vectors.pdf",
                text="retrieved text",
                score=0.91,
                page_number=1,
                source="vectors.pdf#page=1",
                heading="Intro",
            )
        ]


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="doc:p1:c0",
        document_id="doc",
        filename="vectors.pdf",
        text="chunk text",
        page_number=1,
        chunk_index=0,
        source="vectors.pdf#page=1",
        heading="Intro",
    )


def test_vector_service_adds_chunks_and_embeddings():
    store = FakeVectorStore()
    service = VectorService(store)
    chunk = make_chunk()

    service.add_chunks([chunk], [[0.1, 0.2]])

    assert store.added_chunks == [chunk]
    assert store.added_embeddings == [[0.1, 0.2]]


def test_vector_service_rejects_embedding_count_mismatch():
    service = VectorService(FakeVectorStore())

    with pytest.raises(ValueError):
        service.add_chunks([make_chunk()], [])


def test_vector_service_search_delegates_to_store():
    store = FakeVectorStore()
    service = VectorService(store)

    results = service.search([0.3, 0.4], top_k=3)

    assert store.search_calls == [([0.3, 0.4], 3)]
    assert len(results) == 1
    assert results[0].score == 0.91
    assert results[0].source == "vectors.pdf#page=1"


def test_vector_service_returns_empty_for_non_positive_top_k():
    store = FakeVectorStore()
    service = VectorService(store)

    assert service.search([0.3, 0.4], top_k=0) == []
    assert store.search_calls == []
