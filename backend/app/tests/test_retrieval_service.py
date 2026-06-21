from app.domain.retrieval import RetrievedChunk
from app.services.retrieval_service import RetrievalConfig, RetrievalService


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [float(len(query)), 1.0]


class FakeVectorService:
    def __init__(self) -> None:
        self.search_calls: list[tuple[list[float], int]] = []

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        self.search_calls.append((query_embedding, top_k))
        return [
            RetrievedChunk(
                chunk_id="doc:p1:c0",
                document_id="doc",
                filename="retrieval.pdf",
                text="high score text",
                score=0.9,
                page_number=1,
                source="retrieval.pdf#page=1",
                heading="Intro",
            ),
            RetrievedChunk(
                chunk_id="doc:p2:c1",
                document_id="doc",
                filename="retrieval.pdf",
                text="low score text",
                score=0.2,
                page_number=2,
                source="retrieval.pdf#page=2",
                heading=None,
            ),
        ]


def test_semantic_search_embeds_query_and_searches_vectors():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()
    service = RetrievalService(embedding_service, vector_service)

    results = service.semantic_search(" What is RAG? ", top_k=3)

    assert embedding_service.queries == ["What is RAG?"]
    assert vector_service.search_calls == [([12.0, 1.0], 3)]
    assert len(results) == 2


def test_semantic_search_uses_default_for_invalid_top_k():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()
    service = RetrievalService(
        embedding_service,
        vector_service,
        RetrievalConfig(default_top_k=4, max_top_k=10),
    )

    service.semantic_search("question", top_k=0)

    assert vector_service.search_calls == [([8.0, 1.0], 4)]


def test_semantic_search_caps_top_k():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()
    service = RetrievalService(
        embedding_service,
        vector_service,
        RetrievalConfig(default_top_k=5, max_top_k=7),
    )

    service.semantic_search("question", top_k=99)

    assert vector_service.search_calls == [([8.0, 1.0], 7)]


def test_semantic_search_filters_by_min_score():
    service = RetrievalService(
        FakeEmbeddingService(),
        FakeVectorService(),
        RetrievalConfig(min_score=0.5),
    )

    results = service.semantic_search("question")

    assert len(results) == 1
    assert results[0].score == 0.9


def test_semantic_search_returns_empty_for_blank_query():
    embedding_service = FakeEmbeddingService()
    vector_service = FakeVectorService()
    service = RetrievalService(embedding_service, vector_service)

    assert service.semantic_search("   ") == []
    assert embedding_service.queries == []
    assert vector_service.search_calls == []
