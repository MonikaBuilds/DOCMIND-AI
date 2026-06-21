from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk
from app.services.retrieval_service import RetrievalConfig, RetrievalService


class FakeEmbeddingService:
    def embed_query(self, query: str) -> list[float]:
        return [1.0, float(len(query))]


class FakeVectorService:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[list[float], int]] = []

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query_embedding, top_k))
        return self.results[:top_k]


def make_chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"doc:p{index + 1}:c{index}",
        document_id="doc",
        filename="hybrid.pdf",
        text=text,
        page_number=index + 1,
        chunk_index=index,
        source=f"hybrid.pdf#page={index + 1}",
        heading=None,
    )


def make_retrieved(chunk: DocumentChunk, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        text=chunk.text,
        score=score,
        page_number=chunk.page_number,
        source=chunk.source,
        heading=chunk.heading,
    )


def test_keyword_search_finds_exact_term_matches():
    chunks = [
        make_chunk(0, "Semantic search finds similar meaning."),
        make_chunk(1, "OCR extracts text from scanned PDF pages."),
    ]
    service = RetrievalService(FakeEmbeddingService(), FakeVectorService([]))

    results = service.keyword_search("scanned OCR", chunks, top_k=2)

    assert len(results) == 1
    assert results[0].chunk_id == "doc:p2:c1"
    assert results[0].score == 1.0


def test_exact_search_requires_literal_word_or_phrase_matches():
    chunks = [
        make_chunk(0, "A transformer attends to visual patches."),
        make_chunk(1, "The document explains Vision Backbone architecture."),
        make_chunk(2, "Backbone-like should not match a whole word search."),
    ]
    service = RetrievalService(FakeEmbeddingService(), FakeVectorService([]))

    word_results = service.exact_search("backbone", chunks, top_k=5)
    phrase_results = service.exact_search("Vision Backbone", chunks, top_k=5)

    assert [result.chunk_id for result in word_results] == ["doc:p2:c1"]
    assert [result.chunk_id for result in phrase_results] == ["doc:p2:c1"]


def test_hybrid_search_merges_semantic_and_keyword_results_without_duplicates():
    chunks = [
        make_chunk(0, "RAG uses retrieval and generation."),
        make_chunk(1, "BM25 keyword search handles exact rare terms."),
    ]
    semantic_results = [make_retrieved(chunks[0], score=0.8)]
    service = RetrievalService(
        FakeEmbeddingService(),
        FakeVectorService(semantic_results),
        RetrievalConfig(semantic_weight=0.6, keyword_weight=0.4),
    )

    results = service.hybrid_search("BM25 retrieval", chunks, top_k=5)

    assert {result.chunk_id for result in results} == {"doc:p1:c0", "doc:p2:c1"}
    assert results[0].score >= results[1].score


def test_hybrid_search_boosts_chunk_found_by_both_methods():
    chunks = [
        make_chunk(0, "RAG retrieval uses context."),
        make_chunk(1, "Unrelated cooking recipe."),
    ]
    semantic_results = [
        make_retrieved(chunks[0], score=0.7),
        make_retrieved(chunks[1], score=0.6),
    ]
    service = RetrievalService(
        FakeEmbeddingService(),
        FakeVectorService(semantic_results),
        RetrievalConfig(semantic_weight=0.5, keyword_weight=0.5),
    )

    results = service.hybrid_search("retrieval context", chunks, top_k=2)

    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].score > semantic_results[0].score * 0.5


def test_hybrid_search_returns_empty_for_blank_query():
    chunks = [make_chunk(0, "RAG retrieval uses context.")]
    vector_service = FakeVectorService([])
    service = RetrievalService(FakeEmbeddingService(), vector_service)

    assert service.hybrid_search("   ", chunks) == []
    assert vector_service.calls == []
