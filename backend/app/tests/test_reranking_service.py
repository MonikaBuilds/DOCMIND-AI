from app.domain.retrieval import RetrievedChunk
from app.services.reranking_service import HeuristicReranker, RerankingService


class FakeReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int | None = None):
        self.calls.append((query, results, top_k))
        return list(reversed(results))[:top_k] if top_k else list(reversed(results))


def make_result(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        filename="rerank.pdf",
        text=text,
        score=score,
        page_number=1,
        source="rerank.pdf#page=1",
        heading=None,
    )


def test_heuristic_reranker_boosts_query_term_coverage():
    results = [
        make_result("a", "general document intelligence overview", 0.8),
        make_result("b", "retrieval augmented generation citations", 0.7),
    ]

    reranked = HeuristicReranker(
        retrieval_score_weight=0.5,
        term_overlap_weight=0.5,
    ).rerank("retrieval citations", results)

    assert reranked[0].chunk_id == "b"
    assert reranked[0].score > reranked[1].score


def test_reranking_service_delegates_to_provider():
    fake = FakeReranker()
    service = RerankingService(fake)
    results = [
        make_result("a", "first", 0.9),
        make_result("b", "second", 0.8),
    ]

    reranked = service.rerank("query", results, top_k=1)

    assert fake.calls == [("query", results, 1)]
    assert [result.chunk_id for result in reranked] == ["b"]


def test_reranking_service_returns_empty_for_invalid_top_k():
    results = [make_result("a", "first", 0.9)]

    assert RerankingService().rerank("query", results, top_k=0) == []


def test_heuristic_reranker_preserves_order_for_blank_query():
    results = [
        make_result("a", "first", 0.9),
        make_result("b", "second", 0.8),
    ]

    reranked = HeuristicReranker().rerank("   ", results)

    assert [result.chunk_id for result in reranked] == ["a", "b"]
