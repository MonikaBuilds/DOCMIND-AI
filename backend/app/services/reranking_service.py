from abc import ABC, abstractmethod
import re

from app.domain.retrieval import RetrievedChunk


class Reranker(ABC):
    """Interface for heuristic or model-based re-ranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class HeuristicReranker(Reranker):
    """Combines retrieval score with query term coverage."""

    def __init__(self, retrieval_score_weight: float = 0.7, term_overlap_weight: float = 0.3) -> None:
        self.retrieval_score_weight = retrieval_score_weight
        self.term_overlap_weight = term_overlap_weight

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        query_terms = set(self._tokenize(query))
        if not query_terms or not results:
            return results[:top_k] if top_k else results

        reranked = [self._with_rerank_score(result, query_terms) for result in results]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k] if top_k else reranked

    def _with_rerank_score(
        self,
        result: RetrievedChunk,
        query_terms: set[str],
    ) -> RetrievedChunk:
        text_terms = set(self._tokenize(result.text))
        overlap_score = len(query_terms & text_terms) / len(query_terms)
        final_score = (
            self.retrieval_score_weight * result.score
            + self.term_overlap_weight * overlap_score
        )

        return RetrievedChunk(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            filename=result.filename,
            text=result.text,
            score=round(final_score, 6),
            page_number=result.page_number,
            source=result.source,
            heading=result.heading,
        )

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


class RerankingService:
    """Improves candidate ordering after retrieval."""

    def __init__(self, reranker: Reranker | None = None) -> None:
        self.reranker = reranker or HeuristicReranker()

    def rerank(
        self,
        query: str,
        results: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if top_k is not None and top_k <= 0:
            return []

        return self.reranker.rerank(query, results, top_k)
