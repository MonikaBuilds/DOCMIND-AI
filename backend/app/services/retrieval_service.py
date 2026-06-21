from dataclasses import dataclass
import math
import re

from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


@dataclass(frozen=True)
class RetrievalConfig:
    default_top_k: int = 5
    max_top_k: int = 20
    min_score: float | None = None
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3


class RetrievalService:
    """Performs semantic retrieval by embedding a query and searching vectors."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_service: VectorService | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.config = config or RetrievalConfig()

    def semantic_search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        effective_top_k = self._normalize_top_k(top_k)
        if self.embedding_service is None or self.vector_service is None:
            raise ValueError("Semantic search requires embedding and vector services.")

        query_embedding = self.embedding_service.embed_query(normalized_query)
        results = self.vector_service.search(
            query_embedding,
            top_k=effective_top_k,
            document_ids=document_ids,
        )
        return self._filter_by_score(results)

    def keyword_search(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query or not chunks:
            return []

        effective_top_k = self._normalize_top_k(top_k)
        scorer = KeywordScorer(chunks)
        return scorer.search(normalized_query, effective_top_k)

    def exact_search(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query or not chunks:
            return []

        effective_top_k = self._normalize_top_k(top_k)
        pattern = self._exact_pattern(normalized_query)
        results: list[RetrievedChunk] = []

        for chunk in sorted(chunks, key=lambda item: (item.page_number, item.chunk_index)):
            matches = list(pattern.finditer(chunk.text))
            if not matches:
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=self._snippet_around_match(chunk.text, matches[0]),
                    score=1.0,
                    page_number=chunk.page_number,
                    source=chunk.source,
                    heading=chunk.heading,
                )
            )
            if len(results) >= effective_top_k:
                break

        return results

    def hybrid_search(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        effective_top_k = self._normalize_top_k(top_k)
        semantic_results = self.semantic_search(
            normalized_query,
            top_k=effective_top_k,
            document_ids=document_ids,
        )
        keyword_results = self.keyword_search(normalized_query, chunks, top_k=effective_top_k)
        fused_results = self._fuse_results(semantic_results, keyword_results)
        return fused_results[:effective_top_k]

    def _normalize_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return self.config.default_top_k

        if top_k <= 0:
            return self.config.default_top_k

        return min(top_k, self.config.max_top_k)

    def _exact_pattern(self, query: str) -> re.Pattern[str]:
        escaped_query = re.escape(query)
        if re.search(r"\s", query):
            return re.compile(escaped_query, re.IGNORECASE)

        return re.compile(rf"(?<![A-Za-z0-9-]){escaped_query}(?![A-Za-z0-9-])", re.IGNORECASE)

    def _snippet_around_match(self, text: str, match: re.Match[str], radius: int = 320) -> str:
        start = max(match.start() - radius, 0)
        end = min(match.end() + radius, len(text))
        snippet = text[start:end].strip()
        if start > 0:
            snippet = f"...{snippet}"
        if end < len(text):
            snippet = f"{snippet}..."
        return snippet

    def _filter_by_score(self, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self.config.min_score is None:
            return results

        return [result for result in results if result.score >= self.config.min_score]

    def _fuse_results(
        self,
        semantic_results: list[RetrievedChunk],
        keyword_results: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        fused: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}

        for result in semantic_results:
            fused[result.chunk_id] = result
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (
                self.config.semantic_weight * result.score
            )

        for result in keyword_results:
            fused.setdefault(result.chunk_id, result)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + (
                self.config.keyword_weight * result.score
            )

        ranked = sorted(
            fused.values(),
            key=lambda result: scores[result.chunk_id],
            reverse=True,
        )

        return [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                filename=result.filename,
                text=result.text,
                score=round(scores[result.chunk_id], 6),
                page_number=result.page_number,
                source=result.source,
                heading=result.heading,
            )
            for result in ranked
        ]


class KeywordScorer:
    """Small BM25-style scorer used as the first hybrid retrieval baseline."""

    k1 = 1.5
    b = 0.75

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        self.chunks = chunks
        self.tokenized_chunks = [self._tokenize(chunk.text) for chunk in chunks]
        self.avg_doc_length = self._average_doc_length()
        self.document_frequencies = self._document_frequencies()

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        query_terms = self._tokenize(query)
        if not query_terms or top_k <= 0:
            return []

        scored_results = [
            (chunk, self._score_chunk(query_terms, tokens))
            for chunk, tokens in zip(self.chunks, self.tokenized_chunks)
        ]
        scored_results = [
            (chunk, score)
            for chunk, score in scored_results
            if score > 0
        ]
        if not scored_results:
            return []

        max_score = max(score for _, score in scored_results)
        ranked = sorted(scored_results, key=lambda item: item[1], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                text=chunk.text,
                score=round(score / max_score, 6),
                page_number=chunk.page_number,
                source=chunk.source,
                heading=chunk.heading,
            )
            for chunk, score in ranked[:top_k]
        ]

    def _score_chunk(self, query_terms: list[str], chunk_terms: list[str]) -> float:
        if not chunk_terms:
            return 0.0

        term_counts = {term: chunk_terms.count(term) for term in set(chunk_terms)}
        score = 0.0

        for term in query_terms:
            if term not in term_counts:
                continue

            tf = term_counts[term]
            idf = self._idf(term)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (len(chunk_terms) / self.avg_doc_length)
            )
            score += idf * ((tf * (self.k1 + 1)) / denominator)

        return score

    def _idf(self, term: str) -> float:
        chunk_count = len(self.chunks)
        containing_count = self.document_frequencies.get(term, 0)
        return math.log(1 + ((chunk_count - containing_count + 0.5) / (containing_count + 0.5)))

    def _average_doc_length(self) -> float:
        lengths = [len(tokens) for tokens in self.tokenized_chunks]
        return sum(lengths) / len(lengths) if lengths else 1.0

    def _document_frequencies(self) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for tokens in self.tokenized_chunks:
            for term in set(tokens):
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())
