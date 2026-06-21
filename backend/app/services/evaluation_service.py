import re

from app.domain.chat import CitedAnswer
from app.domain.evaluation import (
    AnswerEvaluationResult,
    RAGEvaluationReport,
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
)
from app.domain.retrieval import RetrievedChunk
from app.services.citation_service import CitationService


class EvaluationService:
    """Measures retrieval quality, citation accuracy, and answer grounding."""

    def __init__(self, citation_service: CitationService | None = None) -> None:
        self.citation_service = citation_service or CitationService()

    def evaluate_retrieval(
        self,
        expected_chunk_ids: set[str],
        retrieved_chunks: list[RetrievedChunk],
        k: int,
    ) -> RetrievalEvaluationResult:
        top_chunks = retrieved_chunks[:k]
        retrieved_ids = {chunk.chunk_id for chunk in top_chunks}
        matched_ids = expected_chunk_ids & retrieved_ids

        expected_count = len(expected_chunk_ids)
        retrieved_count = len(top_chunks)
        matched_count = len(matched_ids)

        recall = matched_count / expected_count if expected_count else 0.0
        precision = matched_count / retrieved_count if retrieved_count else 0.0
        hit_rate = 1.0 if matched_count > 0 else 0.0

        return RetrievalEvaluationResult(
            recall_at_k=round(recall, 6),
            precision_at_k=round(precision, 6),
            hit_rate=hit_rate,
            expected_count=expected_count,
            retrieved_count=retrieved_count,
            matched_count=matched_count,
        )

    def evaluate_citation_accuracy(
        self,
        expected_citation_labels: set[str],
        answer: CitedAnswer,
    ) -> float:
        if not expected_citation_labels:
            return 1.0 if not answer.citations else 0.0

        predicted = set(answer.citations)
        matched = expected_citation_labels & predicted
        return round(len(matched) / len(expected_citation_labels), 6)

    def evaluate_answer(
        self,
        expected_citation_labels: set[str],
        answer: CitedAnswer,
        retrieved_chunks: list[RetrievedChunk],
    ) -> AnswerEvaluationResult:
        citation_accuracy = self.evaluate_citation_accuracy(expected_citation_labels, answer)
        claims = self._extract_claims(answer.answer)
        supported_claims = [
            claim
            for claim in claims
            if self._claim_supported_by_context(claim, retrieved_chunks)
        ]

        claim_count = len(claims)
        supported_count = len(supported_claims)
        groundedness = supported_count / claim_count if claim_count else 1.0
        hallucination_risk = 1.0 - groundedness

        return AnswerEvaluationResult(
            citation_accuracy=citation_accuracy,
            groundedness=round(groundedness, 6),
            faithfulness=round(groundedness, 6),
            hallucination_risk=round(hallucination_risk, 6),
            supported_claim_count=supported_count,
            claim_count=claim_count,
        )

    def evaluate_case(
        self,
        case: RetrievalEvaluationCase,
        retrieved_chunks: list[RetrievedChunk],
        answer: CitedAnswer,
        k: int,
    ) -> RAGEvaluationReport:
        return RAGEvaluationReport(
            retrieval=self.evaluate_retrieval(case.expected_chunk_ids, retrieved_chunks, k),
            answer=self.evaluate_answer(case.expected_citation_labels, answer, retrieved_chunks),
        )

    def _extract_claims(self, answer_text: str) -> list[str]:
        claims = [
            claim.strip()
            for claim in re.split(r"[.!?]\s+", answer_text.strip())
            if claim.strip()
        ]
        return [
            claim
            for claim in claims
            if "not contain enough information" not in claim.lower()
        ]

    def _claim_supported_by_context(
        self,
        claim: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> bool:
        claim_terms = self._content_terms(claim)
        if not claim_terms:
            return True

        context_terms = set()
        for chunk in retrieved_chunks:
            context_terms.update(self._content_terms(chunk.text))

        overlap = claim_terms & context_terms
        return len(overlap) / len(claim_terms) >= 0.5

    def _content_terms(self, text: str) -> set[str]:
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "with",
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in stop_words and len(token) > 2
        }
