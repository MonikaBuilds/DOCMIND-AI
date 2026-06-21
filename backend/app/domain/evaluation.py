from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    expected_chunk_ids: set[str]
    expected_citation_labels: set[str]


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    expected_count: int
    retrieved_count: int
    matched_count: int


@dataclass(frozen=True)
class AnswerEvaluationResult:
    citation_accuracy: float
    groundedness: float
    faithfulness: float
    hallucination_risk: float
    supported_claim_count: int
    claim_count: int


@dataclass(frozen=True)
class RAGEvaluationReport:
    retrieval: RetrievalEvaluationResult
    answer: AnswerEvaluationResult
