from app.domain.chat import CitedAnswer
from app.domain.evaluation import RetrievalEvaluationCase
from app.domain.retrieval import RetrievedChunk
from app.services.evaluation_service import EvaluationService


def make_chunk(chunk_id: str, text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-eval",
        filename="eval.pdf",
        text=text,
        score=1.0,
        page_number=page,
        source=f"eval.pdf#page={page}",
        heading=None,
    )


def test_evaluate_retrieval_computes_recall_precision_and_hit_rate():
    result = EvaluationService().evaluate_retrieval(
        expected_chunk_ids={"c1", "c3"},
        retrieved_chunks=[
            make_chunk("c1", "relevant"),
            make_chunk("c2", "irrelevant"),
            make_chunk("c3", "relevant"),
        ],
        k=2,
    )

    assert result.recall_at_k == 0.5
    assert result.precision_at_k == 0.5
    assert result.hit_rate == 1.0
    assert result.matched_count == 1


def test_evaluate_citation_accuracy_handles_expected_labels():
    answer = CitedAnswer(answer="Grounded answer.", citations=["eval.pdf, page 1"])

    accuracy = EvaluationService().evaluate_citation_accuracy({"eval.pdf, page 1"}, answer)

    assert accuracy == 1.0


def test_evaluate_answer_estimates_groundedness_from_context_terms():
    retrieved = [make_chunk("c1", "RAG uses retrieval and citations for grounded answers.")]
    answer = CitedAnswer(
        answer="RAG uses retrieval and citations. It also predicts stock prices.",
        citations=["eval.pdf, page 1"],
    )

    result = EvaluationService().evaluate_answer(
        expected_citation_labels={"eval.pdf, page 1"},
        answer=answer,
        retrieved_chunks=retrieved,
    )

    assert result.citation_accuracy == 1.0
    assert result.claim_count == 2
    assert result.supported_claim_count == 1
    assert result.groundedness == 0.5
    assert result.hallucination_risk == 0.5


def test_evaluate_case_returns_combined_report():
    service = EvaluationService()
    case = RetrievalEvaluationCase(
        query="What is RAG?",
        expected_chunk_ids={"c1"},
        expected_citation_labels={"eval.pdf, page 1"},
    )
    retrieved = [make_chunk("c1", "RAG uses retrieval.", page=1)]
    answer = CitedAnswer(answer="RAG uses retrieval.", citations=["eval.pdf, page 1"])

    report = service.evaluate_case(case, retrieved, answer, k=1)

    assert report.retrieval.recall_at_k == 1.0
    assert report.answer.citation_accuracy == 1.0
