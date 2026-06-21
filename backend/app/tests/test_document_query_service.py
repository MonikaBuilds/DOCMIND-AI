from app.domain.chunks import DocumentChunk
from app.services.document_query_service import DocumentQueryService


def test_detects_document_overview_questions() -> None:
    service = DocumentQueryService()

    assert service.is_overview_question("What is this document about?")
    assert service.is_overview_question("the document is about?")
    assert service.is_overview_question("Tell me about this PDF")
    assert service.is_overview_question("PDF is about?")
    assert service.is_overview_question("Summarize this PDF")
    assert service.is_overview_question("Give me the main topic")
    assert not service.is_overview_question("What model is used?")


def test_overview_chunks_follow_document_order() -> None:
    service = DocumentQueryService()
    chunks = [
        DocumentChunk("c2", "doc-1", "sample.pdf", "Second page", 2, 0, "sample.pdf"),
        DocumentChunk("c1", "doc-1", "sample.pdf", "First page", 1, 0, "sample.pdf"),
    ]

    results = service.overview_chunks(chunks)

    assert [result.text for result in results] == ["First page", "Second page"]
