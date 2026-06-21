from app.domain.chunks import DocumentChunk
from app.services.page_query_service import PageQueryService


def test_extracts_page_number_from_question() -> None:
    service = PageQueryService()

    assert service.extract_page_number("What is the topic on page number 7?") == 7
    assert service.extract_page_number("Summarize page 12") == 12
    assert service.extract_page_number("What is this document about?") is None


def test_returns_only_chunks_for_requested_page() -> None:
    service = PageQueryService()
    chunks = [
        DocumentChunk("c1", "doc-1", "sample.pdf", "Page one text", 1, 0, "sample.pdf"),
        DocumentChunk("c2", "doc-1", "sample.pdf", "Page seven text", 7, 1, "sample.pdf"),
    ]

    results = service.chunks_for_page(chunks, page_number=7)

    assert len(results) == 1
    assert results[0].text == "Page seven text"
    assert results[0].page_number == 7
