from app.domain.retrieval import RetrievedChunk
from app.services.citation_service import CitationService


def make_chunk(
    chunk_id: str,
    page_number: int,
    heading: str | None = None,
    filename: str = "citations.pdf",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-cite",
        filename=filename,
        text="citation text",
        score=0.8,
        page_number=page_number,
        source=f"{filename}#page={page_number}",
        heading=heading,
    )


def test_build_citations_deduplicates_same_page_and_heading():
    chunks = [
        make_chunk("c0", page_number=1, heading="Intro"),
        make_chunk("c1", page_number=1, heading="Intro"),
    ]

    citations = CitationService().build_citations(chunks)

    assert len(citations) == 1
    assert citations[0].label == "citations.pdf, page 1, Intro"
    assert citations[0].chunk_ids == ("c0", "c1")


def test_build_citations_keeps_distinct_pages_in_order():
    chunks = [
        make_chunk("c0", page_number=2),
        make_chunk("c1", page_number=5),
    ]

    citations = CitationService().build_citations(chunks)

    assert [citation.page_number for citation in citations] == [2, 5]
    assert [citation.label for citation in citations] == [
        "citations.pdf, page 2",
        "citations.pdf, page 5",
    ]


def test_citation_labels_returns_display_labels():
    chunks = [make_chunk("c0", page_number=3, heading="Methods")]

    labels = CitationService().citation_labels(chunks)

    assert labels == ["citations.pdf, page 3, Methods"]
