from pathlib import Path

from app.domain.chunks import DocumentChunk
from app.domain.documents import Document, PDFMetadata, ParsedDocument
from app.services.metadata_service import MetadataService


def make_chunk(chunk_index: int, heading: str | None = None) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"doc-meta:p1:c{chunk_index}",
        document_id="doc-meta",
        filename="metadata.pdf",
        text="chunk text",
        page_number=1,
        chunk_index=chunk_index,
        source="metadata.pdf#page=1",
        heading=heading,
    )


def make_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        document=Document(
            document_id="doc-meta",
            filename="metadata.pdf",
            source_path=Path("uploads/doc-meta.pdf"),
            content_type="application/pdf",
            size_bytes=100,
        ),
        metadata=PDFMetadata(
            title="RAG Metadata",
            author="Monika",
            subject=None,
            creator=None,
            producer=None,
            page_count=3,
        ),
        pages=[],
    )


def test_chunk_to_vector_metadata_includes_citation_fields():
    service = MetadataService()
    chunk = make_chunk(0, heading="Introduction")

    metadata = service.chunk_to_vector_metadata(chunk)

    assert metadata == {
        "chunk_id": "doc-meta:p1:c0",
        "document_id": "doc-meta",
        "filename": "metadata.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "source": "metadata.pdf#page=1",
        "citation": "metadata.pdf, page 1, Introduction",
        "heading": "Introduction",
    }


def test_citation_label_omits_missing_heading():
    citation = MetadataService().citation_label(make_chunk(0, heading=None))

    assert citation == "metadata.pdf, page 1"


def test_propagate_headings_inherits_previous_heading_per_document():
    service = MetadataService()
    chunks = [
        make_chunk(0, heading="Chapter 1"),
        make_chunk(1, heading=None),
        make_chunk(2, heading="Chapter 2"),
        make_chunk(3, heading=None),
    ]

    propagated = service.propagate_headings(chunks)

    assert [chunk.heading for chunk in propagated] == [
        "Chapter 1",
        "Chapter 1",
        "Chapter 2",
        "Chapter 2",
    ]


def test_document_summary_counts_chunks_for_document():
    service = MetadataService()
    summary = service.document_summary(make_parsed_document(), [make_chunk(0), make_chunk(1)])

    assert summary.document_id == "doc-meta"
    assert summary.filename == "metadata.pdf"
    assert summary.title == "RAG Metadata"
    assert summary.author == "Monika"
    assert summary.page_count == 3
    assert summary.chunk_count == 2
