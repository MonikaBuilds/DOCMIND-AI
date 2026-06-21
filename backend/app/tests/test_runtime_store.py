from pathlib import Path

from app.core.runtime_store import RuntimeStore
from app.domain.chunks import DocumentChunk
from app.domain.documents import Document, PDFMetadata, ParsedDocument


def make_document(document_id: str) -> Document:
    return Document(
        document_id=document_id,
        filename=f"{document_id}.pdf",
        source_path=Path(f"uploads/{document_id}.pdf"),
        content_type="application/pdf",
        size_bytes=100,
    )


def make_parsed_document(document: Document) -> ParsedDocument:
    return ParsedDocument(
        document=document,
        metadata=PDFMetadata(
            title=None,
            author=None,
            subject=None,
            creator=None,
            producer=None,
            page_count=1,
        ),
        pages=[],
    )


def make_chunk(document_id: str, index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{document_id}:p1:c{index}",
        document_id=document_id,
        filename=f"{document_id}.pdf",
        text="stored chunk",
        page_number=1,
        chunk_index=index,
        source=f"{document_id}.pdf#page=1",
        heading=None,
    )


def test_runtime_store_adds_and_lists_documents():
    store = RuntimeStore()
    document = make_document("doc-a")

    store.add_document(document)

    assert store.get_document("doc-a") == document
    assert store.list_records()[0].document == document


def test_runtime_store_saves_processed_document_and_filters_chunks():
    store = RuntimeStore()
    doc_a = make_document("doc-a")
    doc_b = make_document("doc-b")
    store.add_document(doc_a)
    store.add_document(doc_b)

    store.save_processed_document("doc-a", make_parsed_document(doc_a), [make_chunk("doc-a", 0)])
    store.save_processed_document("doc-b", make_parsed_document(doc_b), [make_chunk("doc-b", 0)])

    assert [chunk.document_id for chunk in store.get_chunks()] == ["doc-a", "doc-b"]
    assert [chunk.document_id for chunk in store.get_chunks(["doc-b"])] == ["doc-b"]


def test_runtime_store_clear_removes_documents():
    store = RuntimeStore()
    store.add_document(make_document("doc-a"))

    store.clear()

    assert store.list_records() == []
