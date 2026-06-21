from pathlib import Path

from app.domain.documents import Document, DocumentPage, PDFMetadata, ParsedDocument
from app.services.preprocessing_service import PreprocessingService


def make_parsed_document(pages: list[DocumentPage]) -> ParsedDocument:
    return ParsedDocument(
        document=Document(
            document_id="doc-clean",
            filename="clean.pdf",
            source_path=Path("uploads/doc-clean.pdf"),
            content_type="application/pdf",
            size_bytes=100,
        ),
        metadata=PDFMetadata(
            title=None,
            author=None,
            subject=None,
            creator=None,
            producer=None,
            page_count=len(pages),
        ),
        pages=pages,
    )


def make_page(page_number: int, text: str) -> DocumentPage:
    return DocumentPage(
        document_id="doc-clean",
        page_number=page_number,
        text=text,
        extraction_method="pymupdf",
        char_count=len(text),
        word_count=len(text.split()),
        needs_ocr=False,
    )


def test_clean_text_repairs_hyphenation_and_soft_line_breaks():
    service = PreprocessingService()

    cleaned = service.clean_text("Retrieval aug-\nmented generation\nuses context.")

    assert cleaned == "Retrieval augmented generation uses context."


def test_clean_document_preserves_page_metadata():
    parsed = make_parsed_document([make_page(3, "  AI   systems\nneed   clean text.  ")])

    cleaned = PreprocessingService().clean_document(parsed)

    assert cleaned.document.document_id == "doc-clean"
    assert cleaned.pages[0].page_number == 3
    assert cleaned.pages[0].extraction_method == "pymupdf"
    assert cleaned.pages[0].text == "AI systems need clean text."
    assert cleaned.pages[0].char_count == len("AI systems need clean text.")


def test_clean_document_removes_repeated_headers_and_footers():
    pages = [
        make_page(1, "DocMind AI\nFirst page content about RAG.\nConfidential"),
        make_page(2, "DocMind AI\nSecond page content about OCR.\nConfidential"),
        make_page(3, "DocMind AI\nThird page content about vectors.\nConfidential"),
    ]

    cleaned = PreprocessingService().clean_document(make_parsed_document(pages))

    assert all("DocMind AI" not in page.text for page in cleaned.pages)
    assert all("Confidential" not in page.text for page in cleaned.pages)
    assert "First page content" in cleaned.pages[0].text
    assert "Second page content" in cleaned.pages[1].text
    assert "Third page content" in cleaned.pages[2].text


def test_short_documents_do_not_remove_repeated_edge_lines():
    pages = [
        make_page(1, "Important Title\nUseful content one."),
        make_page(2, "Important Title\nUseful content two."),
    ]

    cleaned = PreprocessingService().clean_document(make_parsed_document(pages))

    assert "Important Title" in cleaned.pages[0].text
    assert "Important Title" in cleaned.pages[1].text
