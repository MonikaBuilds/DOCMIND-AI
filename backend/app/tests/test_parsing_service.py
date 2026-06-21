from pathlib import Path

import fitz
import pytest

from app.core.exceptions import DocumentProcessingError
from app.domain.documents import Document
from app.services.parsing_service import ParsingService


def create_pdf(path: Path, page_texts: list[str]) -> None:
    pdf = fitz.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def make_document(path: Path) -> Document:
    return Document(
        document_id="doc-123",
        filename=path.name,
        source_path=path,
        content_type="application/pdf",
        size_bytes=path.stat().st_size,
    )


def test_parse_document_extracts_page_text_and_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    create_pdf(
        pdf_path,
        [
            "This is a document intelligence test page with enough words for extraction.",
            "Second page explains retrieval augmented generation with page metadata.",
        ],
    )

    parsed = ParsingService().parse_document(make_document(pdf_path))

    assert parsed.document.document_id == "doc-123"
    assert parsed.metadata.page_count == 2
    assert len(parsed.pages) == 2
    assert parsed.pages[0].page_number == 1
    assert "document intelligence" in parsed.pages[0].text
    assert parsed.pages[0].extraction_method == "pymupdf"
    assert parsed.pages[0].needs_ocr is False


def test_parse_document_marks_blank_page_for_ocr(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    create_pdf(pdf_path, [""])

    parsed = ParsingService().parse_document(make_document(pdf_path))

    assert parsed.metadata.page_count == 1
    assert parsed.pages[0].text == ""
    assert parsed.pages[0].needs_ocr is True


def test_parse_document_raises_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.pdf"
    document = Document(
        document_id="missing-doc",
        filename="missing.pdf",
        source_path=missing_path,
        content_type="application/pdf",
        size_bytes=0,
    )

    with pytest.raises(DocumentProcessingError):
        ParsingService().parse_document(document)
