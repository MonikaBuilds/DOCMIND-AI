from pathlib import Path

from app.domain.documents import Document, DocumentPage, PDFMetadata, ParsedDocument
from app.services.ocr_service import OCRService


class FakeOCRProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        self.calls += 1
        assert image_bytes == b"fake-png"
        return self.text


class FakePageRenderer:
    def __init__(self) -> None:
        self.rendered_pages: list[int] = []

    def render_page_to_png(self, pdf_path: Path, page_number: int, zoom: float = 2.0) -> bytes:
        self.rendered_pages.append(page_number)
        return b"fake-png"


def make_parsed_document() -> ParsedDocument:
    document = Document(
        document_id="doc-ocr",
        filename="scan.pdf",
        source_path=Path("uploads/doc-ocr.pdf"),
        content_type="application/pdf",
        size_bytes=10,
    )
    metadata = PDFMetadata(
        title=None,
        author=None,
        subject=None,
        creator=None,
        producer=None,
        page_count=2,
    )
    pages = [
        DocumentPage(
            document_id="doc-ocr",
            page_number=1,
            text="Already extracted text with enough words.",
            extraction_method="pymupdf",
            char_count=42,
            word_count=6,
            needs_ocr=False,
        ),
        DocumentPage(
            document_id="doc-ocr",
            page_number=2,
            text="",
            extraction_method="pymupdf",
            char_count=0,
            word_count=0,
            needs_ocr=True,
        ),
    ]
    return ParsedDocument(document=document, metadata=metadata, pages=pages)


def test_ocr_service_only_processes_pages_marked_for_ocr():
    ocr_provider = FakeOCRProvider("Recovered scanned page text.")
    renderer = FakePageRenderer()
    service = OCRService(ocr_provider=ocr_provider, page_renderer=renderer)

    parsed = service.apply_ocr_to_needed_pages(make_parsed_document())

    assert ocr_provider.calls == 1
    assert renderer.rendered_pages == [2]
    assert parsed.pages[0].extraction_method == "pymupdf"
    assert parsed.pages[1].text == "Recovered scanned page text."
    assert parsed.pages[1].extraction_method == "ocr"
    assert parsed.pages[1].needs_ocr is False


def test_ocr_service_keeps_original_page_when_ocr_returns_empty_text():
    ocr_provider = FakeOCRProvider("")
    renderer = FakePageRenderer()
    service = OCRService(ocr_provider=ocr_provider, page_renderer=renderer)

    parsed = service.apply_ocr_to_needed_pages(make_parsed_document())

    assert parsed.pages[1].text == ""
    assert parsed.pages[1].needs_ocr is True
