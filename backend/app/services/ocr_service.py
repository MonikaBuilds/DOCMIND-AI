from typing import Protocol

from app.domain.documents import DocumentPage, ParsedDocument
from app.infrastructure.ocr.base import OCRProvider


class PageImageRenderer(Protocol):
    def render_page_to_png(self, pdf_path, page_number: int, zoom: float = 2.0) -> bytes:
        raise NotImplementedError


class OCRService:
    """Applies OCR only to pages where direct PDF extraction was weak."""

    def __init__(self, ocr_provider: OCRProvider, page_renderer: PageImageRenderer) -> None:
        self.ocr_provider = ocr_provider
        self.page_renderer = page_renderer

    def apply_ocr_to_needed_pages(self, parsed_document: ParsedDocument) -> ParsedDocument:
        updated_pages = [
            self._ocr_page(parsed_document, page) if page.needs_ocr else page
            for page in parsed_document.pages
        ]

        return ParsedDocument(
            document=parsed_document.document,
            metadata=parsed_document.metadata,
            pages=updated_pages,
        )

    def _ocr_page(self, parsed_document: ParsedDocument, page: DocumentPage) -> DocumentPage:
        image_bytes = self.page_renderer.render_page_to_png(
            parsed_document.document.source_path,
            page.page_number,
        )
        ocr_text = self.ocr_provider.extract_text_from_image(image_bytes).strip()

        if not ocr_text:
            return page

        return DocumentPage(
            document_id=page.document_id,
            page_number=page.page_number,
            text=ocr_text,
            extraction_method="ocr",
            char_count=len(ocr_text),
            word_count=len(ocr_text.split()),
            needs_ocr=False,
        )
