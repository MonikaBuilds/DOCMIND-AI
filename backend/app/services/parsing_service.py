from pathlib import Path

from app.core.exceptions import DocumentProcessingError
from app.domain.documents import Document, DocumentPage, PDFMetadata, ParsedDocument


class ParsingService:
    """Extracts page-level text and metadata from PDFs with PyMuPDF."""

    min_chars_for_text_page = 40
    min_words_for_text_page = 5

    def parse_document(self, document: Document) -> ParsedDocument:
        if not document.source_path.exists():
            raise DocumentProcessingError(f"PDF not found: {document.source_path}")

        try:
            return self._parse_with_pymupdf(document)
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError(f"Failed to parse PDF: {document.filename}") from exc

    def _parse_with_pymupdf(self, document: Document) -> ParsedDocument:
        import fitz

        with fitz.open(document.source_path) as pdf:
            metadata = self._extract_metadata(pdf.metadata, pdf.page_count)
            pages = [
                self._extract_page(document.document_id, page_number=index + 1, page=page)
                for index, page in enumerate(pdf)
            ]

        return ParsedDocument(document=document, metadata=metadata, pages=pages)

    def _extract_metadata(self, raw_metadata: dict, page_count: int) -> PDFMetadata:
        return PDFMetadata(
            title=self._clean_optional_metadata(raw_metadata.get("title")),
            author=self._clean_optional_metadata(raw_metadata.get("author")),
            subject=self._clean_optional_metadata(raw_metadata.get("subject")),
            creator=self._clean_optional_metadata(raw_metadata.get("creator")),
            producer=self._clean_optional_metadata(raw_metadata.get("producer")),
            page_count=page_count,
        )

    def _extract_page(self, document_id: str, page_number: int, page) -> DocumentPage:
        text = page.get_text("text").strip()
        char_count = len(text)
        word_count = len(text.split())

        return DocumentPage(
            document_id=document_id,
            page_number=page_number,
            text=text,
            extraction_method="pymupdf",
            char_count=char_count,
            word_count=word_count,
            needs_ocr=self._needs_ocr(char_count, word_count),
        )

    def _needs_ocr(self, char_count: int, word_count: int) -> bool:
        return char_count < self.min_chars_for_text_page or word_count < self.min_words_for_text_page

    def _clean_optional_metadata(self, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        return cleaned or None
