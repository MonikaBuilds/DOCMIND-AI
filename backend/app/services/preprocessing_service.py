import re
from collections import Counter

from app.domain.documents import DocumentPage, ParsedDocument


class PreprocessingService:
    """Cleans parsed page text while preserving page-level metadata."""

    def clean_document(self, parsed_document: ParsedDocument) -> ParsedDocument:
        repeated_lines = self._find_repeated_edge_lines(parsed_document.pages)
        cleaned_pages = [
            self._clean_page(page, repeated_lines)
            for page in parsed_document.pages
        ]

        return ParsedDocument(
            document=parsed_document.document,
            metadata=parsed_document.metadata,
            pages=cleaned_pages,
        )

    def clean_text(self, text: str, repeated_lines: set[str] | None = None) -> str:
        repeated_lines = repeated_lines or set()
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._remove_repeated_lines(text, repeated_lines)
        text = self._repair_hyphenated_line_breaks(text)
        text = self._join_soft_line_breaks(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    def _clean_page(self, page: DocumentPage, repeated_lines: set[str]) -> DocumentPage:
        cleaned_text = self.clean_text(page.text, repeated_lines)

        return DocumentPage(
            document_id=page.document_id,
            page_number=page.page_number,
            text=cleaned_text,
            extraction_method=page.extraction_method,
            char_count=len(cleaned_text),
            word_count=len(cleaned_text.split()),
            needs_ocr=page.needs_ocr,
        )

    def _find_repeated_edge_lines(self, pages: list[DocumentPage]) -> set[str]:
        if len(pages) < 3:
            return set()

        candidates: list[str] = []
        for page in pages:
            lines = self._meaningful_lines(page.text)
            if not lines:
                continue
            candidates.append(lines[0])
            if len(lines) > 1:
                candidates.append(lines[-1])

        counts = Counter(candidates)
        minimum_repeats = max(2, len(pages) // 2)
        return {
            line
            for line, count in counts.items()
            if count >= minimum_repeats and len(line) <= 120
        }

    def _remove_repeated_lines(self, text: str, repeated_lines: set[str]) -> str:
        if not repeated_lines:
            return text

        lines = [
            line
            for line in text.split("\n")
            if line.strip() not in repeated_lines
        ]
        return "\n".join(lines)

    def _repair_hyphenated_line_breaks(self, text: str) -> str:
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def _join_soft_line_breaks(self, text: str) -> str:
        text = re.sub(r"(?<![.!?:;])\n(?!\n|[-*•\d]+\s)", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text)

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text

    def _meaningful_lines(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]
