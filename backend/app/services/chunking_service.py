from dataclasses import dataclass

from app.domain.chunks import DocumentChunk
from app.domain.documents import DocumentPage, ParsedDocument


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 180
    overlap_words: int = 40


class ChunkingService:
    """Splits cleaned page text into retrieval-ready, page-aware chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self._validate_config()

    def chunk_document(self, parsed_document: ParsedDocument) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page in parsed_document.pages:
            page_chunks = self._chunk_page(parsed_document, page, start_index=chunk_index)
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        return chunks

    def _chunk_page(
        self,
        parsed_document: ParsedDocument,
        page: DocumentPage,
        start_index: int,
    ) -> list[DocumentChunk]:
        words = page.text.split()
        if not words:
            return []

        chunks: list[DocumentChunk] = []
        start = 0

        while start < len(words):
            end = min(start + self.config.max_words, len(words))
            chunk_words = words[start:end]
            chunk_index = start_index + len(chunks)
            chunk_id = self._build_chunk_id(
                parsed_document.document.document_id,
                page.page_number,
                chunk_index,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=parsed_document.document.document_id,
                    filename=parsed_document.document.filename,
                    text=" ".join(chunk_words),
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    source=self._build_source(parsed_document.document.filename, page.page_number),
                    heading=self._detect_heading(page),
                )
            )

            if end == len(words):
                break

            start = end - self.config.overlap_words

        return chunks

    def _build_chunk_id(self, document_id: str, page_number: int, chunk_index: int) -> str:
        return f"{document_id}:p{page_number}:c{chunk_index}"

    def _build_source(self, filename: str, page_number: int) -> str:
        return f"{filename}#page={page_number}"

    def _detect_heading(self, page: DocumentPage) -> str | None:
        for line in page.text.splitlines():
            cleaned = line.strip()
            if 0 < len(cleaned) <= 80 and len(cleaned.split()) <= 12:
                return cleaned
        return None

    def _validate_config(self) -> None:
        if self.config.max_words <= 0:
            raise ValueError("max_words must be greater than zero.")

        if self.config.overlap_words < 0:
            raise ValueError("overlap_words cannot be negative.")

        if self.config.overlap_words >= self.config.max_words:
            raise ValueError("overlap_words must be smaller than max_words.")
