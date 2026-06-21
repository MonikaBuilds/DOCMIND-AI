from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    document_id: str
    filename: str
    source_path: Path
    content_type: str
    size_bytes: int


@dataclass(frozen=True)
class DocumentPage:
    document_id: str
    page_number: int
    text: str
    extraction_method: str
    char_count: int
    word_count: int
    needs_ocr: bool


@dataclass(frozen=True)
class PDFMetadata:
    title: str | None
    author: str | None
    subject: str | None
    creator: str | None
    producer: str | None
    page_count: int


@dataclass(frozen=True)
class ParsedDocument:
    document: Document
    metadata: PDFMetadata
    pages: list[DocumentPage]
