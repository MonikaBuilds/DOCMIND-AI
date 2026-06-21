from pathlib import Path

import pytest

from app.domain.documents import Document, DocumentPage, PDFMetadata, ParsedDocument
from app.services.chunking_service import ChunkingConfig, ChunkingService


def make_parsed_document(page_texts: list[str]) -> ParsedDocument:
    return ParsedDocument(
        document=Document(
            document_id="doc-chunk",
            filename="chunking.pdf",
            source_path=Path("uploads/doc-chunk.pdf"),
            content_type="application/pdf",
            size_bytes=100,
        ),
        metadata=PDFMetadata(
            title=None,
            author=None,
            subject=None,
            creator=None,
            producer=None,
            page_count=len(page_texts),
        ),
        pages=[
            DocumentPage(
                document_id="doc-chunk",
                page_number=index + 1,
                text=text,
                extraction_method="pymupdf",
                char_count=len(text),
                word_count=len(text.split()),
                needs_ocr=False,
            )
            for index, text in enumerate(page_texts)
        ],
    )


def test_chunk_document_preserves_page_metadata():
    parsed = make_parsed_document(["Introduction\nRAG systems need citations."])

    chunks = ChunkingService(ChunkingConfig(max_words=20, overlap_words=5)).chunk_document(parsed)

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-chunk"
    assert chunks[0].filename == "chunking.pdf"
    assert chunks[0].page_number == 1
    assert chunks[0].source == "chunking.pdf#page=1"
    assert chunks[0].chunk_id == "doc-chunk:p1:c0"
    assert chunks[0].heading == "Introduction"


def test_chunk_document_uses_word_overlap():
    text = " ".join(f"word{i}" for i in range(1, 13))
    parsed = make_parsed_document([text])

    chunks = ChunkingService(ChunkingConfig(max_words=5, overlap_words=2)).chunk_document(parsed)

    assert [chunk.text for chunk in chunks] == [
        "word1 word2 word3 word4 word5",
        "word4 word5 word6 word7 word8",
        "word7 word8 word9 word10 word11",
        "word10 word11 word12",
    ]


def test_chunk_document_skips_empty_pages():
    parsed = make_parsed_document(["", "Useful content for retrieval."])

    chunks = ChunkingService(ChunkingConfig(max_words=10, overlap_words=2)).chunk_document(parsed)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 0


def test_chunking_config_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        ChunkingService(ChunkingConfig(max_words=10, overlap_words=10))
