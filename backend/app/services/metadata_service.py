from dataclasses import replace

from app.domain.chunks import DocumentChunk
from app.domain.documents import ParsedDocument
from app.domain.metadata import DocumentMetadataSummary


class MetadataService:
    """Builds consistent metadata used by vector stores, citations, and APIs."""

    def chunk_to_vector_metadata(self, chunk: DocumentChunk) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "source": chunk.source,
            "citation": self.citation_label(chunk),
        }

        if chunk.heading:
            metadata["heading"] = chunk.heading

        return metadata

    def chunks_to_vector_metadata(self, chunks: list[DocumentChunk]) -> list[dict[str, str | int]]:
        return [self.chunk_to_vector_metadata(chunk) for chunk in chunks]

    def citation_label(self, chunk: DocumentChunk) -> str:
        if chunk.heading:
            return f"{chunk.filename}, page {chunk.page_number}, {chunk.heading}"
        return f"{chunk.filename}, page {chunk.page_number}"

    def propagate_headings(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        last_heading_by_document: dict[str, str] = {}
        updated_chunks: list[DocumentChunk] = []

        for chunk in chunks:
            if chunk.heading:
                last_heading_by_document[chunk.document_id] = chunk.heading
                updated_chunks.append(chunk)
                continue

            inherited_heading = last_heading_by_document.get(chunk.document_id)
            if inherited_heading:
                updated_chunks.append(replace(chunk, heading=inherited_heading))
            else:
                updated_chunks.append(chunk)

        return updated_chunks

    def document_summary(
        self,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
    ) -> DocumentMetadataSummary:
        relevant_chunks = [
            chunk
            for chunk in chunks
            if chunk.document_id == parsed_document.document.document_id
        ]

        return DocumentMetadataSummary(
            document_id=parsed_document.document.document_id,
            filename=parsed_document.document.filename,
            page_count=parsed_document.metadata.page_count,
            chunk_count=len(relevant_chunks),
            title=parsed_document.metadata.title,
            author=parsed_document.metadata.author,
        )
