from dataclasses import replace

from app.domain.citations import Citation
from app.domain.retrieval import RetrievedChunk


class CitationService:
    """Creates deterministic page-aware citations from retrieved chunks."""

    def build_citations(self, retrieved_chunks: list[RetrievedChunk]) -> list[Citation]:
        citation_by_key: dict[tuple[str, int, str | None], Citation] = {}

        for chunk in retrieved_chunks:
            key = (chunk.document_id, chunk.page_number, chunk.heading)
            existing = citation_by_key.get(key)

            if existing:
                citation_by_key[key] = replace(
                    existing,
                    chunk_ids=existing.chunk_ids + (chunk.chunk_id,),
                )
                continue

            citation_by_key[key] = Citation(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                source=chunk.source,
                heading=chunk.heading,
                label=self.format_label(chunk),
                chunk_ids=(chunk.chunk_id,),
            )

        return list(citation_by_key.values())

    def format_label(self, chunk: RetrievedChunk) -> str:
        if chunk.heading:
            return f"{chunk.filename}, page {chunk.page_number}, {chunk.heading}"
        return f"{chunk.filename}, page {chunk.page_number}"

    def citation_labels(self, retrieved_chunks: list[RetrievedChunk]) -> list[str]:
        return [citation.label for citation in self.build_citations(retrieved_chunks)]
