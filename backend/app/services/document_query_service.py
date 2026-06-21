from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk


class DocumentQueryService:
    """Routes document-level questions to representative document context."""

    def is_overview_question(self, query: str) -> bool:
        normalized = " ".join(query.lower().strip().split())
        overview_patterns = [
            "what is this document about",
            "summarize",
            "summary",
            "main idea",
            "main topic",
            "overview",
            "key points",
            "tl;dr",
            "tldr",
            "core theme",
            "essential information",
        ]
        # Keep original patterns but also check for broad keywords
        if any(pattern in normalized for pattern in overview_patterns):
            return True
        
        # Check for very short broad questions
        if normalized in ["what is this?", "explain this", "what is it?"]:
            return True
            
        return False

    def is_meta_question(self, query: str) -> bool:
        """Detects if the user is asking about the system state or document list."""
        normalized = " ".join(query.lower().strip().split())
        # Remove 'the', 'a', 'an' for more robust matching
        essential = re.sub(r"\b(the|a|an)\b", "", normalized).strip()
        essential = " ".join(essential.split())
        
        meta_patterns = [
            "how many pdfs",
            "how many documents",
            "how many files",
            "what files",
            "list documents",
            "list files",
            "total number of",
            "what is uploaded",
            "which documents",
            "filename",
            "names of",
            "page count",
            "how many pages",
        ]
        return any(pattern in essential for pattern in meta_patterns) or any(pattern in normalized for pattern in meta_patterns)

    def overview_chunks(
        self,
        chunks: list[DocumentChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        ordered_chunks = sorted(chunks, key=lambda chunk: (chunk.page_number, chunk.chunk_index))
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                text=chunk.text,
                score=1.0,
                page_number=chunk.page_number,
                source=chunk.source,
                heading=chunk.heading,
            )
            for chunk in ordered_chunks[:top_k]
        ]
