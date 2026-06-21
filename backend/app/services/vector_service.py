from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk
from app.infrastructure.vector_store.base import VectorStore


class VectorService:
    """Persists and searches chunk embeddings through a vector store interface."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        self.vector_store.add_chunks(chunks, embeddings)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        return self.vector_store.search(query_embedding, top_k, document_ids=document_ids)
