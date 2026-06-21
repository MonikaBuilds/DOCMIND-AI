from app.domain.chunks import DocumentChunk
from app.infrastructure.embeddings.base import EmbeddingProvider


class EmbeddingService:
    """Generates embeddings for chunks and user queries through a provider interface."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[list[float]]:
        texts = [chunk.text for chunk in chunks]
        return self.embedding_provider.embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        provider_embed_query = getattr(self.embedding_provider, "embed_query", None)
        if callable(provider_embed_query):
            return provider_embed_query(query)

        embeddings = self.embedding_provider.embed_texts([query])
        if not embeddings:
            return []

        return embeddings[0]
