from app.domain.chunks import DocumentChunk
from app.services.embedding_service import EmbeddingService


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        self.query_calls.append(query)
        return [float(len(query)), 2.0]


class TextOnlyFakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[3.0, float(len(text))] for text in texts]


def make_chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"doc:p1:c{index}",
        document_id="doc",
        filename="embeddings.pdf",
        text=text,
        page_number=1,
        chunk_index=index,
        source="embeddings.pdf#page=1",
        heading=None,
    )


def test_embed_chunks_uses_chunk_texts():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)
    chunks = [make_chunk(0, "short text"), make_chunk(1, "longer chunk text")]

    embeddings = service.embed_chunks(chunks)

    assert provider.text_calls == [["short text", "longer chunk text"]]
    assert embeddings == [[10.0, 1.0], [17.0, 1.0]]


def test_embed_query_prefers_provider_query_method():
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)

    embedding = service.embed_query("What is RAG?")

    assert provider.query_calls == ["What is RAG?"]
    assert embedding == [12.0, 2.0]


def test_embed_query_falls_back_to_text_embedding():
    service = EmbeddingService(TextOnlyFakeEmbeddingProvider())

    embedding = service.embed_query("fallback")

    assert embedding == [3.0, 8.0]
