from io import BytesIO

import fitz
from fastapi.testclient import TestClient

from app.domain.retrieval import RetrievedChunk
from app.infrastructure.llm.base import MockLLMProvider
from app.core.runtime_store import runtime_store
from app.main import create_app


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float("citations" in text.lower()), float(len(text))] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float("citations" in query.lower() or "grounded" in query.lower()), float(len(query))]


class FakeVectorStore:
    def __init__(self) -> None:
        self.stored = []

    def add_chunks(self, chunks, embeddings) -> None:
        self.stored = list(zip(chunks, embeddings))

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
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
            for chunk, _ in self.stored[:top_k]
        ]


def make_pdf_bytes(text: str) -> bytes:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    buffer = BytesIO()
    pdf.save(buffer)
    pdf.close()
    return buffer.getvalue()


def test_upload_process_search_and_chat_api_flow(monkeypatch):
    runtime_store.clear()
    vector_store = FakeVectorStore()
    monkeypatch.setattr("app.api.routes.documents.build_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr("app.api.routes.documents.build_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.api.routes.documents.build_llm_provider", lambda: MockLLMProvider("Mock answer."))
    monkeypatch.setattr("app.api.routes.search.build_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr("app.api.routes.search.build_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.api.routes.chat.build_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr("app.api.routes.chat.build_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.api.routes.chat.build_llm_provider", lambda: MockLLMProvider("RAG answer from API."))

    client = TestClient(create_app())
    pdf_bytes = make_pdf_bytes(
        "Retrieval augmented generation uses document chunks and citations for grounded answers."
    )

    upload_response = client.post(
        "/api/v1/upload/pdfs",
        files={"files": ("rag.pdf", pdf_bytes, "application/pdf")},
    )

    assert upload_response.status_code == 201
    document_id = upload_response.json()["documents"][0]["document_id"]

    process_response = client.post(
        f"/api/v1/documents/{document_id}/process",
        json={"chunk_size": 40, "overlap": 5},
    )

    assert process_response.status_code == 200
    assert process_response.json()["chunk_count"] >= 1

    search_response = client.post(
        "/api/v1/search",
        json={"query": "citations", "top_k": 3},
    )

    assert search_response.status_code == 200
    assert search_response.json()["count"] >= 1

    chat_response = client.post(
        "/api/v1/chat",
        json={"question": "How does RAG stay grounded?", "session_id": "test-session", "top_k": 3},
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["answer"] == "RAG answer from API."
    assert chat_response.json()["retrieved_count"] >= 1
