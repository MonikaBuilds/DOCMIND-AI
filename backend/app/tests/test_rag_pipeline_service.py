from pathlib import Path

import fitz

from app.domain.retrieval import RetrievedChunk
from app.infrastructure.llm.base import MockLLMProvider
from app.services.chunking_service import ChunkingConfig
from app.services.rag_pipeline_service import RAGPipelineService


class FakeEmbeddingProvider:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float("rag" in text.lower()), float(len(text))] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float("rag" in query.lower()), float(len(query))]


class FakeVectorStore:
    def __init__(self) -> None:
        self.stored = []

    def add_chunks(self, chunks, embeddings) -> None:
        self.stored = list(zip(chunks, embeddings))

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        results = []
        for chunk, embedding in self.stored[:top_k]:
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    text=chunk.text,
                    score=1.0 if embedding[0] == query_embedding[0] else 0.2,
                    page_number=chunk.page_number,
                    source=chunk.source,
                    heading=chunk.heading,
                )
            )
        return results


def create_pdf(path: Path, text: str) -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_rag_pipeline_indexes_and_answers_question(tmp_path):
    from app.domain.documents import Document

    pdf_path = tmp_path / "rag.pdf"
    create_pdf(pdf_path, "RAG uses retrieval augmented generation with citations.")
    document = Document(
        document_id="doc-rag",
        filename="rag.pdf",
        source_path=pdf_path,
        content_type="application/pdf",
        size_bytes=pdf_path.stat().st_size,
    )
    vector_store = FakeVectorStore()
    pipeline = RAGPipelineService.from_providers(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=vector_store,
        llm_provider=MockLLMProvider("RAG answer from retrieved context."),
        chunk_config=ChunkingConfig(max_words=50, overlap_words=5),
    )

    indexed = pipeline.index_document(document)
    answer, retrieved = pipeline.answer_question("What is RAG?", top_k=2)

    assert indexed.chunk_count == 1
    assert len(vector_store.stored) == 1
    assert answer.answer == "RAG answer from retrieved context."
    assert answer.citations == ["rag.pdf, page 1, RAG uses retrieval augmented generation with citations."]
    assert retrieved[0].document_id == "doc-rag"
