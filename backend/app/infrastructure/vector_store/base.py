from abc import ABC, abstractmethod
from math import sqrt

from app.core.exceptions import ProviderConfigurationError
from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk
from app.services.metadata_service import MetadataService


class VectorStore(ABC):
    """Interface that keeps retrieval logic independent from ChromaDB."""

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation hidden behind the vector store interface."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "docmind_chunks",
        metadata_service: MetadataService | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.metadata_service = metadata_service or MetadataService()
        self._collection = None

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        collection = self._get_collection()
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=self.metadata_service.chunks_to_vector_metadata(chunks),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []

        collection = self._get_collection()
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._to_retrieved_chunks(results)

    def delete_document(self, document_id: str) -> None:
        collection = self._get_collection()
        collection.delete(where={"document_id": document_id})

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as exc:
            raise ProviderConfigurationError(
                "chromadb is required for ChromaVectorStore. "
                "Install backend requirements before using vector storage."
            ) from exc

        client = chromadb.PersistentClient(path=self.persist_directory)
        self._collection = client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def _to_retrieved_chunks(self, results: dict) -> list[RetrievedChunk]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(metadata["chunk_id"]),
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    text=text,
                    score=self._distance_to_score(float(distance)),
                    page_number=int(metadata["page_number"]),
                    source=str(metadata["source"]),
                    heading=metadata.get("heading"),
                )
            )

        return retrieved

    def _distance_to_score(self, distance: float) -> float:
        return 1.0 / (1.0 + distance)


class InMemoryVectorStore(VectorStore):
    """Dependency-free vector store for local demos and development fallback."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[DocumentChunk, list[float]]] = {}

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length.")

        for chunk, embedding in zip(chunks, embeddings):
            self._records[chunk.chunk_id] = (chunk, embedding)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or not self._records:
            return []

        allowed_document_ids = set(document_ids or [])
        records = [
            (chunk, embedding)
            for chunk, embedding in self._records.values()
            if not allowed_document_ids or chunk.document_id in allowed_document_ids
        ]

        scored = [
            (self._cosine_similarity(query_embedding, embedding), chunk)
            for chunk, embedding in records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                text=chunk.text,
                score=score,
                page_number=chunk.page_number,
                source=chunk.source,
                heading=chunk.heading,
            )
            for score, chunk in scored[:top_k]
        ]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def delete_document(self, document_id: str) -> None:
        chunk_ids = [
            chunk_id
            for chunk_id, (chunk, _) in self._records.items()
            if chunk.document_id == document_id
        ]
        for chunk_id in chunk_ids:
            self._records.pop(chunk_id, None)
