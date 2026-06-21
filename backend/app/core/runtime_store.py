from dataclasses import dataclass, field

from app.domain.chunks import DocumentChunk
from app.domain.documents import Document, ParsedDocument


@dataclass
class DocumentRuntimeRecord:
    document: Document
    parsed_document: ParsedDocument | None = None
    chunks: list[DocumentChunk] = field(default_factory=list)


class RuntimeStore:
    """In-memory store for local API development before database persistence."""

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRuntimeRecord] = {}

    def add_document(self, document: Document) -> None:
        self._documents[document.document_id] = DocumentRuntimeRecord(document=document)

    def get_document(self, document_id: str) -> Document | None:
        record = self._documents.get(document_id)
        return record.document if record else None

    def get_record(self, document_id: str) -> DocumentRuntimeRecord | None:
        return self._documents.get(document_id)

    def delete_document(self, document_id: str) -> DocumentRuntimeRecord | None:
        return self._documents.pop(document_id, None)

    def list_records(self) -> list[DocumentRuntimeRecord]:
        return list(self._documents.values())

    def save_processed_document(
        self,
        document_id: str,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
    ) -> None:
        record = self._documents[document_id]
        record.parsed_document = parsed_document
        record.chunks = chunks

    def get_chunks(self, document_ids: list[str] | None = None) -> list[DocumentChunk]:
        records = self.list_records()
        if document_ids:
            allowed = set(document_ids)
            records = [
                record
                for record in records
                if record.document.document_id in allowed
            ]

        chunks: list[DocumentChunk] = []
        for record in records:
            chunks.extend(record.chunks)
        return chunks

    def clear(self) -> None:
        self._documents.clear()


runtime_store = RuntimeStore()
