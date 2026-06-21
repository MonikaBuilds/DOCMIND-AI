from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.core.exceptions import UnsupportedFileTypeError
from app.domain.documents import Document
from app.infrastructure.file_storage.local import LocalFileStorage


class UploadedFileLike(Protocol):
    filename: str | None
    content_type: str | None

    async def read(self) -> bytes:
        raise NotImplementedError


class UploadService:
    """Validates PDF uploads and persists original files for later processing."""

    allowed_content_types = {"application/pdf", "application/x-pdf"}

    def __init__(self, storage: LocalFileStorage) -> None:
        self.storage = storage

    async def save_pdf(self, upload_file: UploadedFileLike) -> Document:
        content = await upload_file.read()
        self._validate_pdf(upload_file.filename, upload_file.content_type, content)

        document_id = str(uuid4())
        original_filename = Path(upload_file.filename or "document.pdf").name
        stored_filename = f"{document_id}.pdf"
        stored_path = self.storage.save_bytes(stored_filename, content)

        return Document(
            document_id=document_id,
            filename=original_filename,
            source_path=stored_path,
            content_type=upload_file.content_type or "application/pdf",
            size_bytes=len(content),
        )

    async def save_many_pdfs(self, upload_files: list[UploadedFileLike]) -> list[Document]:
        return [await self.save_pdf(upload_file) for upload_file in upload_files]

    def _validate_pdf(self, filename: str | None, content_type: str | None, content: bytes) -> None:
        safe_name = Path(filename or "").name

        if not safe_name.lower().endswith(".pdf"):
            raise UnsupportedFileTypeError("Only PDF files with a .pdf extension are supported.")

        if content_type and content_type not in self.allowed_content_types:
            raise UnsupportedFileTypeError(f"Unsupported content type: {content_type}")

        if not content.startswith(b"%PDF"):
            raise UnsupportedFileTypeError("Uploaded file does not look like a valid PDF.")
