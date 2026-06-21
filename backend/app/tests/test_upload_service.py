import pytest
import asyncio

from app.core.exceptions import UnsupportedFileTypeError
from app.infrastructure.file_storage.local import LocalFileStorage
from app.services.upload_service import UploadService


class FakeUploadFile:
    def __init__(self, filename: str, content_type: str, content: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self) -> bytes:
        return self._content


def test_save_pdf_persists_valid_pdf(tmp_path):
    service = UploadService(LocalFileStorage(str(tmp_path)))
    upload = FakeUploadFile("paper.pdf", "application/pdf", b"%PDF-1.7\ncontent")

    document = asyncio.run(service.save_pdf(upload))

    assert document.filename == "paper.pdf"
    assert document.content_type == "application/pdf"
    assert document.size_bytes == len(b"%PDF-1.7\ncontent")
    assert document.source_path.exists()
    assert document.source_path.suffix == ".pdf"


def test_save_pdf_rejects_non_pdf_extension(tmp_path):
    service = UploadService(LocalFileStorage(str(tmp_path)))
    upload = FakeUploadFile("notes.txt", "application/pdf", b"%PDF-1.7\ncontent")

    with pytest.raises(UnsupportedFileTypeError):
        asyncio.run(service.save_pdf(upload))


def test_save_pdf_rejects_invalid_pdf_signature(tmp_path):
    service = UploadService(LocalFileStorage(str(tmp_path)))
    upload = FakeUploadFile("paper.pdf", "application/pdf", b"not a pdf")

    with pytest.raises(UnsupportedFileTypeError):
        asyncio.run(service.save_pdf(upload))
