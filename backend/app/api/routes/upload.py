from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import UnsupportedFileTypeError
from app.core.runtime_store import runtime_store
from app.infrastructure.file_storage.local import LocalFileStorage
from app.schemas.upload import MultiUploadResponse, UploadedDocumentResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/upload", tags=["upload"])


def get_upload_service() -> UploadService:
    settings = get_settings()
    return UploadService(LocalFileStorage(settings.upload_dir))


@router.get("/status")
def upload_module_status() -> dict[str, str]:
    return {
        "module": "upload",
        "status": "implemented",
        "next_step": "Phase 6 will parse uploaded PDFs with PyMuPDF.",
    }


@router.post(
    "/pdfs",
    response_model=MultiUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdfs(files: list[UploadFile] = File(...)) -> MultiUploadResponse:
    service = get_upload_service()

    try:
        documents = await service.save_many_pdfs(files)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    for document in documents:
        runtime_store.add_document(document)

    responses = [
        UploadedDocumentResponse(
            document_id=document.document_id,
            original_filename=document.filename,
            stored_filename=document.source_path.name,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            storage_path=str(document.source_path),
            status="uploaded",
        )
        for document in documents
    ]

    return MultiUploadResponse(documents=responses, count=len(responses))
