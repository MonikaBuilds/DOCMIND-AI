import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import DocMindError
from app.core.providers import build_embedding_provider, build_llm_provider, build_vector_store
from app.core.runtime_store import runtime_store
from app.schemas.documents import (
    DocumentRecordResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
)
from app.services.chunking_service import ChunkingConfig, ChunkingService
from app.services.metadata_service import MetadataService
from app.services.parsing_service import ParsingService
from app.services.preprocessing_service import PreprocessingService
from app.services.rag_pipeline_service import RAGPipelineService
from app.services.vector_service import VectorService

router = APIRouter(prefix="/documents", tags=["documents"])
logger = logging.getLogger(__name__)


@router.get("/status")
def documents_module_status() -> dict[str, str]:
    return {
        "module": "documents",
        "status": "api_ready",
        "next_step": "Use POST /documents/{document_id}/process after uploading a PDF.",
    }


@router.get("", response_model=list[DocumentRecordResponse])
def list_documents() -> list[DocumentRecordResponse]:
    return [
        _record_to_response(record)
        for record in runtime_store.list_records()
    ]


@router.post("/{document_id}/process", response_model=ProcessDocumentResponse)
def process_document(
    document_id: str,
    request: ProcessDocumentRequest,
) -> ProcessDocumentResponse:
    record = runtime_store.get_record(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    if request.overlap >= request.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="overlap must be smaller than chunk_size",
        )

    try:
        pipeline = RAGPipelineService.from_providers(
            embedding_provider=build_embedding_provider(),
            vector_store=build_vector_store(),
            llm_provider=build_llm_provider(),
            chunk_config=ChunkingConfig(
                max_words=request.chunk_size,
                overlap_words=request.overlap,
            ),
        )
        indexed = pipeline.index_document(record.document)
    except DocMindError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while processing document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not prepare this document: {exc}",
        ) from exc

    # The pipeline already performed all the work, just update the store using results.
    parsed_document = ParsingService().parse_document(record.document)
    cleaned_document = PreprocessingService().clean_document(parsed_document)
    
    runtime_store.save_processed_document(
        document_id,
        cleaned_document,
        indexed.chunks,
        ai_summary=indexed.ai_summary
    )

    return ProcessDocumentResponse(
        document_id=document_id,
        filename=record.document.filename,
        page_count=indexed.page_count,
        chunk_count=indexed.chunk_count,
        status="processed",
        ai_summary=indexed.ai_summary,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str) -> None:
    record = runtime_store.delete_document(document_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )

    try:
        VectorService(build_vector_store()).delete_document(document_id)
    except DocMindError:
        logger.exception("Could not delete vectors for document %s", document_id)

    try:
        source_path = Path(record.document.source_path)
        if source_path.exists() and source_path.is_file():
            source_path.unlink()
    except OSError:
        logger.exception("Could not delete source file for document %s", document_id)


def _record_to_response(record) -> DocumentRecordResponse:
    return DocumentRecordResponse(
        document_id=record.document.document_id,
        filename=record.document.filename,
        content_type=record.document.content_type,
        size_bytes=record.document.size_bytes,
        storage_path=str(record.document.source_path),
        status="processed" if record.chunks else "uploaded",
        page_count=record.parsed_document.metadata.page_count if record.parsed_document else None,
        chunk_count=len(record.chunks),
        ai_summary=record.ai_summary,
    )
