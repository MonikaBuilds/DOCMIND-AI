class DocMindError(Exception):
    """Base exception for expected DocMind AI application errors."""


class UnsupportedFileTypeError(DocMindError):
    """Raised when an uploaded file type is not supported."""


class DocumentProcessingError(DocMindError):
    """Raised when a document cannot be processed."""


class ProviderConfigurationError(DocMindError):
    """Raised when an embedding, LLM, OCR, or vector provider is misconfigured."""
