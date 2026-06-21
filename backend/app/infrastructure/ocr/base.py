from abc import ABC, abstractmethod


class OCRProvider(ABC):
    """Interface for PaddleOCR, EasyOCR, or fallback OCR engines."""

    @abstractmethod
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        raise NotImplementedError


class EasyOCRProvider(OCRProvider):
    """Future EasyOCR implementation. The dependency is installed in Phase 7 setup."""

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        raise NotImplementedError("Install and configure EasyOCR before using this provider.")


class PaddleOCRProvider(OCRProvider):
    """Future PaddleOCR implementation for stronger production OCR."""

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        raise NotImplementedError("Install and configure PaddleOCR before using this provider.")
