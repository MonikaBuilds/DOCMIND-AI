from pathlib import Path


class PyMuPDFPageRenderer:
    """Renders PDF pages to PNG bytes so OCR engines can read scanned pages."""

    def render_page_to_png(self, pdf_path: Path, page_number: int, zoom: float = 2.0) -> bytes:
        import fitz

        with fitz.open(pdf_path) as pdf:
            page = pdf[page_number - 1]
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix)
            return pixmap.tobytes("png")
