from app.domain.documents import ParsedDocument
from app.services.llm_service import LLMService


class SummaryService:
    """Generates concise AI summaries for documents to aid user understanding."""

    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def generate_document_summary(self, parsed_document: ParsedDocument) -> str:
        # Use content from the first few pages for the summary
        summary_text = ""
        for page in parsed_document.pages[:3]:
            summary_text += page.text + "\n"
        
        # Limit text size for the LLM
        truncated_text = summary_text[:4000]
        
        prompt = (
            "You are an AI document analyst. Summarize the following document content in exactly 3-4 bullet points. "
            "Focus on the main topic, key objectives, and any significant findings. "
            "Be professional and concise.\n\n"
            f"Content:\n{truncated_text}\n\n"
            "Summary:"
        )
        
        try:
            return self.llm_service.generate_simple(prompt)
        except Exception:
            return "Summary currently unavailable for this document."

    def generate_simple_title(self, filename: str, content: str) -> str:
        """Generates a human-friendly title if the PDF metadata title is missing."""
        truncated_content = content[:1000]
        prompt = (
            f"Based on the filename '{filename}' and the following snippet, generate a short, "
            "descriptive title for this document (max 5 words).\n\n"
            f"Snippet: {truncated_content}\n\n"
            "Title:"
        )
        try:
            return self.llm_service.generate_simple(prompt).strip(' "')
        except Exception:
            return filename
