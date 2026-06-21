from app.domain.chat import ChatMessage, GroundedPrompt
from app.domain.metadata import DocumentMetadataSummary
from app.domain.retrieval import RetrievedChunk


class PromptService:
    """Builds strict document-grounded prompts for RAG answer generation."""

    system_prompt = (
        "You are DocMind AI, a document-grounded assistant. "
        "You have access to a manifest of uploaded documents and specific content snippets. "
        "Answer strictly using the provided document context and manifest. "
        "If you are asked about the number of documents or filenames, use the 'Available Documents' manifest. "
        "If the answer is not present in the context or manifest, say that the document does not contain enough information. "
        "Do not use outside knowledge. "
        "Cite sources using the provided citation labels."
    )

    def build_grounded_prompt(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        summaries: list[DocumentMetadataSummary] | None = None,
        conversation_history: list[ChatMessage] | None = None,
    ) -> GroundedPrompt:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question cannot be empty.")

        inventory = self.format_document_inventory(summaries or [])
        context = self.format_context(retrieved_chunks)
        history = self.format_history(conversation_history or [])
        user_prompt = self._build_user_prompt(clean_question, context, inventory, history)
        return GroundedPrompt(system_prompt=self.system_prompt, user_prompt=user_prompt)

    def format_document_inventory(self, summaries: list[DocumentMetadataSummary]) -> str:
        if not summaries:
            return "No documents are currently indexed."

        inventory_lines = [f"Total Documents: {len(summaries)}"]
        for i, summary in enumerate(summaries, start=1):
            title_str = f", Title: {summary.title}" if summary.title else ""
            inventory_lines.append(
                f"{i}. Filename: {summary.filename}{title_str} ({summary.page_count} pages, {summary.chunk_count} sections)"
            )

        return "\n".join(inventory_lines)

    def format_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "No document context was retrieved."

        context_blocks = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            citation = self._citation_label(chunk)
            heading = f"\nHeading: {chunk.heading}" if chunk.heading else ""
            context_blocks.append(
                f"[Source {index}]\n"
                f"Citation: {citation}\n"
                f"Score: {chunk.score:.4f}{heading}\n"
                f"Text:\n{chunk.text.strip()}"
            )

        return "\n\n".join(context_blocks)

    def format_history(self, conversation_history: list[ChatMessage]) -> str:
        if not conversation_history:
            return "No prior conversation."

        return "\n".join(
            f"{message.role.title()}: {message.content.strip()}"
            for message in conversation_history
            if message.content.strip()
        )

    def _build_user_prompt(self, question: str, context: str, inventory: str, history: str) -> str:
        return (
            "Instructions:\n"
            "1. Use only the document manifest and context as factual evidence.\n"
            "2. Conversation history is only for understanding follow-up questions.\n"
            "3. Include page-aware citations for claims, using the Citation labels.\n"
            "4. If context is insufficient, say: \"The provided documents do not contain enough information to answer this.\"\n\n"
            "Available Documents (Manifest):\n"
            f"{inventory}\n\n"
            "Conversation History:\n"
            f"{history}\n\n"
            "Document Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer:"
        )

    def _citation_label(self, chunk: RetrievedChunk) -> str:
        if chunk.heading:
            return f"{chunk.filename}, page {chunk.page_number}, {chunk.heading}"
        return f"{chunk.filename}, page {chunk.page_number}"
