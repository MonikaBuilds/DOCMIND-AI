from app.domain.chat import GroundedPrompt
from app.infrastructure.llm.base import LLMProvider


class LLMService:
    """Calls the configured LLM provider with grounded prompts."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    def generate(self, prompt: GroundedPrompt) -> str:
        response = self.llm_provider.generate(prompt).strip()
        if not response:
            return "The model did not return an answer."
        return response
