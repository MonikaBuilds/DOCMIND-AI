import pytest

from app.core.exceptions import ProviderConfigurationError
from app.domain.chat import GroundedPrompt
from app.infrastructure.llm.base import GeminiLLMProvider, MockLLMProvider, OpenAILLMProvider
from app.services.llm_service import LLMService


def make_prompt() -> GroundedPrompt:
    return GroundedPrompt(
        system_prompt="Use only document context.",
        user_prompt="Question: What is RAG?",
    )


def test_llm_service_uses_provider_response():
    provider = MockLLMProvider("Grounded answer with citation.")
    service = LLMService(provider)
    prompt = make_prompt()

    response = service.generate(prompt)

    assert response == "Grounded answer with citation."
    assert provider.prompts == [prompt]


def test_llm_service_handles_empty_provider_response():
    service = LLMService(MockLLMProvider("   "))

    response = service.generate(make_prompt())

    assert response == "The model did not return an answer."


def test_gemini_provider_requires_api_key_before_loading_dependency():
    provider = GeminiLLMProvider(api_key=None)

    with pytest.raises(ProviderConfigurationError):
        provider.generate(make_prompt())


def test_openai_provider_requires_api_key_before_loading_dependency():
    provider = OpenAILLMProvider(api_key=None)

    with pytest.raises(ProviderConfigurationError):
        provider.generate(make_prompt())
