from app.domain.chat import GroundedPrompt
from app.infrastructure.embeddings.base import HashEmbeddingProvider
from app.infrastructure.llm.base import ExtractiveLLMProvider


def test_hash_embedding_provider_returns_normalized_vectors():
    provider = HashEmbeddingProvider(dimensions=16)

    embeddings = provider.embed_texts(["document intelligence", "document intelligence"])

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 16
    assert embeddings[0] == embeddings[1]


def test_extractive_llm_provider_answers_from_source_text():
    provider = ExtractiveLLMProvider()
    prompt = GroundedPrompt(
        system_prompt="Use only sources.",
        user_prompt=(
            "Document Context:\n"
            "[Source 1]\n"
            "Citation: report.pdf, page 2\n"
            "Score: 1.0000\n"
            "Text:\n"
            "The project was about building a document assistant for question answering. "
            "It supports summaries and page references.\n\n"
            "Question:\n"
            "What was the project about?\n\n"
            "Answer:"
        ),
    )

    answer = provider.generate(prompt)

    assert "document assistant" in answer
    assert "report.pdf, page 2" in answer


def test_extractive_llm_provider_handles_broad_project_question():
    provider = ExtractiveLLMProvider()
    prompt = GroundedPrompt(
        system_prompt="Use only sources.",
        user_prompt=(
            "Document Context:\n"
            "[Source 1]\n"
            "Citation: brain-dead.pdf, page 1\n"
            "Score: 0.3000\n"
            "Text:\n"
            "This submission presents an AI and data analysis solution for hackathon evaluation. "
            "The work includes document processing, model-driven insights, and a user-facing assistant.\n\n"
            "Question:\n"
            "What's the project about?\n\n"
            "Answer:"
        ),
    )

    answer = provider.generate(prompt)

    assert "AI and data analysis" in answer
    assert "brain-dead.pdf, page 1" in answer
