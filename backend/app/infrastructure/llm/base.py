from abc import ABC, abstractmethod
import json
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.exceptions import ProviderConfigurationError
from app.domain.chat import GroundedPrompt


class LLMProvider(ABC):
    """Interface for Gemini, OpenAI, Anthropic, or local LLM implementations."""

    @abstractmethod
    def generate(self, prompt: GroundedPrompt) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests and local pipeline validation."""

    def __init__(self, response: str = "Mock grounded answer.") -> None:
        self.response = response
        self.prompts: list[GroundedPrompt] = []

    def generate(self, prompt: GroundedPrompt) -> str:
        self.prompts.append(prompt)
        return self.response


class ExtractiveLLMProvider(LLMProvider):
    """Quota-free answer fallback that extracts the most relevant source text."""

    def generate(self, prompt: GroundedPrompt) -> str:
        question = self._extract_question(prompt.user_prompt)
        sources = self._extract_sources(prompt.user_prompt)
        if not sources:
            return "The provided documents do not contain enough information to answer this."

        query_terms = set(self._tokens(question))
        is_broad_question = self._is_broad_question(question)
        ranked = sorted(
            sources,
            key=lambda source: self._score_source(source["text"], query_terms),
            reverse=True,
        )
        best = ranked[0]
        best_score = self._score_source(best["text"], query_terms)
        if best_score <= 0 and not is_broad_question:
            return "The provided documents do not contain enough information to answer this."

        answer_text = self._best_sentences(best["text"], query_terms, is_broad_question)
        return f"{answer_text} ({best['citation']})"

    def _extract_question(self, user_prompt: str) -> str:
        marker = "Question:"
        if marker not in user_prompt:
            return ""
        question_block = user_prompt.split(marker, 1)[1]
        return question_block.split("Answer:", 1)[0].strip()

    def _extract_sources(self, user_prompt: str) -> list[dict[str, str]]:
        blocks = re.split(r"\n\[Source \d+\]\n", "\n" + user_prompt)
        sources: list[dict[str, str]] = []
        for block in blocks:
            if "Citation:" not in block or "Text:" not in block:
                continue
            citation = block.split("Citation:", 1)[1].splitlines()[0].strip()
            text = block.split("Text:", 1)[1].split("\n\nQuestion:", 1)[0].strip()
            if text:
                sources.append({"citation": citation, "text": text})
        return sources

    def _score_source(self, text: str, query_terms: set[str]) -> int:
        if not query_terms:
            return 0
        text_terms = set(self._tokens(text))
        return len(query_terms.intersection(text_terms))

    def _best_sentences(self, text: str, query_terms: set[str], is_broad_question: bool = False) -> str:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if not sentences:
            return text[:700].strip()

        if is_broad_question and not query_terms.intersection(set(self._tokens(text))):
            return " ".join(sentences[:3])[:900].strip()

        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: self._score_source(sentence, query_terms),
            reverse=True,
        )
        selected = ranked_sentences[:2]
        answer = " ".join(selected).strip()
        return answer[:900]

    def _tokens(self, text: str) -> list[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
            "to", "was", "were", "what", "when", "where", "which", "who", "why",
            "with", "about",
        }
        return [
            token for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if token not in stopwords and len(token) > 2
        ]

    def _is_broad_question(self, question: str) -> bool:
        normalized = question.lower()
        broad_patterns = [
            "what is this about",
            "what's this about",
            "what is the document about",
            "what's the document about",
            "what is the project about",
            "what's the project about",
            "project about",
            "summarize",
            "summary",
            "overview",
            "main idea",
            "main topic",
        ]
        return any(pattern in normalized for pattern in broad_patterns)


class GeminiLLMProvider(LLMProvider):
    """Gemini provider skeleton with lazy dependency and credential handling."""

    def __init__(self, api_key: str | None, model_name: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._model = None

    def generate(self, prompt: GroundedPrompt) -> str:
        model = self._load_model()
        response = model.generate_content(
            [
                {"role": "user", "parts": [prompt.system_prompt]},
                {"role": "user", "parts": [prompt.user_prompt]},
            ]
        )
        return getattr(response, "text", "").strip()

    def _load_model(self):
        if self._model is not None:
            return self._model

        if not self.api_key:
            raise ProviderConfigurationError("GEMINI_API_KEY is required for GeminiLLMProvider.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderConfigurationError(
                "google-generativeai is required for GeminiLLMProvider. "
                "Install the Gemini dependency before using this provider."
            ) from exc

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)
        return self._model


class OpenAILLMProvider(LLMProvider):
    """OpenAI provider skeleton with lazy dependency and credential handling."""

    def __init__(self, api_key: str | None, model_name: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def generate(self, prompt: GroundedPrompt) -> str:
        client = self._load_client()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _load_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required for OpenAILLMProvider.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigurationError(
                "openai is required for OpenAILLMProvider. "
                "Install the OpenAI dependency before using this provider."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)
        return self._client


class RemoteHTTPLLMProvider(LLMProvider):
    """Calls a remote generation server, such as a Kaggle GPU notebook exposed by a tunnel."""

    def __init__(
        self,
        endpoint_url: str | None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: GroundedPrompt) -> str:
        response = self._post_json(
            {
                "system_prompt": prompt.system_prompt,
                "user_prompt": prompt.user_prompt,
                "prompt": f"{prompt.system_prompt}\n\n{prompt.user_prompt}",
            }
        )

        for key in ("answer", "text", "response", "generated_text"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        raise ProviderConfigurationError("Remote LLM server response must include answer text.")

    def _post_json(self, payload: dict) -> dict:
        if not self.endpoint_url:
            raise ProviderConfigurationError("REMOTE_LLM_URL is required when LLM_PROVIDER=remote.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            self.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ProviderConfigurationError(f"Remote LLM server returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderConfigurationError(f"Remote LLM server is unavailable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderConfigurationError("Remote LLM server returned invalid JSON.") from exc
