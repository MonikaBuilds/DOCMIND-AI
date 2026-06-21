from abc import ABC, abstractmethod
import hashlib
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.exceptions import ProviderConfigurationError


class EmbeddingProvider(ABC):
    """Interface used by retrieval services without knowing the model provider."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free lexical embedding fallback for local demos and quota-free runs."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class BGEEmbeddingProvider(EmbeddingProvider):
    """Default implementation using BAAI/bge-small-en-v1.5 through SentenceTransformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_model()
        normalized_texts = [self._prefix_passage(text) for text in texts]
        embeddings = model.encode(normalized_texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(self._prefix_query(query), normalize_embeddings=True)
        return embedding.tolist()

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderConfigurationError(
                "sentence-transformers is required for BGE embeddings. "
                "Install backend requirements before using BGEEmbeddingProvider."
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        return self._model

    def _prefix_passage(self, text: str) -> str:
        return f"passage: {text}"

    def _prefix_query(self, query: str) -> str:
        return f"query: {query}"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with lazy SDK loading."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._load_client()
        response = client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []

    def _load_client(self):
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderConfigurationError(
                "openai is required for OpenAIEmbeddingProvider. "
                "Install the OpenAI dependency before using this provider."
            ) from exc

        self._client = OpenAI(api_key=self.api_key)
        return self._client


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding provider with lazy SDK loading."""

    def __init__(
        self,
        api_key: str | None,
        model_name: str = "models/embedding-001",
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self._genai = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        genai = self._load_client()
        embeddings: list[list[float]] = []
        for text in texts:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(list(response["embedding"]))
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        genai = self._load_client()
        response = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query",
        )
        return list(response["embedding"])

    def _load_client(self):
        if self._genai is not None:
            return self._genai

        if not self.api_key:
            raise ProviderConfigurationError("GEMINI_API_KEY is required for GeminiEmbeddingProvider.")

        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderConfigurationError(
                "google-generativeai is required for GeminiEmbeddingProvider. "
                "Install the Gemini dependency before using this provider."
            ) from exc

        genai.configure(api_key=self.api_key)
        self._genai = genai
        return self._genai


class RemoteHTTPEmbeddingProvider(EmbeddingProvider):
    """Calls a remote embedding server, such as a Kaggle GPU notebook exposed by a tunnel."""

    def __init__(
        self,
        endpoint_url: str | None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._post_json({"texts": texts, "task": "document"})
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list):
            raise ProviderConfigurationError("Remote embedding server response must include an embeddings list.")
        return [[float(value) for value in embedding] for embedding in embeddings]

    def embed_query(self, query: str) -> list[float]:
        response = self._post_json({"texts": [query], "task": "query"})
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return [float(value) for value in embeddings[0]]

        embedding = response.get("embedding")
        if isinstance(embedding, list):
            return [float(value) for value in embedding]

        raise ProviderConfigurationError("Remote embedding server response must include embedding data.")

    def _post_json(self, payload: dict) -> dict:
        if not self.endpoint_url:
            raise ProviderConfigurationError("REMOTE_EMBEDDING_URL is required when EMBEDDING_PROVIDER=remote.")

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
            raise ProviderConfigurationError(f"Remote embedding server returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise ProviderConfigurationError(f"Remote embedding server is unavailable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProviderConfigurationError("Remote embedding server returned invalid JSON.") from exc
