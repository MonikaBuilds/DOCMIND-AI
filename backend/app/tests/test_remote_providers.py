import json

from app.domain.chat import GroundedPrompt
from app.infrastructure.embeddings.base import RemoteHTTPEmbeddingProvider
from app.infrastructure.llm.base import RemoteHTTPLLMProvider


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_remote_embedding_provider_posts_texts(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers.get("Authorization")
        return FakeHTTPResponse({"embeddings": [[1, 2], [3, 4]]})

    monkeypatch.setattr("app.infrastructure.embeddings.base.urlopen", fake_urlopen)
    provider = RemoteHTTPEmbeddingProvider(
        endpoint_url="https://demo.trycloudflare.com/embed",
        api_key="secret",
        timeout_seconds=12,
    )

    embeddings = provider.embed_texts(["first", "second"])

    assert embeddings == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["url"] == "https://demo.trycloudflare.com/embed"
    assert captured["timeout"] == 12
    assert captured["payload"] == {"texts": ["first", "second"], "task": "document"}
    assert captured["authorization"] == "Bearer secret"


def test_remote_embedding_provider_embeds_query(monkeypatch):
    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        assert payload == {"texts": ["What is this?"], "task": "query"}
        return FakeHTTPResponse({"embeddings": [[0.5, 0.25]]})

    monkeypatch.setattr("app.infrastructure.embeddings.base.urlopen", fake_urlopen)
    provider = RemoteHTTPEmbeddingProvider(endpoint_url="https://demo.trycloudflare.com/embed")

    assert provider.embed_query("What is this?") == [0.5, 0.25]


def test_remote_llm_provider_posts_grounded_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeHTTPResponse({"answer": "Remote answer with citations."})

    monkeypatch.setattr("app.infrastructure.llm.base.urlopen", fake_urlopen)
    provider = RemoteHTTPLLMProvider(endpoint_url="https://demo.trycloudflare.com/generate")
    prompt = GroundedPrompt(system_prompt="Use sources only.", user_prompt="Question: Explain.")

    answer = provider.generate(prompt)

    assert answer == "Remote answer with citations."
    assert captured["payload"]["system_prompt"] == "Use sources only."
    assert captured["payload"]["user_prompt"] == "Question: Explain."
    assert "Use sources only." in captured["payload"]["prompt"]
