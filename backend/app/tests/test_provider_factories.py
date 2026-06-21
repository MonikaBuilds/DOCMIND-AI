from app.core.config import Settings
from app.core.providers import build_vector_store


def test_memory_vector_store_is_shared_for_backend_session():
    settings = Settings(VECTOR_STORE_PROVIDER="memory")

    first = build_vector_store(settings)
    second = build_vector_store(settings)

    assert first is second
