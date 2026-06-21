import pytest

from app.core.exceptions import ProviderConfigurationError
from app.infrastructure.vector_store.base import ChromaVectorStore


def test_chroma_vector_store_reports_missing_dependency_when_unavailable():
    store = ChromaVectorStore(persist_directory="unused")

    try:
        import chromadb  # noqa: F401
    except ImportError:
        with pytest.raises(ProviderConfigurationError):
            store.search([0.1, 0.2], top_k=1)
    else:
        pytest.skip("chromadb is installed; dependency error test is not applicable.")
