from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="DocMind AI", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    storage_dir: str = Field(default="storage", alias="STORAGE_DIR")
    parsed_docs_dir: str = Field(default="parsed_docs", alias="PARSED_DOCS_DIR")
    ocr_output_dir: str = Field(default="ocr_outputs", alias="OCR_OUTPUT_DIR")
    chroma_db_dir: str = Field(default="chroma_db", alias="CHROMA_DB_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    embedding_provider: str = Field(default="bge", alias="EMBEDDING_PROVIDER")
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL_NAME",
    )
    openai_embedding_model_name: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL_NAME",
    )
    gemini_embedding_model_name: str = Field(
        default="models/gemini-embedding-001",
        alias="GEMINI_EMBEDDING_MODEL_NAME",
    )
    remote_embedding_url: str | None = Field(default=None, alias="REMOTE_EMBEDDING_URL")
    remote_llm_url: str | None = Field(default=None, alias="REMOTE_LLM_URL")
    remote_model_api_key: str | None = Field(default=None, alias="REMOTE_MODEL_API_KEY")
    remote_request_timeout_seconds: float = Field(default=60.0, alias="REMOTE_REQUEST_TIMEOUT_SECONDS")
    vector_store_provider: str = Field(default="chroma", alias="VECTOR_STORE_PROVIDER")
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_model_name: str = Field(default="models/gemini-flash-latest", alias="GEMINI_MODEL_NAME")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openai_model_name: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL_NAME")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    ocr_provider: str = Field(default="easyocr", alias="OCR_PROVIDER")

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
