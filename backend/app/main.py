from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import chat, documents, evaluation, search, summary, upload
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="RAG-based document intelligence backend.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(upload.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(chat.router, prefix=settings.api_prefix)
    app.include_router(search.router, prefix=settings.api_prefix)
    app.include_router(summary.router, prefix=settings.api_prefix)
    app.include_router(evaluation.router, prefix=settings.api_prefix)

    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    return app


app = create_app()
