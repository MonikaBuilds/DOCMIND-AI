from fastapi import APIRouter

router = APIRouter(prefix="/summary", tags=["summary"])


@router.get("/status")
def summary_module_status() -> dict[str, str]:
    return {
        "module": "summary",
        "status": "planned",
        "next_step": "Summaries will be added after parsing and LLM integration.",
    }
