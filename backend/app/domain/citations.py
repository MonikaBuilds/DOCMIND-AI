from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    document_id: str
    filename: str
    page_number: int
    source: str
    label: str
    heading: str | None = None
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)
