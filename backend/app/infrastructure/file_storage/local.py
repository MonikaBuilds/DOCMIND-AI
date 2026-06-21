from pathlib import Path


class LocalFileStorage:
    """Stores uploaded files on the local filesystem for the learning-first version."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)

    def save_bytes(self, filename: str, content: bytes) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.base_dir / filename
        target_path.write_bytes(content)
        return target_path
