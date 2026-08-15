from __future__ import annotations

import json
from pathlib import Path


class ChunkStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.items = {}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, chunk: dict) -> None:
        self.items[chunk["chunk_id"]] = chunk
        self.save()

    def get(self, chunk_id: str) -> dict | None:
        return self.items.get(chunk_id)

    def all(self) -> list[dict]:
        return list(self.items.values())

    def source_chunks(self, source: str) -> list[dict]:
        return [x for x in self.items.values() if x.get("source") == source]
