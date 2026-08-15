from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    name: str
    type: str = "Concept"
    aliases: list[str] = field(default_factory=list)


@dataclass
class Relation:
    source: str
    relation: str
    target: str
    confidence: float = 0.8


def normalize_extraction(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Extraction payload must be an object")
    raw_entities = payload.get("entities", [])
    raw_relations = payload.get("relationships", payload.get("relations", []))
    entities: list[dict] = []
    relations: list[dict] = []
    seen_entities = set()
    for item in raw_entities if isinstance(raw_entities, list) else []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"]).strip()
        key = name.casefold()
        if key in seen_entities:
            continue
        seen_entities.add(key)
        entities.append({"name": name, "type": str(item.get("type", "Concept")), "aliases": item.get("aliases", []) or []})
    for item in raw_relations if isinstance(raw_relations, list) else []:
        if not isinstance(item, dict):
            continue
        if not all(item.get(k) for k in ("source", "relation", "target")):
            continue
        relations.append({
            "source": str(item["source"]).strip(),
            "relation": str(item["relation"]).strip().lower().replace(" ", "_"),
            "target": str(item["target"]).strip(),
            "confidence": float(item.get("confidence", 0.8)),
        })
    return {"entities": entities, "relationships": relations}
