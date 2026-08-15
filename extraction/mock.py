from __future__ import annotations

import re

ORG_HINTS = ("Inc", "Corp", "Company", "Technologies", "Logistics", "Robotics", "Labs", "Systems")
PRODUCT_HINTS = ("Atlas", "Product", "Model", "Platform")
LOCATION_WORDS = {"Pune", "Mumbai", "Bengaluru", "Bangalore", "Delhi", "London", "Paris", "Tokyo"}


def _entity_type(name: str) -> str:
    if name in LOCATION_WORDS:
        return "Location"
    if any(h in name for h in ORG_HINTS):
        return "Organization"
    if any(h in name for h in PRODUCT_HINTS):
        return "Product"
    if re.fullmatch(r"[A-Z][a-z]+ [A-Z][a-z]+", name):
        return "Person"
    if name in {"Python", "SQL", "Linux", "AI", "Machine Learning"}:
        return "Technology"
    return "Concept"


def extract_heuristically(text: str) -> dict:
    entities: dict[str, dict] = {}
    def add(name: str, typ: str | None = None):
        name = re.sub(r"\s+", " ", name.strip(" .,:;\n\t"))
        if not name:
            return
        entities.setdefault(name.casefold(), {"name": name, "type": typ or _entity_type(name), "aliases": []})

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    for sentence in sentences:
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,3}\b", sentence):
            candidate = m.group(0).strip(" ?!.,")
            if len(candidate) > 2 and candidate.lower() not in {"the", "this", "who", "what", "when", "where", "how"}:
                add(candidate)

    relations = []
    patterns = [
        (r"(?P<s>[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)?)\s+founded\s+(?P<t>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})", "founded"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})\s+was\s+founded(?:\s+in\s+\d{4})?\s+by\s+(?P<t>[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)?)", "founded_by"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})\s+acquired\s+(?P<t>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})", "acquired"),
        (r"(?P<s>[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)?)\s+joined\s+(?P<t>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})", "joined"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})\s+developed\s+(?P<t>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})", "developed"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})\s+works\s+at\s+(?P<t>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})", "works_at"),
        (r"(?P<s>[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,2})\s+is\s+located\s+in\s+(?P<t>[A-Z][A-Za-z]+)", "located_in"),
    ]
    for sentence in sentences:
        for pattern, relation in patterns:
            for m in re.finditer(pattern, sentence):
                src = m.group("s").strip(" .,:;")
                tgt = m.group("t").strip(" .,:;")
                add(src); add(tgt)
                relations.append({"source": src, "relation": relation, "target": tgt, "confidence": 0.75})
    return {"entities": list(entities.values()), "relationships": relations}
