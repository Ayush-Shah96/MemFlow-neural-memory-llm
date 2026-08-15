EXTRACTION_SYSTEM = """You extract entities and relationships from a document chunk.
Return ONLY valid JSON with this exact top-level shape:
{
  "entities": [{"name": "...", "type": "Person|Organization|Product|Location|Technology|Concept", "aliases": []}],
  "relationships": [{"source": "...", "relation": "short_snake_case", "target": "...", "confidence": 0.0}]
}
Rules:
- Use canonical names from the text.
- Capture multi-word people and organizations as complete entities.
- Only create a relationship when the text supports it.
- Prefer relations such as founded, acquired, developed, works_at, joined, located_in, leads, partnered_with, ordered, deployed_in, created, used_by.
- Relationship source/target strings MUST exactly match an entity name.
- Do not invent entities or facts.
"""
