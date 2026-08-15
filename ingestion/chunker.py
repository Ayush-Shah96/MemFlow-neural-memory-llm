from __future__ import annotations

import re


def chunk_document(text: str, document_id: str, source: str, chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    # Prefer paragraph/sentence boundaries while keeping a predictable max size.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + chunk_size)
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    current = ""
                    break
                start = max(0, end - overlap)
    if current:
        chunks.append(current)

    result = []
    for idx, content in enumerate(chunks):
        result.append(
            {
                "chunk_id": f"{document_id}::chunk-{idx+1}",
                "document_id": document_id,
                "source": source,
                "text": content,
                "index": idx,
            }
        )
    return result
