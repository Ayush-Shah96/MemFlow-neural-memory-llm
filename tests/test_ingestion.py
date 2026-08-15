from ingestion.chunker import chunk_document
from ingestion.loader import load_document_text


def test_txt_and_chunking(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Alice founded Company X.\n\nCompany X acquired Company Y.", encoding="utf-8")
    text = load_document_text(p)
    chunks = chunk_document(text, "doc1", p.name, chunk_size=100, overlap=10)
    assert text.startswith("Alice founded")
    assert chunks
    assert chunks[0]["chunk_id"] == "doc1::chunk-1"
