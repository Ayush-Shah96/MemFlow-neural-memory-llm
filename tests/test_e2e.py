import pipeline as pipeline_module
from pipeline import MemoryPipeline


def test_basic_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline_module, "DOCUMENTS_DIR", tmp_path / "documents")
    monkeypatch.setattr(pipeline_module, "GRAPH_PATH", tmp_path / "memory" / "graph.json")
    monkeypatch.setattr(pipeline_module, "CHUNKS_PATH", tmp_path / "memory" / "chunks.json")
    (tmp_path / "documents").mkdir(); (tmp_path / "memory").mkdir()
    f = tmp_path / "example.txt"
    f.write_text("Alice founded Company X. Company X acquired Company Y. Company Y developed Product Z.", encoding="utf-8")
    p = MemoryPipeline()
    result = p.ingest_file(f)
    assert result["relationships"] >= 3
    answer = p.answer("Who founded Company X?")
    assert "Alice" in answer["answer"]
    multi = p.retriever.retrieve("Alice Product Z")
    assert multi["paths"]
    assert answer["sources"]
