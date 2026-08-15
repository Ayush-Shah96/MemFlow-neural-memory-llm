from generation.generator import AnswerGenerator


def test_offline_mock_does_not_expose_internal_graph_trace(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    retrieval = {
        "paths": ["Alice --founded--> Company X"],
        "chunks": [{"source": "doc.txt", "chunk_id": "c1", "text": "Alice founded Company X."}],
    }
    answer = AnswerGenerator()._mock_answer("Who founded Company X?", retrieval)
    assert "Graph evidence:" not in answer
    assert "Alice founded Company X." in answer
