from graph.store import KnowledgeGraph
from retrieval.retriever import GraphRetriever
from storage.chunks import ChunkStore


def _build(tmp_path):
    g = KnowledgeGraph(tmp_path / "g.json")
    c = ChunkStore(tmp_path / "c.json")
    for name in ["Alice", "Company X", "Company Y", "Product Z"]:
        g.add_entity(name, "Concept", "doc.txt", f"{name}::chunk")
    g.add_relationship("Alice", "founded", "Company X", "doc.txt", "c1")
    g.add_relationship("Company X", "acquired", "Company Y", "doc.txt", "c2")
    g.add_relationship("Company Y", "developed", "Product Z", "doc.txt", "c3")
    for cid, text in {
        "c1": "Alice founded Company X.",
        "c2": "Company X acquired Company Y.",
        "c3": "Company Y developed Product Z.",
    }.items():
        c.upsert({"chunk_id": cid, "document_id": "doc", "source": "doc.txt", "text": text, "index": 0})
    return g, c


def test_spreading_activation_recalls_associated_memories(tmp_path):
    g, c = _build(tmp_path)
    r = GraphRetriever(g, c, max_hops=3).retrieve("Alice")
    names = [x["name"] for x in r["activations"]]
    assert "Company X" in names
    assert "Company Y" in names
    assert any(x["relation"] == "founded" for x in r["activation_trace"])


def test_recall_strengthens_synapses(tmp_path):
    g, c = _build(tmp_path)
    before = next(data["weight"] for _, _, _, data in g.graph.edges(keys=True, data=True) if data["relation"] == "founded")
    GraphRetriever(g, c, max_hops=3).retrieve("Alice")
    after = next(data["weight"] for _, _, _, data in g.graph.edges(keys=True, data=True) if data["relation"] == "founded")
    assert after >= before
