from graph.store import KnowledgeGraph
from retrieval.retriever import GraphRetriever
from storage.chunks import ChunkStore


def test_two_hop_path(tmp_path):
    g = KnowledgeGraph(tmp_path / "g.json")
    c = ChunkStore(tmp_path / "c.json")
    for name in ["Alice", "Company X", "Company Y", "Product Z"]:
        g.add_entity(name, "Concept", "doc.txt", f"{name}::chunk")
    g.add_relationship("Alice", "founded", "Company X", "doc.txt", "c1")
    g.add_relationship("Company X", "acquired", "Company Y", "doc.txt", "c2")
    g.add_relationship("Company Y", "developed", "Product Z", "doc.txt", "c3")
    for cid,text in {"c1":"Alice founded Company X.","c2":"Company X acquired Company Y.","c3":"Company Y developed Product Z."}.items():
        c.upsert({"chunk_id":cid,"document_id":"doc","source":"doc.txt","text":text,"index":0})
    r = GraphRetriever(g,c,max_hops=3).retrieve("Alice Product Z")
    assert any("Alice -> Company X -> Company Y -> Product Z" in p for p in r["multi_hop_paths"])
