from graph.store import KnowledgeGraph


def test_graph_resolution_and_persistence(tmp_path):
    path = tmp_path / "graph.json"
    g = KnowledgeGraph(path)
    a = g.add_entity("Company X", "Organization", "a.txt", "a::1")
    b = g.add_entity("Company Y", "Organization", "a.txt", "a::1")
    g.add_relationship("Company X", "acquired", "Company Y", "a.txt", "a::1")
    g.save()
    g2 = KnowledgeGraph(path)
    assert g2.resolve_node("company x") == a
    assert g2.graph.has_edge(a, b)
