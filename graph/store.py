from __future__ import annotations

import json
import math
import re
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph


def canonicalize(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip())
    return value.casefold()


class KnowledgeGraph:
    """Persistent associative memory graph.

    Nodes are memory neurons/concepts. Edges are weighted synapses. Node salience,
    recall counts, and synaptic weights persist between sessions.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = nx.MultiDiGraph()
        self.load()
        self._upgrade_loaded_graph()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.graph = json_graph.node_link_graph(
                payload, directed=True, multigraph=True, edges="links"
            )
        except Exception:
            self.graph = nx.MultiDiGraph()

    def _upgrade_loaded_graph(self) -> None:
        for _, data in self.graph.nodes(data=True):
            data.setdefault("salience", 1.0)
            data.setdefault("recall_count", 0)
            data.setdefault("last_recalled", None)
            data.setdefault("aliases", [])
            data.setdefault("sources", [])
            data.setdefault("chunks", [])
            data.setdefault("confidence", 0.5)
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            data.setdefault("weight", 1.0)
            data.setdefault("activation_count", 0)
            data.setdefault("last_activated", None)

    def save(self) -> None:
        payload = json_graph.node_link_data(self.graph, edges="links")
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def resolve_node(self, name: str) -> str | None:
        key = canonicalize(name)
        for node, data in self.graph.nodes(data=True):
            aliases = [canonicalize(x) for x in data.get("aliases", [])]
            if canonicalize(data.get("name", node)) == key or key in aliases or canonicalize(node) == key:
                return node
        return None

    def add_entity(
        self,
        name: str,
        entity_type: str,
        source: str,
        chunk_id: str,
        confidence: float = 0.8,
        aliases: list[str] | None = None,
    ) -> str:
        name = re.sub(r"\s+", " ", name.strip(" .,:;\n\t"))
        existing = self.resolve_node(name)
        node_id = existing or canonicalize(name)
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                name=name,
                type=entity_type,
                aliases=[],
                sources=[],
                chunks=[],
                confidence=confidence,
                salience=1.0,
                recall_count=0,
                last_recalled=None,
            )
        data = self.graph.nodes[node_id]
        data["name"] = data.get("name") or name
        data["type"] = entity_type if data.get("type") in (None, "Concept") else data.get("type")
        if aliases:
            data["aliases"] = sorted(set(data.get("aliases", []) + aliases))
        data["sources"] = sorted(set(data.get("sources", []) + [source]))
        data["chunks"] = sorted(set(data.get("chunks", []) + [chunk_id]))
        data["confidence"] = max(float(data.get("confidence", 0)), confidence)
        data.setdefault("salience", 1.0)
        data.setdefault("recall_count", 0)
        return node_id

    def add_relationship(
        self,
        source: str,
        relation: str,
        target: str,
        source_doc: str,
        chunk_id: str,
        confidence: float = 0.8,
        timestamp: str | None = None,
    ) -> None:
        s = self.resolve_node(source) or canonicalize(source)
        t = self.resolve_node(target) or canonicalize(target)
        if s not in self.graph:
            self.add_entity(source, "Concept", source_doc, chunk_id, confidence)
        if t not in self.graph:
            self.add_entity(target, "Concept", source_doc, chunk_id, confidence)
        relation = relation.strip().lower().replace(" ", "_")

        # Same semantic edge from another source updates confidence and provenance
        # instead of creating unbounded duplicate synapses.
        existing_edges = self.graph.get_edge_data(s, t, default={}) or {}
        for key, data in existing_edges.items():
            if data.get("relation") == relation:
                data["sources"] = sorted(set(data.get("sources", []) + [source_doc]))
                data["chunks"] = sorted(set(data.get("chunks", []) + [chunk_id]))
                data["confidence"] = max(float(data.get("confidence", 0)), confidence)
                return

        key = f"{relation}|{canonicalize(source_doc)}|{chunk_id}"
        self.graph.add_edge(
            s,
            t,
            key=key,
            relation=relation,
            sources=[source_doc],
            chunks=[chunk_id],
            confidence=confidence,
            timestamp=timestamp,
            weight=max(0.1, min(2.0, float(confidence))),
            activation_count=0,
            last_activated=None,
        )

    def spread_activation(
        self,
        seed_scores: dict[str, float],
        max_hops: int = 3,
        decay: float = 0.72,
        threshold: float = 0.08,
        bidirectional_factor: float = 0.90,
        max_nodes: int = 100,
    ) -> tuple[dict[str, float], list[dict]]:
        """Spread activation through weighted synapses in both semantic directions."""
        activation: dict[str, float] = {n: float(v) for n, v in seed_scores.items() if n in self.graph}
        frontier = [(n, float(v), 0) for n, v in activation.items()]
        traversals: list[dict] = []
        visited_best: dict[tuple[str, int], float] = {}

        while frontier and len(activation) <= max_nodes * 3:
            current, current_activation, depth = frontier.pop(0)
            if depth >= max_hops or current_activation < threshold:
                continue

            # Outgoing edges keep their direction; incoming edges create associative recall
            # with a configurable reduction, so semantic direction is preserved while memory
            # remains associative.
            neighbors: list[tuple[str, dict, float, str]] = []
            for _, nxt, _, data in self.graph.out_edges(current, keys=True, data=True):
                neighbors.append((nxt, data, 1.0, "forward"))
            for prv, _, _, data in self.graph.in_edges(current, keys=True, data=True):
                neighbors.append((prv, data, bidirectional_factor, "backward"))

            for nxt, data, direction_factor, direction in neighbors:
                weight = max(0.05, min(2.5, float(data.get("weight", 1.0))))
                edge_conf = max(0.1, min(1.0, float(data.get("confidence", 0.8))))
                propagated = current_activation * weight * edge_conf * decay * direction_factor
                if propagated < threshold:
                    continue
                key = (nxt, depth + 1)
                if propagated <= visited_best.get(key, 0.0):
                    continue
                visited_best[key] = propagated
                activation[nxt] = max(activation.get(nxt, 0.0), propagated)
                traversals.append(
                    {
                        "source": current,
                        "target": nxt,
                        "relation": data.get("relation", "related_to"),
                        "direction": direction,
                        "activation": round(propagated, 6),
                        "depth": depth + 1,
                        "weight": round(weight, 6),
                    }
                )
                frontier.append((nxt, propagated, depth + 1))

        ranked = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        return dict(ranked), traversals

    def reinforce(self, traversals: list[dict], learning_rate: float = 0.05) -> None:
        """Hebbian-style reinforcement: repeatedly recalled synapses become stronger."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        touched = set()
        for item in traversals:
            source, target = item["source"], item["target"]
            relation = item.get("relation")
            edge_data = self.graph.get_edge_data(source, target, default={}) or {}
            for _, data in edge_data.items():
                if data.get("relation") != relation:
                    continue
                weight = float(data.get("weight", 1.0))
                boost = learning_rate * min(1.0, float(item.get("activation", 0.0)))
                data["weight"] = min(2.5, weight + boost)
                data["activation_count"] = int(data.get("activation_count", 0)) + 1
                data["last_activated"] = now
                touched.add((source, target, relation))
        for node in {x for item in traversals for x in (item["source"], item["target"])}:
            if node in self.graph:
                d = self.graph.nodes[node]
                d["recall_count"] = int(d.get("recall_count", 0)) + 1
                d["salience"] = min(5.0, float(d.get("salience", 1.0)) + learning_rate)
                d["last_recalled"] = now
        if touched:
            self.save()

    def ego_paths(self, seed_nodes: list[str], max_hops: int = 2, max_nodes: int = 100) -> tuple[set[str], list[dict]]:
        seeds = [n for n in seed_nodes if n in self.graph]
        seen = set(seeds)
        queue = [(n, 0) for n in seeds]
        paths: list[dict] = []
        while queue and len(seen) <= max_nodes:
            current, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            neighbors = list(self.graph.successors(current)) + list(self.graph.predecessors(current))
            for nxt in neighbors:
                if nxt not in seen and len(seen) < max_nodes:
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
                forward = self.graph.get_edge_data(current, nxt, default={}) or {}
                for data in forward.values():
                    paths.append(self._path_item(current, nxt, depth + 1, data))
                backward = self.graph.get_edge_data(nxt, current, default={}) or {}
                for data in backward.values():
                    paths.append(self._path_item(nxt, current, depth + 1, data))
        return seen, self._dedupe_paths(paths)

    def connect_seed_nodes(self, seed_nodes: list[str], max_hops: int = 2, max_nodes: int = 100) -> tuple[set[str], list[str]]:
        raw_paths: list[list[str]] = []
        for i, source in enumerate(seed_nodes):
            for target in seed_nodes[i + 1 :]:
                try:
                    path = nx.shortest_path(self.graph.to_undirected(), source, target)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if 1 <= len(path) - 1 <= max_hops:
                    raw_paths.append(path)
        nodes = set().union(*(set(p) for p in raw_paths)) if raw_paths else set(seed_nodes)
        formatted = [" -> ".join(self.graph.nodes[n].get("name", n) for n in path) for path in raw_paths]
        return nodes, formatted

    def _path_item(self, src: str, dst: str, depth: int, data: dict) -> dict:
        return {
            "source": src,
            "source_name": self.graph.nodes[src].get("name", src),
            "target": dst,
            "target_name": self.graph.nodes[dst].get("name", dst),
            "relation": data.get("relation", "related_to"),
            "depth": depth,
            "sources": data.get("sources", []),
            "chunks": data.get("chunks", []),
            "confidence": data.get("confidence", 0.0),
            "weight": data.get("weight", 1.0),
        }

    @staticmethod
    def _dedupe_paths(items: list[dict]) -> list[dict]:
        seen = set(); out = []
        for x in items:
            key = (x["source"], x["relation"], x["target"], tuple(x["chunks"]))
            if key not in seen:
                seen.add(key); out.append(x)
        return out

    def stats(self) -> dict:
        return {
            "entities": self.graph.number_of_nodes(),
            "relationships": self.graph.number_of_edges(),
            "connected_components": nx.number_connected_components(self.graph.to_undirected()) if self.graph.number_of_nodes() else 0,
            "learned_synapses": sum(1 for _, _, _, d in self.graph.edges(keys=True, data=True) if int(d.get("activation_count", 0)) > 0),
        }
