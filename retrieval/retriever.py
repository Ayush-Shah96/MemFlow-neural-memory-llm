from __future__ import annotations

import re
from collections import Counter

from config import SETTINGS
from graph.store import KnowledgeGraph
from storage.chunks import ChunkStore


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class GraphRetriever:
    """Neural-memory retrieval using query seeding + spreading activation."""

    def __init__(self, graph: KnowledgeGraph, chunks: ChunkStore, max_hops: int = 3, max_nodes: int = 100, top_k: int = 8):
        self.graph = graph
        self.chunks = chunks
        self.max_hops = max_hops
        self.max_nodes = max_nodes
        self.top_k = top_k

    def detect_query_entities(self, query: str) -> list[dict]:
        tokens = tokenize(query)
        qtext = " ".join(tokens)
        candidates = []
        for node, data in self.graph.graph.nodes(data=True):
            name = data.get("name", node)
            aliases = list(data.get("aliases", []))
            variants = [name] + aliases
            best = 0.0
            for variant in variants:
                nt = tokenize(variant)
                if not nt:
                    continue
                if len(nt) == 1:
                    score = 1.0 if nt[0] in tokens else 0.0
                else:
                    phrase = " ".join(nt)
                    score = 1.0 if phrase in qtext else 0.0
                best = max(best, score)
            if best:
                salience = float(data.get("salience", 1.0))
                candidates.append({
                    "node": node,
                    "name": name,
                    "type": data.get("type", "Concept"),
                    "score": round(best * min(1.5, 0.75 + 0.1 * salience), 4),
                })
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:12]

    def retrieve(self, query: str) -> dict:
        detected = self.detect_query_entities(query)
        seeds = [x["node"] for x in detected]

        # When an exact entity is not found, lexical overlap against stored concepts
        # provides a local-only fallback for query activation.
        if not seeds:
            qtokens = set(tokenize(query))
            for node, data in self.graph.graph.nodes(data=True):
                overlap = len(qtokens & set(tokenize(data.get("name", node))))
                if overlap:
                    seeds.append(node)
                    detected.append({
                        "node": node,
                        "name": data.get("name", node),
                        "type": data.get("type", "Concept"),
                        "score": min(1.0, 0.35 + 0.2 * overlap),
                    })
            detected.sort(key=lambda x: x["score"], reverse=True)
            detected = detected[:12]
            seeds = [x["node"] for x in detected]

        seed_scores = {x["node"]: max(0.2, float(x["score"])) for x in detected}
        activation, traversals = self.graph.spread_activation(
            seed_scores,
            max_hops=self.max_hops,
            decay=SETTINGS.activation_decay,
            threshold=SETTINGS.activation_threshold,
            bidirectional_factor=SETTINGS.activation_bidirectional_factor,
            max_nodes=self.max_nodes,
        )

        # Seed nodes should remain strongest, even if they have sparse connectivity.
        for node, score in seed_scores.items():
            activation[node] = max(activation.get(node, 0.0), score)

        chunk_scores: Counter[str] = Counter()
        activation_sources = {}
        for node, score in activation.items():
            activation_sources[node] = score
            data = self.graph.graph.nodes[node]
            node_bonus = score * (0.7 + 0.15 * float(data.get("salience", 1.0)))
            for cid in data.get("chunks", []):
                chunk_scores[cid] += node_bonus

        for item in traversals:
            for cid in (self.graph.graph.nodes[item["target"]].get("chunks", []) if item["target"] in self.graph.graph else []):
                chunk_scores[cid] += float(item["activation"]) * 0.5

        # Lexical backstop so a question can still retrieve a supporting chunk when
        # entity linking is weak, while neural activation remains the main signal.
        qtokens = set(tokenize(query))
        for chunk in self.chunks.all():
            overlap = len(qtokens & set(tokenize(chunk.get("text", ""))))
            if overlap:
                chunk_scores[chunk["chunk_id"]] += min(1.5, overlap * 0.15)

        ranked_chunks = []
        for cid, score in chunk_scores.most_common(self.top_k):
            chunk = self.chunks.get(cid)
            if chunk:
                ranked_chunks.append({**chunk, "score": round(float(score), 4)})

        edge_items = []
        for item in traversals[: self.max_nodes * 2]:
            edge_items.append({
                "source": item["source"],
                "source_name": self.graph.graph.nodes[item["source"]].get("name", item["source"]),
                "target": item["target"],
                "target_name": self.graph.graph.nodes[item["target"]].get("name", item["target"]),
                "relation": item["relation"],
                "depth": item["depth"],
                "confidence": 0.0,
                "weight": item["weight"],
                "activation": item["activation"],
                "direction": item["direction"],
            })

        # Learning: recalled paths reinforce the activated synapses and concepts.
        if traversals:
            self.graph.reinforce(traversals, learning_rate=SETTINGS.synapse_learning_rate)

        node_views = [
            {
                "node": n,
                "name": self.graph.graph.nodes[n].get("name", n),
                "type": self.graph.graph.nodes[n].get("type", "Concept"),
                "activation": round(score, 6),
                "salience": round(float(self.graph.graph.nodes[n].get("salience", 1.0)), 4),
            }
            for n, score in sorted(activation.items(), key=lambda x: x[1], reverse=True)[: self.max_nodes]
        ]

        _, explicit_paths = self.graph.connect_seed_nodes(seeds, self.max_hops, self.max_nodes)
        multi_hop_paths = explicit_paths + self._activation_paths(seeds, traversals)
        # Preserve order while removing duplicates.
        multi_hop_paths = list(dict.fromkeys(multi_hop_paths))[:20]
        return {
            "query": query,
            "detected_entities": detected,
            "nodes": node_views,
            "activations": node_views,
            "activation_trace": traversals[: self.max_nodes * 2],
            "paths": edge_items,
            "multi_hop_paths": multi_hop_paths,
            "chunks": ranked_chunks,
            "edge_items": edge_items,
        }

    def _activation_paths(self, seeds: list[str], traversals: list[dict]) -> list[str]:
        lines = []
        seed_set = set(seeds)
        for item in traversals:
            if item["source"] in seed_set or item["depth"] > 1:
                s = self.graph.graph.nodes[item["source"]].get("name", item["source"])
                t = self.graph.graph.nodes[item["target"]].get("name", item["target"])
                lines.append(f"{s} --{item['relation']}--> {t}")
        return lines[:20]
