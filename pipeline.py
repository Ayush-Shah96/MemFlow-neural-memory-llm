from __future__ import annotations

import hashlib
import logging
import shutil
import time
from pathlib import Path

from config import SETTINGS, DOCUMENTS_DIR, GRAPH_PATH, CHUNKS_PATH
from ingestion.loader import load_document_text
from ingestion.chunker import chunk_document
from extraction.extractor import ExtractionService
from graph.store import KnowledgeGraph
from storage.chunks import ChunkStore
from retrieval.retriever import GraphRetriever
from generation.generator import AnswerGenerator


logger = logging.getLogger("neural-memory.pipeline")


class MemoryPipeline:
    def __init__(self):
        self.extractor = ExtractionService()
        self.graph = KnowledgeGraph(GRAPH_PATH)
        self.chunks = ChunkStore(CHUNKS_PATH)
        self.generator = AnswerGenerator()
        self.retriever = GraphRetriever(
            self.graph,
            self.chunks,
            SETTINGS.max_hops,
            SETTINGS.max_graph_nodes,
            SETTINGS.top_k_memories,
        )

        logger.info(
            "Pipeline initialized | provider=%s | model=%s",
            SETTINGS.llm_provider,
            self._provider_model(),
        )

    @staticmethod
    def _provider_model() -> str:
        if SETTINGS.llm_provider == "openrouter":
            return SETTINGS.openrouter_model
        if SETTINGS.llm_provider == "ollama":
            return SETTINGS.ollama_model
        if SETTINGS.llm_provider == "openai":
            return SETTINGS.openai_model
        return "mock"

    def ingest_file(self, path: str | Path) -> dict:
        path = Path(path)
        started = time.perf_counter()

        logger.info("=" * 72)
        logger.info("INGEST START | file=%s", path.name)
        logger.info("File path: %s", path.resolve())

        try:
            load_started = time.perf_counter()
            logger.info("[1/6] Loading document | file=%s", path.name)
            text = load_document_text(path)
            load_time = time.perf_counter() - load_started
            logger.info(
                "[1/6] Document loaded | file=%s | chars=%d | %.2fs",
                path.name,
                len(text),
                load_time,
            )

            if not text.strip():
                logger.error(
                    "Document contains no extractable text | file=%s",
                    path.name,
                )
                raise ValueError("Document contains no extractable text")

            document_id = hashlib.sha1(
                f"{path.resolve()}:{text}".encode("utf-8")
            ).hexdigest()[:16]

            dest = DOCUMENTS_DIR / f"{document_id}{path.suffix.lower()}"
            if not dest.exists():
                shutil.copy2(path, dest)
                logger.info(
                    "Document archived | file=%s | destination=%s",
                    path.name,
                    dest,
                )

            chunk_started = time.perf_counter()
            logger.info(
                "[2/6] Chunking document | file=%s | chunk_size=%d | overlap=%d",
                path.name,
                SETTINGS.chunk_size,
                SETTINGS.chunk_overlap,
            )
            chunks = chunk_document(
                text,
                document_id,
                path.name,
                SETTINGS.chunk_size,
                SETTINGS.chunk_overlap,
            )
            chunk_time = time.perf_counter() - chunk_started
            logger.info(
                "[2/6] Chunking complete | file=%s | chunks=%d | %.2fs",
                path.name,
                len(chunks),
                chunk_time,
            )

            entity_count: set[str] = set()
            relation_count = 0

            for index, chunk in enumerate(chunks, start=1):
                logger.info(
                    "[3/6] Extracting chunk %d/%d | file=%s | chunk_id=%s | chars=%d",
                    index,
                    len(chunks),
                    path.name,
                    chunk.get("chunk_id", "unknown"),
                    len(chunk.get("text", "")),
                )

                extract_started = time.perf_counter()
                extraction = self.extractor.extract(chunk["text"])
                extract_time = time.perf_counter() - extract_started

                entities = extraction.get("entities", [])
                relationships = extraction.get("relationships", [])

                logger.info(
                    "[3/6] Extraction complete | file=%s | chunk=%d/%d | entities=%d | relationships=%d | confidence=%.3f | %.2fs",
                    path.name,
                    index,
                    len(chunks),
                    len(entities),
                    len(relationships),
                    float(extraction.get("confidence", 0.0)),
                    extract_time,
                )

                logger.info(
                    "[4/6] Updating memory graph | file=%s | chunk=%d/%d",
                    path.name,
                    index,
                    len(chunks),
                )

                graph_started = time.perf_counter()
                for ent in entities:
                    node = self.graph.add_entity(
                        ent["name"],
                        ent.get("type", "Concept"),
                        path.name,
                        chunk["chunk_id"],
                        extraction["confidence"],
                        ent.get("aliases"),
                    )
                    entity_count.add(node)

                for rel in relationships:
                    self.graph.add_relationship(
                        rel["source"],
                        rel["relation"],
                        rel["target"],
                        path.name,
                        chunk["chunk_id"],
                        rel.get(
                            "confidence",
                            extraction["confidence"],
                        ),
                        extraction["timestamp"],
                    )
                    relation_count += 1

                chunk["entities"] = entities
                chunk["relationships"] = relationships
                self.chunks.upsert(chunk)

                graph_time = time.perf_counter() - graph_started
                graph_stats = self.graph.stats()
                logger.info(
                    "[4/6] Memory graph updated | nodes=%d | edges=%d | %.2fs",
                    graph_stats.get("entities", 0),
                    graph_stats.get("relationships", 0),
                    graph_time,
                )

            logger.info(
                "[5/6] Persisting memory | file=%s",
                path.name,
            )
            save_started = time.perf_counter()
            self.graph.save()
            save_time = time.perf_counter() - save_started
            logger.info(
                "[5/6] Memory persisted | graph=%s | chunks=%s | %.2fs",
                GRAPH_PATH,
                CHUNKS_PATH,
                save_time,
            )

            elapsed = time.perf_counter() - started
            result = {
                "source": path.name,
                "document_id": document_id,
                "chunks": len(chunks),
                "entities": len(entity_count),
                "relationships": relation_count,
                "elapsed_seconds": elapsed,
            }

            logger.info(
                "[6/6] INGEST COMPLETE | file=%s | chunks=%d | neurons=%d | synapses=%d | %.2fs",
                path.name,
                len(chunks),
                len(entity_count),
                relation_count,
                elapsed,
            )
            logger.info("=" * 72)
            return result

        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception(
                "INGEST FAILED | file=%s | after %.2fs",
                path.name,
                elapsed,
            )
            logger.info("=" * 72)
            raise

    def answer(self, query: str) -> dict:
        logger.info("QUESTION | %s", query)
        retrieval = self.retriever.retrieve(query)
        logger.info(
            "RETRIEVAL COMPLETE | detected_entities=%d | activated=%d | chunks=%d",
            len(retrieval.get("detected_entities", [])),
            len(retrieval.get("activations", [])),
            len(retrieval.get("chunks", [])),
        )
        answer_started = time.perf_counter()
        answer = self.generator.answer(query, retrieval)
        logger.info(
            "ANSWER GENERATED | provider=%s | %.2fs",
            SETTINGS.llm_provider,
            time.perf_counter() - answer_started,
        )
        source_list = sorted(
            {c["source"] for c in retrieval.get("chunks", [])}
        )
        return {
            "answer": answer,
            "sources": source_list,
            **retrieval,
        }

    def stats(self) -> dict:
        return {
            "documents": len({
                c.get("source") for c in self.chunks.all()
            }),
            "chunks": len(self.chunks.all()),
            **self.graph.stats(),
        }