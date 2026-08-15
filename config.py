from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    chunk_size: int = _int("CHUNK_SIZE", 1200)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)
    max_hops: int = _int("MAX_HOPS", 3)
    top_k_memories: int = _int("TOP_K_MEMORIES", 8)
    max_graph_nodes: int = _int("MAX_GRAPH_NODES", 100)
    max_graph_edges: int = _int("MAX_GRAPH_EDGES", 250)

    # Tests force mock mode through tests/conftest.py.
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3:8b")

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
    openrouter_http_referer: str = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:7860").strip()
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "Neural Memory LLM").strip()

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    activation_decay: float = _float("ACTIVATION_DECAY", 0.72)
    activation_threshold: float = _float("ACTIVATION_THRESHOLD", 0.08)
    activation_bidirectional_factor: float = _float("ACTIVATION_BIDIRECTIONAL_FACTOR", 0.90)
    synapse_learning_rate: float = _float("SYNAPSE_LEARNING_RATE", 0.05)


SETTINGS = Settings()
MEMORY_DIR = SETTINGS.data_dir / "memory"
DOCUMENTS_DIR = SETTINGS.data_dir / "documents"
GRAPH_PATH = MEMORY_DIR / "graph.json"
CHUNKS_PATH = MEMORY_DIR / "chunks.json"

for p in (SETTINGS.data_dir, MEMORY_DIR, DOCUMENTS_DIR):
    p.mkdir(parents=True, exist_ok=True)