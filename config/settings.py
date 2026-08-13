"""Central configuration for geo_graphrag."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "index"
GRAPHRAG_DIR = INDEX_DIR / "graphrag"
CHROMA_DIR = INDEX_DIR / "chroma"
PROMPTS_DIR = ROOT_DIR / "prompts"

DEFAULT_GRAPH_ARTIFACTS_DIR = GRAPHRAG_DIR / "output"
DEFAULT_TEXT_VECTOR_DIR = CHROMA_DIR

CORPUS1_DIR = DATA_DIR / "corpus1"
CORPUS2_DIR = DATA_DIR / "corpus2"
LEGACY_SHARED_CORPUS_DIR = DATA_DIR / "corpus"


def _env_path(name: str, default: Optional[Path]) -> Optional[Path]:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value)


@dataclass
class LLMConfig:
    base_url: str = field(default_factory=lambda: os.environ.get("GEO_LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("GEO_LLM_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: os.environ.get("GEO_LLM_MODEL", "qwen2.5:14b"))
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 600


@dataclass
class EmbeddingConfig:
    use_ollama: bool = True
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("GEO_OLLAMA_BASE_URL", "http://localhost:11434"))
    model_name: str = field(default_factory=lambda: os.environ.get("GEO_EMBEDDING_MODEL", "nomic-embed-text"))
    device: str = "cpu"
    batch_size: int = 64
    max_length: int = 512
    normalize: bool = True


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    top_p: int = 5


@dataclass
class ChunkConfig:
    chunk_size: int = 500
    overlap: int = 100
    bridge_prev: int = 300
    bridge_next: int = 300


@dataclass
class RetrievalConfig:
    initial_top_k: int = 10
    # Thresholds reported in the manuscript: decompose when max similarity
    # does not exceed 0.75 or topical coherence falls below 0.60.
    sim_threshold: float = 0.75
    coherence_threshold: float = 0.60
    exact_match_threshold: float = 0.85
    fuzzy_keep: int = 3
    oversample_scaler: int = 2
    community_top_n: int = 5
    community_alpha: float = 0.6
    community_beta: float = 0.4


@dataclass
class FusionConfig:
    graph_weight: float = 0.7
    text_weight: float = 0.3


@dataclass
class IndexPathsConfig:
    graph_artifacts_dir: Path = field(
        default_factory=lambda: _env_path("GEO_GRAPH_ARTIFACTS_DIR", DEFAULT_GRAPH_ARTIFACTS_DIR)
    )
    text_vector_dir: Path = field(
        default_factory=lambda: _env_path("GEO_TEXT_VECTOR_DIR", DEFAULT_TEXT_VECTOR_DIR)
    )
    corpus1_dir: Path = field(
        default_factory=lambda: _env_path("GEO_CORPUS1_DIR", CORPUS1_DIR)
    )
    corpus2_dir: Path = field(
        default_factory=lambda: _env_path("GEO_CORPUS2_DIR", CORPUS2_DIR)
    )


@dataclass
class GraphRAGConfig:
    workspace: Path = GRAPHRAG_DIR
    llm_model: str = field(default_factory=lambda: os.environ.get("GEO_GRAPHRAG_MODEL", "qwen2.5:32b"))
    llm_api_base: str = field(default_factory=lambda: os.environ.get("GEO_LLM_BASE_URL", "http://localhost:11434/v1"))
    embedding_model: str = field(default_factory=lambda: os.environ.get("GEO_EMBEDDING_MODEL", "nomic-embed-text"))
    community_algorithm: str = "leiden"
    external_output_dir: Path = field(
        default_factory=lambda: _env_path("GEO_GRAPH_ARTIFACTS_DIR", DEFAULT_GRAPH_ARTIFACTS_DIR)
    )


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    index_paths: IndexPathsConfig = field(default_factory=IndexPathsConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    log_level: str = "INFO"


cfg = AppConfig()
