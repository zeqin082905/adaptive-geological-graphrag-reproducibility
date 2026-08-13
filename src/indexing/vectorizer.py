"""Embedding and configurable Chroma vector-store helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import cfg
from src.indexing.chunker import TextChunk

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, config=None):
        self._cfg = config or cfg.embedding
        self._st_model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._cfg.use_ollama:
            return self._encode_ollama(texts)
        return self._encode_st(texts)

    def encode_single(self, text: str) -> list[float]:
        return self.encode([text])[0]

    def _encode_ollama(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        batch_size = self._cfg.batch_size
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            if total > batch_size:
                logger.info("Embedding progress: %s/%s", min(i + batch_size, total), total)
            results.extend(self._embed_batch_with_retry(batch))
        return results

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        import requests

        try:
            resp = requests.post(
                f"{self._cfg.ollama_base_url}/api/embed",
                json={"model": self._cfg.model_name, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("embeddings", [])
        except Exception as exc:
            if len(batch) <= 1:
                logger.error("Embedding request failed for single-item batch: %s", exc)
                raise
            mid = max(1, len(batch) // 2)
            logger.warning(
                "Embedding batch of %s failed, retrying in smaller chunks: %s",
                len(batch),
                exc,
            )
            return self._embed_batch_with_retry(batch[:mid]) + self._embed_batch_with_retry(batch[mid:])

    def _encode_st(self, texts: list[str]) -> list[list[float]]:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._cfg.model_name)
            self._st_model = SentenceTransformer(
                self._cfg.model_name,
                device=self._cfg.device,
            )
        vectors = self._st_model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=self._cfg.normalize,
            show_progress_bar=len(texts) > 50,
        )
        return vectors.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.encode(input)


embedder = EmbeddingModel()


class _VectorStoreBackend:
    TEXT_COLLECTION = "text_chunks"
    BRIDGE_COLLECTION = "bridge_chunks"

    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._text_col = self._get_or_create(self.TEXT_COLLECTION)
        self._bridge_col = self._get_or_create(self.BRIDGE_COLLECTION)

    def add_chunks(self, chunks: list[TextChunk], batch_size: int = 100):
        regular = [c for c in chunks if not c.is_bridge]
        bridges = [c for c in chunks if c.is_bridge]
        if regular:
            self._upsert_to_collection(self._text_col, regular, batch_size)
            logger.info("Wrote regular text chunks: %s", len(regular))
        if bridges:
            self._upsert_to_collection(self._bridge_col, bridges, batch_size)
            logger.info("Wrote bridge chunks: %s", len(bridges))

    def replace_chunks(self, chunks: list[TextChunk], batch_size: int = 100):
        self.clear()
        self.add_chunks(chunks, batch_size=batch_size)

    def clear(self):
        for name in (self.TEXT_COLLECTION, self.BRIDGE_COLLECTION):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._text_col = self._get_or_create(self.TEXT_COLLECTION)
        self._bridge_col = self._get_or_create(self.BRIDGE_COLLECTION)

    def search(self, query: str, top_k: int = 10, include_bridge: bool = True) -> list[dict]:
        query_vec = embedder.encode_single(query)
        results: list[dict] = []
        collections = [self._text_col, self._bridge_col] if include_bridge else [self._text_col]
        for col in collections:
            count = col.count()
            if count == 0:
                continue
            res = col.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            for idx, (doc, meta, dist) in enumerate(
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
            ):
                similarity = 1.0 - dist
                results.append(
                    {
                        "chunk_id": res["ids"][0][idx],
                        "text": doc,
                        "similarity": round(similarity, 4),
                        "doc_id": meta.get("doc_id"),
                        "start_page": meta.get("start_page"),
                        "end_page": meta.get("end_page"),
                        "is_bridge": meta.get("is_bridge") == "True",
                        "chunk_index": meta.get("chunk_index"),
                        "source_label": meta.get("source_label", "unknown"),
                    }
                )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        seen = set()
        deduped = []
        for item in results:
            key = (item["doc_id"], item["text"][:100])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:top_k]

    def count(self) -> dict:
        return {
            "text_chunks": self._text_col.count(),
            "bridge_chunks": self._bridge_col.count(),
        }

    def _upsert_to_collection(self, collection: chromadb.Collection, chunks: list[TextChunk], batch_size: int):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "start_page": c.start_page,
                        "end_page": c.end_page,
                        "chunk_index": c.chunk_index,
                        "is_bridge": str(c.is_bridge),
                        "triggered_rules": ",".join(c.triggered_rules),
                        "source_label": getattr(c, "source_label", "base"),
                    }
                    for c in batch
                ],
                embeddings=embedder.encode([c.text for c in batch]),
            )

    def _get_or_create(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )


class ConfigurableVectorStore:
    def __init__(self):
        self._persist_dir = Path(cfg.index_paths.text_vector_dir)
        self._backend = _VectorStoreBackend(self._persist_dir)

    @property
    def persist_dir(self) -> Path:
        return self._persist_dir

    def configure(self, persist_dir: Optional[Path] = None, reset: bool = False):
        target = Path(persist_dir or cfg.index_paths.text_vector_dir)
        if reset or target != self._persist_dir:
            logger.info("Switch text vector store to: %s", target)
            self._persist_dir = target
            self._backend = _VectorStoreBackend(target)
        return self

    def add_chunks(self, chunks: list[TextChunk], batch_size: int = 100):
        return self._backend.add_chunks(chunks, batch_size=batch_size)

    def replace_chunks(self, chunks: list[TextChunk], batch_size: int = 100):
        return self._backend.replace_chunks(chunks, batch_size=batch_size)

    def clear(self):
        return self._backend.clear()

    def search(self, query: str, top_k: int = 10, include_bridge: bool = True) -> list[dict]:
        return self._backend.search(query, top_k=top_k, include_bridge=include_bridge)

    def count(self) -> dict:
        return self._backend.count()


vector_store = ConfigurableVectorStore()
