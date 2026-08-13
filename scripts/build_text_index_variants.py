"""Build isolated text vector-store variants from GraphRAG text_units and optional extra chunks."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config.settings import cfg
from src.indexing.chunker import TextChunk
from src.indexing.vectorizer import vector_store


def load_base_chunks(graphrag_dir: Path) -> list[TextChunk]:
    for name in ("text_units.parquet", "create_final_text_units.parquet"):
        path = graphrag_dir / name
        if path.exists():
            df = pd.read_parquet(path)
            chunks = []
            for idx, row in df.iterrows():
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                doc_ids = row.get("document_ids", None)
                doc_id = "unknown"
                if isinstance(doc_ids, list) and doc_ids:
                    doc_id = str(doc_ids[0])
                elif doc_ids:
                    doc_id = str(doc_ids)
                chunk = TextChunk(
                    chunk_id=str(row.get("id", f"base_{idx}")),
                    doc_id=doc_id,
                    text=text,
                    start_page=0,
                    end_page=0,
                    chunk_index=idx,
                    is_bridge=False,
                )
                chunk.source_label = "base"
                chunks.append(chunk)
            return chunks
    raise FileNotFoundError(f"Missing text_units parquet under {graphrag_dir}")


def load_extra_chunks(path: Path, start_index: int) -> list[TextChunk]:
    if not path:
        return []
    if path.suffix.lower() == ".parquet":
        rows = pd.read_parquet(path).to_dict("records")
    elif path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("chunks", [])
    else:
        rows = [
            {"text": part.strip(), "doc_id": path.stem}
            for part in split_plain_text(read_text_with_fallback(path))
            if part.strip()
        ]

    chunks = []
    for idx, row in enumerate(rows, start=start_index):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunk = TextChunk(
            chunk_id=str(row.get("chunk_id", f"extra_{idx}")),
            doc_id=str(row.get("doc_id", f"extra_doc_{idx}")),
            text=text,
            start_page=int(row.get("start_page", 0) or 0),
            end_page=int(row.get("end_page", 0) or 0),
            chunk_index=idx,
            is_bridge=bool(row.get("is_bridge", False)),
        )
        chunk.source_label = "augmented"
        chunks.append(chunk)
    return chunks


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


def split_plain_text(text: str, target_size: int = 360, min_size: int = 220) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[。！？；!?;])", text) if s.strip()]
    if not sentences:
        return [text[i : i + target_size] for i in range(0, len(text), target_size)]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        if len(current) + len(sentence) <= target_size:
            current += sentence
            continue
        if len(current) < min_size and len(sentence) < min_size:
            current += sentence
            continue
        chunks.append(current)
        current = sentence

    if current:
        if chunks and len(current) < min_size:
            chunks[-1] += current
        else:
            chunks.append(current)
    return chunks


def write_manifest(output_dir: Path, base_count: int, extra_count: int, graph_dir: Path, extra_path: Path | None):
    manifest = {
        "graph_artifacts_dir": str(graph_dir),
        "output_vector_dir": str(output_dir),
        "base_chunk_count": base_count,
        "extra_chunk_count": extra_count,
        "total_chunk_count": base_count + extra_count,
        "extra_chunks_path": str(extra_path) if extra_path else None,
    }
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Build text_base/text_augmented vector stores.")
    parser.add_argument("--graphrag-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-chunks", default=None, help="Optional extra chunk file for augmented store.")
    parser.add_argument("--variant", choices=["text_base", "text_augmented"], required=True)
    parser.add_argument("--embed-batch-size", type=int, default=1, help="Embedding batch size for Ollama.")
    args = parser.parse_args()

    graph_dir = Path(args.graphrag_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.embedding.batch_size = max(1, args.embed_batch_size)

    base_chunks = load_base_chunks(graph_dir)
    extra_chunks: list[TextChunk] = []
    if args.variant == "text_augmented" and args.extra_chunks:
        extra_chunks = load_extra_chunks(Path(args.extra_chunks), len(base_chunks))

    vector_store.configure(output_dir, reset=True)
    vector_store.replace_chunks(base_chunks + extra_chunks)
    write_manifest(output_dir, len(base_chunks), len(extra_chunks), graph_dir, Path(args.extra_chunks) if args.extra_chunks else None)
    print(json.dumps({"variant": args.variant, "counts": vector_store.count(), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
