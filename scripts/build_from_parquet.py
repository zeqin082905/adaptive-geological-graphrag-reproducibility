"""
scripts/build_from_parquet.py
从已有的 GraphRAG 输出直接构建文本向量库，同时预热实体向量缓存。

使用方式：
    python scripts/build_from_parquet.py --graphrag-dir "index/graphrag/output"

完成后效果：
    ✅ ChromaDB 向量库建好（文本路可以工作）
    ✅ 实体向量缓存建好（图谱路从10分钟→2秒）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import cfg, GRAPHRAG_DIR
from src.indexing.chunker import TextChunk
from src.indexing.vectorizer import vector_store, embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_from_parquet")


def build_vector_store(graphrag_dir: Path, batch_size: int):
    logger.info("═══ 步骤1：构建文本向量库 ═══")
    candidates = ["text_units.parquet", "create_final_text_units.parquet"]
    df = None
    for name in candidates:
        path = graphrag_dir / name
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"加载 {name}: {len(df)} 条")
            break
    if df is None:
        logger.error("未找到 text_units.parquet，跳过")
        return

    chunks = []
    for idx, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunk_id = str(row.get("id", f"chunk_{idx}"))
        doc_ids = row.get("document_ids", None)
        if isinstance(doc_ids, list) and doc_ids:
            doc_id = str(doc_ids[0])
        elif isinstance(doc_ids, str) and doc_ids:
            doc_id = doc_ids
        else:
            doc_id = "unknown"
        chunks.append(TextChunk(
            chunk_id=chunk_id, doc_id=doc_id, text=text,
            start_page=0, end_page=0, chunk_index=idx, is_bridge=False,
        ))

    logger.info(f"有效文本块: {len(chunks)} 条，写入 ChromaDB...")
    vector_store.add_chunks(chunks, batch_size=batch_size)
    counts = vector_store.count()
    logger.info(f"✅ 向量库完成 | 普通块: {counts['text_chunks']}，桥接块: {counts['bridge_chunks']}")


def build_entity_cache(graphrag_dir: Path):
    logger.info("═══ 步骤2：预热实体向量缓存 ═══")
    candidates = ["entities.parquet", "create_final_entities.parquet"]
    df = None
    for name in candidates:
        path = graphrag_dir / name
        if path.exists():
            df = pd.read_parquet(path)
            logger.info(f"加载 {name}: {len(df)} 条实体")
            break
    if df is None:
        logger.warning("未找到 entities.parquet，跳过")
        return

    fingerprint = f"{len(df)}_{str(df.iloc[0].get('id', '0'))[:8]}"
    cache_path = GRAPHRAG_DIR / f"entity_vecs_{fingerprint}.npy"

    if cache_path.exists():
        logger.info(f"✅ 缓存已存在，跳过: {cache_path}")
        return

    name_col = "title" if "title" in df.columns else "name"
    name_texts = df.get(name_col, df.get("name", "")).fillna("")
    desc_texts = df.get("description", "").fillna("")
    texts = (name_texts + " " + desc_texts).tolist()

    logger.info(f"编码 {len(texts)} 个实体（仅此一次）...")
    vecs = np.array(embedder.encode(texts))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), vecs)
    logger.info(f"✅ 缓存已保存: {cache_path}")
    logger.info(f"   后续查询图谱路耗时 < 2秒（原来约10分钟）")


def main():
    parser = argparse.ArgumentParser(description="构建向量库并预热实体缓存")
    parser.add_argument("--graphrag-dir", required=True, help="GraphRAG 输出目录路径")
    parser.add_argument("--batch-size", type=int, default=100, help="批次大小（默认100）")
    parser.add_argument("--skip-vector-store", action="store_true", help="仅预热实体缓存")
    parser.add_argument("--skip-entity-cache", action="store_true", help="仅构建向量库")
    args = parser.parse_args()

    graphrag_dir = Path(args.graphrag_dir)
    if not graphrag_dir.exists():
        logger.error(f"目录不存在: {graphrag_dir}")
        sys.exit(1)

    if not args.skip_vector_store:
        build_vector_store(graphrag_dir, args.batch_size)
    if not args.skip_entity_cache:
        build_entity_cache(graphrag_dir)

    logger.info("🎉 准备完成！运行：python main.py query \"您的地质问题\"")


if __name__ == "__main__":
    main()
