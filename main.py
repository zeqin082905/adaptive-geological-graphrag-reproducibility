"""CLI entrypoint for build/query/evaluation workflows."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import cfg
from src.ingestion.loader import CorpusLoader
from src.indexing.chunker import GeoChunker
from src.indexing.graph_builder import GraphRAGBuilder
from src.indexing.vectorizer import vector_store
from src.query.generator import answer_generator
from src.query.normalizer import normalizer
from src.query.retriever import DualPathRetriever

logging.basicConfig(
    level=getattr(logging, cfg.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("geo_graphrag.main")


def resolve_runtime_paths(
    graph_dir: Optional[str] = None,
    text_vector_dir: Optional[str] = None,
) -> dict:
    graph_path = Path(graph_dir) if graph_dir else Path(cfg.index_paths.graph_artifacts_dir)
    text_path = Path(text_vector_dir) if text_vector_dir else Path(cfg.index_paths.text_vector_dir)
    return {
        "graph_artifacts_dir": graph_path,
        "text_vector_dir": text_path,
        "corpus1_dir": Path(cfg.index_paths.corpus1_dir),
        "corpus2_dir": Path(cfg.index_paths.corpus2_dir),
    }


def build_pipeline(
    skip_graphrag: bool = False,
    text_vector_dir: Optional[str] = None,
    corpus1_dir: Optional[str] = None,
    corpus2_dir: Optional[str] = None,
):
    loader = CorpusLoader()
    chunker = GeoChunker()
    corpus1_root = Path(corpus1_dir) if corpus1_dir else Path(cfg.index_paths.corpus1_dir)
    corpus2_root = Path(corpus2_dir) if corpus2_dir else Path(cfg.index_paths.corpus2_dir)

    logger.info("Build corpus1 from %s", corpus1_root)
    logger.info("Build corpus2 from %s", corpus2_root)

    corpus1_docs = loader.load_directory(corpus1_root)
    corpus2_docs = loader.load_directory(corpus2_root)
    logger.info("Loaded corpus1=%s corpus2=%s", len(corpus1_docs), len(corpus2_docs))

    vector_store.configure(text_vector_dir)
    chunks = chunker.chunk_corpus(corpus2_docs)
    logger.info("Generated chunks=%s", len(chunks))
    vector_store.replace_chunks(chunks)
    logger.info("Text vector store status: %s", vector_store.count())

    if skip_graphrag or not corpus1_docs:
        return

    builder = GraphRAGBuilder()
    builder.setup_workspace()
    builder.prepare_corpus(corpus1_docs)
    builder.run_prompt_tuning()
    builder.run_indexing()


def query_pipeline(
    question: str,
    graphrag_output_dir: Optional[str] = None,
    text_vector_dir: Optional[str] = None,
    ablation_flags: Optional[dict] = None,
    return_details: bool = False,
):
    runtime = resolve_runtime_paths(graphrag_output_dir, text_vector_dir)
    vector_store.configure(runtime["text_vector_dir"])

    try:
        builder = GraphRAGBuilder()
        graph_artifacts = builder.load_artifacts(output_dir=runtime["graph_artifacts_dir"])
    except Exception as exc:
        logger.warning("Graph artifacts load failed, graph path disabled: %s", exc)
        graph_artifacts = None

    norm_result = normalizer.normalize(question)
    retriever = DualPathRetriever(graph_artifacts=graph_artifacts, ablation_flags=ablation_flags)
    retrieval_result = retriever.retrieve(
        normalized_query=norm_result.normalized_query,
        extracted_entities=norm_result.extracted_entities,
        keyword_terms=norm_result.keyword_terms,
    )
    if graph_artifacts and "create_final_community_reports" in graph_artifacts:
        retrieval_result.community_reports_df = graph_artifacts["create_final_community_reports"]

    answer = answer_generator.generate(question, retrieval_result)
    if not return_details:
        return answer

    return {
        "answer": answer,
        "runtime": {
            "graph_artifacts_dir": str(runtime["graph_artifacts_dir"]),
            "text_vector_dir": str(runtime["text_vector_dir"]),
        },
        "retrieval": {
            "text_evidence_count": len(retrieval_result.text_evidences),
            "graph_evidence_count": len(retrieval_result.graph_evidences),
            "triggered_decomposition": retrieval_result.triggered_decomposition,
            "sub_question_count": len(retrieval_result.sub_questions),
        },
        "ablation_flags": ablation_flags or {},
    }


def _mode_to_flags(mode: str) -> dict:
    return {
        "disable_graph": mode in ("base", "no_graph", "graph_disabled", "text_base_only", "text_augmented_only", "naive_rag"),
        "disable_reranker": mode in ("base", "no_reranker", "naive_rag"),
        "disable_decompose": mode in ("base", "no_decompose", "graph_only", "naive_rag"),
    }


def main():
    parser = argparse.ArgumentParser(description="Geo GraphRAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build corpora indexes")
    build_parser.add_argument("--skip-graphrag", action="store_true")
    build_parser.add_argument("--text-vector-dir", type=str, default=None)
    build_parser.add_argument("--corpus1-dir", type=str, default=None)
    build_parser.add_argument("--corpus2-dir", type=str, default=None)

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str)
    query_parser.add_argument("--graph-dir", type=str, default=None)
    query_parser.add_argument("--graphrag-dir", type=str, default=None)
    query_parser.add_argument("--text-vector-dir", type=str, default=None)
    query_parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "full",
            "base",
            "no_graph",
            "no_reranker",
            "no_decompose",
            "graph_only",
            "text_base_only",
            "text_augmented_only",
            "naive_rag",
        ],
    )
    query_parser.add_argument("--json", action="store_true", help="Print machine-readable result")

    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "build":
        build_pipeline(
            skip_graphrag=args.skip_graphrag,
            text_vector_dir=args.text_vector_dir,
            corpus1_dir=args.corpus1_dir,
            corpus2_dir=args.corpus2_dir,
        )
        return

    if args.command == "query":
        graph_dir = args.graph_dir or args.graphrag_dir
        flags = _mode_to_flags(args.mode)
        result = query_pipeline(
            args.question,
            graphrag_output_dir=graph_dir,
            text_vector_dir=args.text_vector_dir,
            ablation_flags=flags,
            return_details=args.json,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        print("\n" + "=" * 60)
        print("【系统答案】")
        print("=" * 60)
        print(result)
        print("\n" + "=" * 60)
        print("【运行配置】")
        print("=" * 60)
        runtime = resolve_runtime_paths(graph_dir, args.text_vector_dir)
        print(f"graph_artifacts_dir={runtime['graph_artifacts_dir']}")
        print(f"text_vector_dir={runtime['text_vector_dir']}")
        print(f"mode={args.mode}")
        return

    if args.command == "serve":
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel
            import uvicorn
        except ImportError as exc:
            raise ImportError("Please install fastapi and uvicorn.") from exc

        app = FastAPI(title="地质资料智能问答 API", version="1.0.0")

        class QueryRequest(BaseModel):
            question: str
            graph_dir: Optional[str] = None
            text_vector_dir: Optional[str] = None
            mode: str = "full"

        @app.post("/query")
        def query_endpoint(req: QueryRequest):
            flags = _mode_to_flags(req.mode)
            return query_pipeline(
                req.question,
                graphrag_output_dir=req.graph_dir,
                text_vector_dir=req.text_vector_dir,
                ablation_flags=flags,
                return_details=True,
            )

        @app.get("/health")
        def health():
            return {"status": "ok", "vector_store": vector_store.count()}

        uvicorn.run(app, host=args.host, port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
