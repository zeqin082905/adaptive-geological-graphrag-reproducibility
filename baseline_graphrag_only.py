"""
基线系统2：GraphRAG-only
═════════════════════════
只用知识图谱路检索（实体+关系+社区），不使用向量文本路。
与 geo_graphrag 共享同一份 parquet 文件和 LLM。

用法：
    cd adaptive-geological-graphrag-reproducibility
    python baseline_graphrag_only.py query "霍邱县班台子铁矿的控矿构造是什么？"
"""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("graphrag_only")


def query_graphrag_only(question: str) -> tuple[str, float]:
    """
    GraphRAG-only 查询：仅图谱路检索 + LLM 生成。
    返回 (答案文本, 耗时秒数)
    """
    t0 = time.time()

    # ── 1. 加载图谱 artifacts ─────────────────────────────────────────
    from src.indexing.graph_builder import GraphRAGBuilder
    builder = GraphRAGBuilder()
    artifacts = builder.load_artifacts()

    # ── 2. 图谱路检索 ─────────────────────────────────────────────────
    from src.query.retriever import GraphRetriever
    graph_retriever = GraphRetriever(artifacts=artifacts)
    graph_evidences = graph_retriever.retrieve(
        normalized_query=question,
        extracted_entities=[],
        keyword_terms=[]
    )

    if not graph_evidences:
        return "知识图谱中未找到相关实体。", time.time() - t0

    # ── 3. 构建图谱上下文 ─────────────────────────────────────────────
    context_parts = []
    for ev in graph_evidences[:8]:  # 取前8个实体
        entity_block = f"实体: {ev.entity_name}（{ev.entity_type}）\n描述: {ev.entity_description[:200]}"
        if ev.related_relationships:
            rels = ev.related_relationships[:3]
            rel_texts = []
            for r in rels:
                src = r.get("source", "")
                tgt = r.get("target", "")
                desc = r.get("description", "")[:100]
                rel_texts.append(f"  {src} → {tgt}: {desc}")
            entity_block += "\n相关关系:\n" + "\n".join(rel_texts)
        if ev.community_summary:
            entity_block += f"\n社区摘要: {ev.community_summary[:150]}"
        context_parts.append(entity_block)

    context = "\n\n---\n\n".join(context_parts)

    # ── 4. LLM 生成 ───────────────────────────────────────────────────
    from src.llm.client import llm

    prompt = f"""你是地质领域的专业问答助手。请根据以下知识图谱中的实体和关系回答问题。

【知识图谱证据】
{context}

【问题】
{question}

【要求】
- 直接给出简洁明确的答案
- 基于图谱实体和关系作答
- 若信息不足，说明"根据现有图谱无法确定"

【答案】"""

    answer = llm.generate(prompt)
    elapsed = time.time() - t0
    return answer, elapsed


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "query":
        print("用法: python baseline_graphrag_only.py query <问题>")
        sys.exit(1)

    question = sys.argv[2]
    answer, elapsed = query_graphrag_only(question)
    print(f"\n{'='*60}")
    print(f"【GraphRAG-only 答案】（耗时 {elapsed:.1f}s）")
    print(f"{'='*60}")
    print(answer)
