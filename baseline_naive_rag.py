"""
基线系统1：Naive RAG
════════════════════
只用 ChromaDB 向量检索 + LLM 直接生成，不使用知识图谱。
与 geo_graphrag 共享同一个向量库和 LLM，保证控制变量。

用法：
    cd adaptive-geological-graphrag-reproducibility
    python baseline_naive_rag.py query "霍邱县班台子铁矿的控矿构造是什么？"
"""
import sys
import time
import logging
from pathlib import Path

# 复用主系统的路径和配置
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)  # 基线只打印WARNING以上，保持输出简洁
logger = logging.getLogger("naive_rag")


def query_naive_rag(question: str) -> tuple[str, float]:
    """
    Naive RAG 查询：向量检索 Top-5 + LLM 直接回答。
    返回 (答案文本, 耗时秒数)
    """
    t0 = time.time()

    # ── 1. 向量检索 ───────────────────────────────────────────────────
    from src.indexing.vectorizer import vector_store
    results = vector_store.search(question, top_k=5)

    if not results:
        return "未检索到相关内容。", time.time() - t0

    # ── 2. 拼接上下文 ─────────────────────────────────────────────────
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(f"[段落{i}] {r['text'][:500]}")
    context = "\n\n".join(context_parts)

    # ── 3. LLM 生成 ───────────────────────────────────────────────────
    from src.llm.client import llm

    prompt = f"""你是地质领域的专业问答助手。请根据以下检索到的文本段落回答问题。

【检索段落】
{context}

【问题】
{question}

【要求】
- 直接给出简洁明确的答案
- 若段落中有明确信息，直接引用
- 若信息不足，说明"根据现有资料无法确定"

【答案】"""

    answer = llm.generate(prompt)
    elapsed = time.time() - t0
    return answer, elapsed


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "query":
        print("用法: python baseline_naive_rag.py query <问题>")
        sys.exit(1)

    question = sys.argv[2]
    answer, elapsed = query_naive_rag(question)
    print(f"\n{'='*60}")
    print(f"【Naive RAG 答案】（耗时 {elapsed:.1f}s）")
    print(f"{'='*60}")
    print(answer)
