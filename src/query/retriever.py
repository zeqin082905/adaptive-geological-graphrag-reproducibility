"""
步骤8-11：双路并行混合检索器
═══════════════════════════════════════
文本路（步骤8-10）：
  初步检索 → 反馈判断（相似度 + 主题一致性）→ 子问题分解 → Reranker精排

图谱路（步骤11）：
  精确实体匹配 → 模糊相似检索 → 社区重排序 → 图谱证据集合

两路并行，结果汇聚至步骤12融合生成。
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import cfg, RetrievalConfig, RerankerConfig
from src.indexing.vectorizer import vector_store, embedder
from src.llm.client import llm

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────
@dataclass
class TextEvidence:
    chunk_id: str
    text: str
    similarity: float
    doc_id: str
    start_page: int
    end_page: int
    sub_question_id: Optional[str] = None
    rerank_score: Optional[float] = None
    source_title: Optional[str] = None   # 原始文档标题（文件名）
    page_info: Optional[str] = None      # 可读页码信息，如 "第3-4页"


@dataclass
class GraphEvidence:
    entity_id: str
    entity_name: str
    entity_type: str
    entity_description: str
    related_relationships: list[dict] = field(default_factory=list)
    community_id: Optional[str] = None
    community_summary: Optional[str] = None
    community_findings: Optional[object] = None   # findings 字段（结构化发现列表）
    community_rank: Optional[float] = None        # 社区重要性评分
    entity_degree: Optional[int] = None           # 实体连接度（越高越重要）
    entity_frequency: Optional[int] = None        # 实体出现频次
    match_type: str = "exact"                     # exact / fuzzy / keyword


@dataclass
class RetrievalResult:
    query: str
    normalized_query: str
    triggered_decomposition: bool
    sub_questions: list[dict] = field(default_factory=list)
    text_evidences: list[TextEvidence] = field(default_factory=list)
    graph_evidences: list[GraphEvidence] = field(default_factory=list)
    community_reports_df: Optional[object] = None  # 供 generator 读取 findings


# ── Reranker ──────────────────────────────────────────────────────────────────
class Reranker:
    """BGE-Reranker-v2 本地精排序模型。"""

    def __init__(self, config: Optional[RerankerConfig] = None):
        self._cfg = config or cfg.reranker
        self._model = None

    def _load(self):
        if self._model is None:
            # 修复 FlagEmbedding 内部引用 Optional 未导入的问题
            import builtins
            from typing import Optional as _Optional
            if not hasattr(builtins, 'Optional'):
                builtins.Optional = _Optional

            from FlagEmbedding import FlagReranker
            logger.info(f"加载 Reranker: {self._cfg.model_name}")
            self._model = FlagReranker(
                self._cfg.model_name,
                use_fp16=True,
                device=self._cfg.device,
            )
    def rerank(self, query: str, candidates: list[dict], top_p: int) -> list[dict]:
        """对候选文本块精排，返回 Top-P 结果。若 Reranker 不可用则降级为相似度排序。"""
        if not candidates:
            return []
        try:
            self._load()
            pairs = [[query, c["text"]] for c in candidates]
            scores = self._model.compute_score(pairs, normalize=True)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_p]
        except Exception as e:
            logger.warning(f"Reranker 不可用，降级为相似度排序: {e}")
            # 降级：直接按向量相似度排序
            sorted_candidates = sorted(candidates, key=lambda x: x.get("similarity", 0), reverse=True)
            return sorted_candidates[:top_p]


# ── 子问题分解器（步骤9） ─────────────────────────────────────────────────────
class SubQuestionDecomposer:

    DECOMPOSE_PROMPT = """\
【Role 角色】
你是一位地质矿产领域的专业分析师，具备矿床学、构造地质学、地球化学等多学科背景，\
能够将复杂综合性地质问题拆解为逻辑清晰、可独立检索的子问题单元。

【Action 动作】
对输入的复杂地质问题执行结构化拆解：
1. 识别问题中包含的多个独立地质查询意图
2. 按地质领域任务逻辑将其映射为 2-5 个子问题
3. 为每个子问题生成主查询句与 2 个表达变体（用于提升检索召回率）
4. 判断每个子问题最适合从文本库（text）还是知识图谱（graph）或两者（both）中检索证据
5. 以严格 JSON 数组格式输出，不附加任何说明文字

【Scope 范围】
拆解策略框架（按地质认知层次优先排序）：
① 物质组成：矿石矿物、岩石类型、化学成分
② 结构构造：控矿构造、赋矿层位、空间展布
③ 成因机制：成矿作用、物质来源、流体条件
④ 时空分布：成矿时代、矿体规模、矿区范围
⑤ 资源评价：储量级别、品位、资源量

子问题数量约束：
- 复杂综合问题：2-5 个子问题
- 单一意图问题（如"XX矿的储量是多少"）：输出含原问题的单元素数组，不强行拆解

【Example 示例】
输入问题："为什么云南兰坪铅锌矿能形成如此大规模的矿床？"

输出：
[
  {{
    "id": "sq_1",
    "core_intent": "成矿物质来源",
    "primary_query": "兰坪铅锌矿的成矿物质主要来源是什么？",
    "query_variants": ["兰坪矿床铅锌的物质来源分析", "兰坪铅锌矿成矿元素富集机制"],
    "evidence_type_preference": "graph",
    "constraint": "云南兰坪"
  }},
  {{
    "id": "sq_2",
    "core_intent": "构造控矿条件",
    "primary_query": "哪些构造为兰坪矿液运移提供了通道？",
    "query_variants": ["兰坪矿区控矿断裂特征", "兰坪矿床构造控矿机制"],
    "evidence_type_preference": "graph",
    "constraint": "云南兰坪"
  }},
  {{
    "id": "sq_3",
    "core_intent": "大规模成矿的物化条件",
    "primary_query": "兰坪铅锌矿大规模沉淀的关键物理化学条件是什么？",
    "query_variants": ["兰坪矿床成矿流体特征", "兰坪铅锌沉淀的温压条件"],
    "evidence_type_preference": "both",
    "constraint": null
  }}
]

【Format 格式】
输出为严格 JSON 数组，每个元素包含以下字段：
{{
  "id": "sq_N",
  "core_intent": "一句话核心意图",
  "primary_query": "完整主查询语句",
  "query_variants": ["变体1", "变体2"],
  "evidence_type_preference": "text" | "graph" | "both",
  "constraint": "时间/空间/矿种约束" | null
}}
禁止输出 JSON 以外的任何内容（无前缀、无解释、无 markdown 代码块标记）。

原始问题：{query}

子问题 JSON 数组："""

    def decompose(self, query: str) -> list[dict]:
        """调用 LLM 将复杂问题拆解为结构化子问题集合。"""
        try:
            prompt = self.DECOMPOSE_PROMPT.format(query=query)
            result = llm.generate_json(prompt)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "sub_questions" in result:
                return result["sub_questions"]
            return []
        except Exception as e:
            logger.error(f"子问题分解失败: {e}")
            # 降级：返回原始问题作为唯一子问题
            return [{
                "id": "sq_1",
                "core_intent": query,
                "primary_query": query,
                "query_variants": [query],
                "evidence_type_preference": "both",
                "constraint": None,
            }]


# ── 文本路检索器（步骤8-10） ──────────────────────────────────────────────────
class TextRetriever:

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        ablation_flags: Optional[dict] = None,
    ):
        self._cfg = config or cfg.retrieval
        flags = ablation_flags or {}
        self._disable_reranker    = flags.get("disable_reranker", False)
        self._disable_decompose   = flags.get("disable_decompose", False)
        self._reranker = Reranker()
        self._decomposer = SubQuestionDecomposer()

    def retrieve(self, normalized_query: str) -> tuple[list[TextEvidence], bool, list[dict]]:
        """
        返回 (text_evidences, triggered_decomposition, sub_questions)
        """
        # ── 步骤8：初步检索 + 反馈判断 ──────────────────────────────────────
        initial_results = vector_store.search(normalized_query, top_k=self._cfg.initial_top_k)

        if not initial_results:
            logger.warning("向量库为空，跳过检索")
            return [], False, []

        triggered = self._feedback_check(initial_results)

        # 消融：强制禁用子问题分解
        if self._disable_decompose:
            triggered = False
            logger.info("[消融] 子问题分解已禁用，直接使用初步检索结果")

        if not triggered:
            # 初步检索质量足够，直接转换为证据
            evidences = [
                TextEvidence(
                    chunk_id=r.get("chunk_id", ""),
                    text=r["text"],
                    similarity=r["similarity"],
                    doc_id=r.get("doc_id", ""),
                    start_page=r.get("start_page", 0),
                    end_page=r.get("end_page", 0),
                )
                for r in initial_results
            ]
            return evidences, False, []

        # ── 步骤9：子问题分解 ─────────────────────────────────────────────────
        sub_questions = self._decomposer.decompose(normalized_query)
        logger.info(f"子问题分解: {len(sub_questions)} 个子问题")

        # ── 步骤10：子问题向量检索 + Reranker精排 ────────────────────────────
        all_evidences: list[TextEvidence] = []
        for sq in sub_questions:
            queries = [sq["primary_query"]] + sq.get("query_variants", [])
            candidate_pool: list[dict] = []

            for q in queries:
                results = vector_store.search(q, top_k=50)
                candidate_pool.extend(results)

            # 去重
            seen_texts = set()
            deduped_pool = []
            for c in candidate_pool:
                key = c["text"][:80]
                if key not in seen_texts:
                    seen_texts.add(key)
                    deduped_pool.append(c)

            # Reranker 精排（消融：禁用时降级为相似度排序）
            if self._disable_reranker:
                logger.info("[消融] Reranker 已禁用，使用相似度排序")
                reranked = sorted(
                    deduped_pool, key=lambda x: x.get("similarity", 0), reverse=True
                )[:cfg.reranker.top_p]
            else:
                reranked = self._reranker.rerank(
                    sq["primary_query"], deduped_pool, cfg.reranker.top_p
                )

            for r in reranked:
                all_evidences.append(TextEvidence(
                    chunk_id=r.get("chunk_id", ""),
                    text=r["text"],
                    similarity=r.get("similarity", 0.0),
                    doc_id=r.get("doc_id", ""),
                    start_page=r.get("start_page", 0),
                    end_page=r.get("end_page", 0),
                    sub_question_id=sq["id"],
                    rerank_score=r.get("rerank_score"),
                ))

        return all_evidences, True, sub_questions

    def _feedback_check(self, results: list[dict]) -> bool:
        """
        步骤8 检索反馈判断：
          返回 True  → 需要触发子问题分解（步骤9）
          返回 False → 初步检索质量足够，直接使用
        """
        if not results:
            return True

        # 相似度判断
        s1 = results[0]["similarity"]
        if s1 < self._cfg.sim_threshold:
            logger.info(f"初步检索相似度不足 ({s1:.3f} < {self._cfg.sim_threshold})，触发分解")
            return True

        # 主题一致性判断（公式：C = 2/k(k-1) * Σ cos(vi,vj)）
        k = len(results)
        if k < 2:
            return False

        texts = [r["text"] for r in results]
        vectors = embedder.encode(texts)
        vecs = np.array(vectors)

        cos_sum = 0.0
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                vi, vj = vecs[i], vecs[j]
                cos_sim = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9))
                cos_sum += cos_sim
                count += 1

        C = cos_sum / count if count > 0 else 1.0

        if C < self._cfg.coherence_threshold:
            logger.info(f"主题一致性不足 (C={C:.3f} < {self._cfg.coherence_threshold})，触发分解")
            return True

        logger.info(f"初步检索通过: S1={s1:.3f}, C={C:.3f}")
        return False


# ── 图谱路检索器（步骤11） ────────────────────────────────────────────────────
class GraphRetriever:
    """
    从 GraphRAG 输出的实体/关系/社区 Parquet 文件中检索图谱证据。
    """

    def __init__(self, artifacts: Optional[dict] = None, config: Optional[RetrievalConfig] = None):
        self._artifacts = artifacts  # GraphRAGBuilder.load_artifacts() 的返回值
        self._cfg = config or cfg.retrieval

    def retrieve(self, normalized_query: str, extracted_entities: list[dict], keyword_terms: list[str] = None) -> list[GraphEvidence]:
        """步骤11：关键词预匹配 + 向量相似度检索，返回图谱证据集合。"""
        if not self._artifacts:
            logger.warning("图谱未加载，跳过图谱路检索")
            return []

        entities_df = self._artifacts.get("create_final_entities")
        relationships_df = self._artifacts.get("create_final_relationships")
        community_reports_df = self._artifacts.get("create_final_community_reports")

        if entities_df is None or entities_df.empty:
            return []

        entity_name_col = "title" if "title" in entities_df.columns else "name"
        entity_type_col = "type" if "type" in entities_df.columns else "entity_type"

        # ── 第一步：关键词精确预匹配 ─────────────────────────────────────────
        keyword_indices = self._keyword_match(
            normalized_query, entities_df, entity_name_col,
            extra_keywords=keyword_terms or []
        )
        logger.info(f"关键词预匹配命中实体: {len(keyword_indices)} 个")

        # ── 第二步：向量相似度检索 ──────────────────────────────────────────
        query_vec = np.array(embedder.encode_single(normalized_query))
        entity_vecs = self._get_entity_vectors(entities_df, entity_name_col)
        similarities = self._cosine_similarity_batch(query_vec, entity_vecs)

        exact_mask = similarities >= self._cfg.exact_match_threshold
        vector_indices = list(np.where(exact_mask)[0])

        if not vector_indices:
            E = self._cfg.fuzzy_keep * self._cfg.oversample_scaler
            top_indices = np.argsort(similarities)[::-1][:E]
            vector_indices = list(top_indices[:self._cfg.fuzzy_keep])
            match_type = "fuzzy"
        else:
            match_type = "exact"

        # ── 第三步：合并结果，关键词命中优先 ─────────────────────────────────
        # 策略：若关键词命中数量充足（≥3），则向量检索仅作补充而非主导
        # 这样可防止"柯奎北断裂"因向量相似命中"柯大兴矿区"等跑偏实体
        if len(keyword_indices) >= 3:
            # 关键词命中充足：向量只补充关键词未命中的实体，上限3个
            vector_supplement = [i for i in vector_indices if i not in set(keyword_indices)][:3]
            all_indices = list(dict.fromkeys(keyword_indices + vector_supplement))
            logger.info(f"关键词命中充足，向量仅补充 {len(vector_supplement)} 个实体")
        else:
            # 关键词命中不足：全量合并，关键词优先
            all_indices = list(dict.fromkeys(keyword_indices + vector_indices))

        # ── 第四步：对向量命中的实体做名称相关性二次验证 ──────────────────────
        # 防止向量把名字相似但语义不同的实体混入（如"柯奎北断裂"混入"柯大兴矿区"）
        keyword_set = set(keyword_indices)
        verified_indices = []
        query_chars = set(normalized_query)
        for idx in all_indices:
            if idx in keyword_set:
                verified_indices.append(idx)  # 关键词命中的直接保留
                continue
            # 向量命中的：实体名称至少与查询共享2个以上连续汉字才保留
            entity_name_str = str(entities_df.iloc[idx].get(entity_name_col, ""))
            if self._name_relevance_check(normalized_query, entity_name_str):
                verified_indices.append(idx)
            else:
                logger.debug(f"过滤低相关实体: '{entity_name_str}'（向量命中但名称与查询不相关）")

        all_indices = verified_indices
        logger.info(f"合并后候选实体: {len(all_indices)} 个（关键词:{len(keyword_indices)} + 向量验证后:{len(all_indices)-len(keyword_indices)}）")

        evidences = []
        for idx in all_indices:
            row = entities_df.iloc[idx]
            entity_id = str(row.get("id", idx))
            entity_name = str(row.get(entity_name_col, ""))
            is_keyword_hit = idx in keyword_indices

            related_rels = self._get_relations(entity_name, relationships_df)
            community_id, community_summary = self._get_community(
                entity_id, community_reports_df
            )

            evidences.append(GraphEvidence(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=str(row.get(entity_type_col, "")),
                entity_description=str(row.get("description", "")),
                related_relationships=related_rels,
                community_id=community_id,
                community_summary=community_summary,
                entity_degree=int(row.get("degree", 0) or 0),
                entity_frequency=int(row.get("frequency", 0) or 0),
                match_type="keyword" if is_keyword_hit else match_type,
            ))

        # ── 第五步：反向全文扫描（针对"XX在哪些矿区/省份"类广度查询）──────────
        # 问题含"哪些"且核心词是矿物/地层/构造名时，
        # 在关系description里全文搜索，补充正向检索遗漏的实体
        reverse_evidences = self._reverse_relation_scan(
            normalized_query, keyword_terms or [],
            entities_df, relationships_df, community_reports_df,
            entity_name_col, entity_type_col,
            existing_ids={str(entities_df.iloc[i].get("id", i)) for i in all_indices}
        )
        if reverse_evidences:
            logger.info(f"反向扫描补充实体: {len(reverse_evidences)} 个")
            evidences.extend(reverse_evidences)

        evidences = self._rerank_by_community(evidences)
        return evidences

    @staticmethod
    def _keyword_match(query: str, entities_df, name_col: str, extra_keywords: list[str] = None) -> list[int]:
        """
        关键词精确预匹配：
          1. 从查询提取中文词组
          2. 合并 normalizer 提取的 keyword_terms（更精准）
          3. 在实体 title 和 description 中做包含匹配
          4. 短词（≤3字）且泛词且命中>20个时跳过，避免过度召回
        """
        import re

        # 通用地质泛词：这些词单独匹配会命中大量无关实体
        GENERIC_GEO_TERMS = {
            "铁矿", "铜矿", "金矿", "银矿", "锰矿", "铅矿", "锌矿", "磷矿",
            "钒矿", "镍矿", "矿区", "矿床", "矿体", "矿石", "地质", "构造",
            "断层", "背斜", "向斜", "岩石", "地层", "勘查", "勘察", "资源",
            "储量", "什么", "是否", "哪些", "如何", "怎么", "有没有",
            "的", "了", "吗", "呢", "控矿",
        }

        stop_words = {"什么", "是否", "哪些", "如何", "怎么", "有没有", "的", "了", "吗", "呢"}
        raw_candidates = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        keywords = [w for w in raw_candidates if w not in stop_words and 2 <= len(w) <= 8]

        # 合并 normalizer 提供的精确关键词（优先级更高）
        if extra_keywords:
            for kw in extra_keywords:
                if kw not in keywords:
                    keywords.insert(0, kw)

        if not keywords:
            return []

        logger.info(f"关键词预匹配词组: {keywords}")

        hit_indices = []
        name_series = entities_df[name_col].fillna("").str
        desc_series = entities_df.get("description", "").fillna("").str

        for kw in keywords:
            name_hits = entities_df[name_series.contains(kw, na=False)].index.tolist()

            # 短词（≤3字）且是泛词且命中过多时，跳过title匹配，避免"铁矿"命中几百条
            if len(kw) <= 3 and kw in GENERIC_GEO_TERMS and len(name_hits) > 20:
                logger.debug(f"跳过泛词title匹配: '{kw}' 命中{len(name_hits)}个，过于宽泛")
                continue

            hit_indices.extend(name_hits)

            # description匹配：只对长词（≥4字）且非泛词做，避免"铁矿"污染
            if len(kw) >= 4 and kw not in GENERIC_GEO_TERMS:
                desc_hits = entities_df[desc_series.contains(kw, na=False)].index.tolist()[:5]
                hit_indices.extend(desc_hits)

        unique_ids = list(dict.fromkeys(hit_indices))
        id_to_pos = {idx: pos for pos, idx in enumerate(entities_df.index)}
        return [id_to_pos[i] for i in unique_ids if i in id_to_pos]

    @staticmethod
    def _name_relevance_check(query: str, entity_name: str, min_overlap: int = 2) -> bool:
        """
        判断实体名称与查询是否有足够的字符重叠，用于过滤向量检索的跑偏实体。

        策略：
          - 实体名称中存在连续 min_overlap 个汉字出现在查询里，则认为相关
          - 或者查询中存在连续 min_overlap 个汉字出现在实体名称里
          - 实体名称很短（≤3字）时降低门槛为1字重叠
        """
        if not entity_name or not query:
            return True  # 无法判断时保守保留

        threshold = 1 if len(entity_name) <= 3 else min_overlap

        # 从实体名抽连续子串，检查是否出现在查询里
        for length in range(min(len(entity_name), 6), threshold - 1, -1):
            for start in range(len(entity_name) - length + 1):
                substr = entity_name[start:start + length]
                if len(substr) >= threshold and substr in query:
                    return True

        # 从查询抽连续子串，检查是否出现在实体名里
        for length in range(min(len(query), 6), threshold - 1, -1):
            for start in range(len(query) - length + 1):
                substr = query[start:start + length]
                if len(substr) >= threshold and substr in entity_name:
                    return True

        return False

    def _reverse_relation_scan(
        self,
        query: str,
        keyword_terms: list[str],
        entities_df,
        relationships_df,
        community_reports_df,
        entity_name_col: str,
        entity_type_col: str,
        existing_ids: set,
        max_extra: int = 8,
    ) -> list[GraphEvidence]:
        """
        反向全文扫描：针对"XX在哪些矿区/省份出现"类广度查询。

        正向检索只找与查询词直接匹配的实体，但对于"钨在哪些省份"这类问题，
        答案散落在多个含"钨"关系的不同实体里，正向检索只能找到部分。

        本方法在关系表的 description 字段做全文搜索，收集所有提及核心词的关系，
        再从这些关系推出相关实体，补充到检索结果中。

        触发条件：查询包含"哪些""哪几个""分布""出现""发现""存在于"等广度词。
        """
        import re

        # 只对广度查询触发
        broad_triggers = ["哪些", "哪几个", "哪些地", "分布", "出现", "发现", "存在于", "见于"]
        if not any(t in query for t in broad_triggers):
            return []

        if relationships_df is None or relationships_df.empty:
            return []

        # 提取核心搜索词：优先用 keyword_terms，其次从查询里提取≥2字的非泛词
        GENERIC = {
            "矿区", "矿床", "地层", "构造", "岩石", "省份", "地区",
            "哪些", "哪几", "发现", "分布", "出现", "存在", "找到",
        }
        search_terms = [kw for kw in keyword_terms if len(kw) >= 2 and kw not in GENERIC]
        if not search_terms:
            candidates = re.findall(r'[\u4e00-\u9fff]{2,6}', query)
            search_terms = [w for w in candidates if w not in GENERIC][:3]

        if not search_terms:
            return []

        src_col = "source" if "source" in relationships_df.columns else "source_id"
        tgt_col = "target" if "target" in relationships_df.columns else "target_id"
        desc_col = "description" if "description" in relationships_df.columns else None

        if desc_col is None:
            return []

        desc_series = relationships_df[desc_col].fillna("")

        # 在关系描述中搜索所有包含核心词的关系
        hit_mask = desc_series.str.contains("|".join(search_terms), na=False, regex=False)
        # 同时搜索实体名列（source/target）
        hit_mask = hit_mask | (
            relationships_df[src_col].fillna("").str.contains(
                "|".join(search_terms), na=False, regex=False
            )
        ) | (
            relationships_df[tgt_col].fillna("").str.contains(
                "|".join(search_terms), na=False, regex=False
            )
        )

        hit_rels = relationships_df[hit_mask]
        if hit_rels.empty:
            return []

        # 收集这些关系涉及的所有实体名
        related_entity_names: set[str] = set()
        for _, row in hit_rels.iterrows():
            related_entity_names.add(str(row.get(src_col, "")))
            related_entity_names.add(str(row.get(tgt_col, "")))
        related_entity_names.discard("")

        # 在实体表中找到这些实体，过滤掉已检索到的
        name_mask = entities_df[entity_name_col].isin(related_entity_names)
        extra_entities = entities_df[name_mask]

        extra_evidences = []
        seen_ids = set(existing_ids)
        for _, row in extra_entities.iterrows():
            entity_id = str(row.get("id", ""))
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)

            entity_name = str(row.get(entity_name_col, ""))
            related_rels = self._get_relations(entity_name, relationships_df, two_hop=False)
            community_id, community_summary = self._get_community(entity_id, community_reports_df)

            extra_evidences.append(GraphEvidence(
                entity_id=entity_id,
                entity_name=entity_name,
                entity_type=str(row.get(entity_type_col, "")),
                entity_description=str(row.get("description", "")),
                related_relationships=related_rels,
                community_id=community_id,
                community_summary=community_summary,
                entity_degree=int(row.get("degree", 0) or 0),
                entity_frequency=int(row.get("frequency", 0) or 0),
                match_type="reverse_scan",
            ))

            if len(extra_evidences) >= max_extra:
                break

        return extra_evidences

    def _get_entity_vectors(self, entities_df, name_col: str = "title") -> np.ndarray:
        """
        获取实体向量。优先级：
          1. parquet 自带 embedding 列（最快）
          2. 本地磁盘缓存 index/graphrag/entity_vecs.npy（第一次编码后保存）
          3. 实时编码（慢，仅首次触发）
        """
        # 优先使用 parquet 内置向量
        if "embedding" in entities_df.columns:
            vecs = entities_df["embedding"].tolist()
            if vecs and vecs[0] is not None:
                return np.array(vecs)

        # 读取磁盘缓存
        cache_path = self._get_cache_path(entities_df)
        if cache_path.exists():
            logger.info(f"✅ 加载实体向量缓存: {cache_path} ({len(entities_df)} 条)")
            return np.load(str(cache_path))

        # 首次：实时编码并保存缓存
        logger.info(f"首次编码 {len(entities_df)} 个实体向量，完成后将缓存到磁盘...")
        name_texts = entities_df.get(name_col, entities_df.get("name", "")).fillna("")
        desc_texts = entities_df.get("description", "").fillna("")
        texts = (name_texts + " " + desc_texts).tolist()
        vecs = np.array(embedder.encode(texts))

        # 保存缓存
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache_path), vecs)
        logger.info(f"✅ 实体向量已缓存: {cache_path}（后续查询将直接加载，无需重新编码）")
        return vecs

    @staticmethod
    def _get_cache_path(entities_df) -> Path:
        """根据实体数量和首个ID生成缓存文件路径，不同语料自动区分。"""
        from config.settings import GRAPHRAG_DIR
        fingerprint = f"{len(entities_df)}_{str(entities_df.iloc[0].get('id', '0'))[:8]}"
        return GRAPHRAG_DIR / f"entity_vecs_{fingerprint}.npy"

    @staticmethod
    def _cosine_similarity_batch(query_vec: np.ndarray, entity_vecs: np.ndarray) -> np.ndarray:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        entity_norms = entity_vecs / (np.linalg.norm(entity_vecs, axis=1, keepdims=True) + 1e-9)
        return entity_norms @ query_norm

    def _get_relations(self, entity_name: str, relationships_df,
                       two_hop: bool = True, max_rels: int = 15) -> list[dict]:
        """
        获取实体关系列表，支持二跳扩展。
        
        一跳：直接与 entity_name 相连的关系。
        二跳：沿一跳邻居继续展开，只取 weight≥5 的高权重关系，
              帮助发现"断裂→矿区"等藏在多跳路径里的答案。
        """
        if relationships_df is None or relationships_df.empty:
            return []
        src_col = "source" if "source" in relationships_df.columns else "source_id"
        tgt_col = "target" if "target" in relationships_df.columns else "target_id"

        src_series = relationships_df[src_col].fillna("")
        tgt_series = relationships_df[tgt_col].fillna("")

        # ── 一跳 ──────────────────────────────────────────────────────────────
        mask1 = (src_series == entity_name) | (tgt_series == entity_name)
        direct_rels = relationships_df[mask1]
        result = direct_rels.head(max_rels).to_dict("records")

        if not two_hop or len(result) >= max_rels:
            return result

        # ── 二跳 ──────────────────────────────────────────────────────────────
        neighbor_names: set[str] = set()
        for _, row in direct_rels.iterrows():
            s, t = str(row.get(src_col, "")), str(row.get(tgt_col, ""))
            if s != entity_name:
                neighbor_names.add(s)
            if t != entity_name:
                neighbor_names.add(t)

        if neighbor_names:
            mask2 = (src_series.isin(neighbor_names)) | (tgt_series.isin(neighbor_names))
            two_hop_df = relationships_df[mask2 & ~mask1]
            # 只保留高权重，避免引入噪音
            try:
                two_hop_df = two_hop_df[
                    two_hop_df["weight"].fillna(0).astype(float) >= 5.0
                ]
            except Exception:
                pass
            extra = two_hop_df.head(max_rels - len(result)).to_dict("records")
            result.extend(extra)

        return result

    def _get_community(self, entity_id: str, community_reports_df) -> tuple[Optional[str], Optional[str]]:
        if community_reports_df is None or community_reports_df.empty:
            return None, None
        # 按 entity_ids 字段查找所属社区
        for _, row in community_reports_df.iterrows():
            entity_ids = row.get("entity_ids", []) or []
            if entity_id in entity_ids:
                return str(row.get("community", "")), str(row.get("full_content", ""))
        return None, None

    def _rerank_by_community(self, evidences: list[GraphEvidence]) -> list[GraphEvidence]:
        """
        专利公式：Score_comm = α·A + β·B
        A = 实体引用频次（用 related_relationships 数量近似）
        B = 社区重要性得分（此处简化为是否有社区摘要）
        """
        α = self._cfg.community_alpha
        β = self._cfg.community_beta

        for ev in evidences:
            A = len(ev.related_relationships)  # 引用频次近似
            B = 1.0 if ev.community_summary else 0.0
            ev._score = α * A + β * B  # type: ignore

        return sorted(evidences, key=lambda e: getattr(e, "_score", 0), reverse=True)


# ── 双路检索协调器 ────────────────────────────────────────────────────────────
class DualPathRetriever:


    def __init__(
        self,
        graph_artifacts: Optional[dict] = None,
        ablation_flags: Optional[dict] = None,
    ):
        flags = ablation_flags or {}
        self._disable_graph = flags.get("disable_graph", False)

        self._text_retriever = TextRetriever(ablation_flags=flags)
        self._graph_retriever = GraphRetriever(artifacts=graph_artifacts)
        self._source_index = self._build_source_index(graph_artifacts)

    def retrieve(
        self,
        normalized_query: str,
        extracted_entities: list[dict],
        keyword_terms: Optional[list[str]] = None,
    ) -> RetrievalResult:
        """双路检索，keyword_terms 用于图谱路精确预匹配。"""
        # 文本路
        text_evidences, triggered, sub_questions = self._text_retriever.retrieve(normalized_query)

        # 图谱路（消融：可禁用）
        if self._disable_graph:
            logger.info("[消融] 图谱路已禁用")
            graph_evidences = []
        else:
            graph_evidences = self._graph_retriever.retrieve(
                normalized_query, extracted_entities, keyword_terms=keyword_terms or []
            )

        # 溯源信息注入：为每条 TextEvidence 填充 source_title 和 page_info
        self._enrich_text_evidences(text_evidences)

        return RetrievalResult(
            query=normalized_query,
            normalized_query=normalized_query,
            triggered_decomposition=triggered,
            sub_questions=sub_questions,
            text_evidences=text_evidences,
            graph_evidences=graph_evidences,
        )

    @staticmethod
    def _build_source_index(graph_artifacts: Optional[dict]) -> dict:
        """
        构建溯源索引：doc_id → 文档标题（文件名）
        键名对应 graph_builder.load_artifacts() 的标准键：
          - documents          → "documents"（需在 graph_builder 中补充加载）
          - text_units         → "create_final_text_units"
        """
        index = {"doc_id_to_title": {}, "text_prefix_to_info": {}}
        if not graph_artifacts:
            return index

        # documents：graph_builder 里键名为 "documents"
        docs_df = graph_artifacts.get("documents")
        if docs_df is not None and not docs_df.empty:
            for _, row in docs_df.iterrows():
                doc_id = str(row.get("id", ""))
                title = str(row.get("title", ""))
                if doc_id and title:
                    index["doc_id_to_title"][doc_id] = title
                    index["doc_id_to_title"][doc_id[:16]] = title
            logger.info(f"溯源索引: 加载 {len(docs_df)} 个文档映射")

        # text_units：键名为 "create_final_text_units"
        tu_df = graph_artifacts.get("create_final_text_units")
        if tu_df is not None and not tu_df.empty:
            for _, row in tu_df.iterrows():
                text_prefix = str(row.get("text", ""))[:40]
                doc_ids = row.get("document_ids", []) or []
                readable_id = str(row.get("human_readable_id", ""))
                if text_prefix and doc_ids:
                    first_doc_id = str(doc_ids[0]) if doc_ids else ""
                    index["text_prefix_to_info"][text_prefix] = {
                        "doc_id": first_doc_id,
                        "readable_id": readable_id,
                    }
            logger.info(f"溯源索引: 加载 {len(tu_df)} 个文本块映射")

        return index

    def _enrich_text_evidences(self, evidences: list[TextEvidence]) -> None:
        """
        为 TextEvidence 填充 source_title 和 page_info。
        匹配逻辑（优先级从高到低）：
          1. doc_id 精确匹配 documents.id
          2. doc_id 前缀匹配
          3. 文本内容前40字匹配 text_units
          4. doc_id 本身就是文件名（已有可读信息）
        """
        doc_id_map = self._source_index.get("doc_id_to_title", {})
        text_prefix_map = self._source_index.get("text_prefix_to_info", {})

        for ev in evidences:
            title = None

            # 1. 精确匹配
            if ev.doc_id in doc_id_map:
                title = doc_id_map[ev.doc_id]

            # 2. 前缀匹配（doc_id 可能只存了前16位）
            if not title and ev.doc_id:
                prefix = ev.doc_id[:16]
                title = doc_id_map.get(prefix)

            # 3. 文本内容前缀匹配 text_units
            if not title and ev.text:
                text_key = ev.text[:40]
                info = text_prefix_map.get(text_key)
                if info:
                    full_doc_id = info.get("doc_id", "")
                    title = doc_id_map.get(full_doc_id) or doc_id_map.get(full_doc_id[:16])

            # 4. doc_id 本身看起来像文件名（不是纯哈希，含中文或.txt）
            if not title and ev.doc_id and (
                any('\u4e00' <= c <= '\u9fff' for c in ev.doc_id)
                or ev.doc_id.endswith('.txt')
                or len(ev.doc_id) < 32  # 短ID更可能是可读名
            ):
                title = ev.doc_id

            if title:
                # 去掉 .txt 后缀使显示更干净
                ev.source_title = title.replace(".txt", "").replace(".pdf", "").replace(".docx", "")

            # 填充页码信息
            if ev.start_page and ev.start_page > 0:
                if ev.end_page and ev.end_page != ev.start_page:
                    ev.page_info = f"第{ev.start_page}-{ev.end_page}页"
                else:
                    ev.page_info = f"第{ev.start_page}页"
