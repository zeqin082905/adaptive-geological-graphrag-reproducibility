"""
步骤12：答案生成器（RASCEF Prompt 优化版）
═══════════════════════════════════════════════════════════════════════════════
充分利用 GraphRAG 全部输出字段：
  - entities:          title / description / type / frequency / degree
  - relationships:     source / target / description / weight
  - community_reports: summary / full_content / findings / rank
  - text_units:        原始文本（溯源）

【RASCEF Prompt 设计框架】
  R - Role        角色定义：明确模型身份与专业背景
  A - Action      行动指令：明确任务目标与操作步骤
  S - Steps       步骤规范：分层推理顺序（图谱优先→文本补充→冲突消解→输出）
  C - Context     上下文：结构化双路证据包（图谱数据 + 文本段落）
  E - Examples    示例：引用标记格式示范，防止幻觉和格式漂移
  F - Format      格式约束：严格输出结构，包含结论/证据链/不确定性三段式

Master Prompt 设计原则：
  1. 图谱结构化知识优先（W_g=0.7），文本证据补充（W_t=0.3）
  2. findings 字段逐条展开，避免摘要信息丢失
  3. 高权重关系（weight > 5）优先呈现
  4. 核心实体（degree 高）标注重要性
  5. 强制引用格式，保证可溯源
  6. 专项 Prompt 补丁（大地构造/控矿构造/勘察单位/资源量）按问题类型动态注入
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from config.settings import cfg
from src.llm.client import llm

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# RASCEF Master Prompt 模板
# ══════════════════════════════════════════════════════════════════════════════
MASTER_PROMPT = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[R] ROLE · 角色定义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
你是一位拥有20年野外经验的地质矿产领域知识工程专家，专长于以下三类能力：
1. 精读地质勘查报告、矿产资源储量报告，提炼矿区"含矿层位—控矿构造—矿石类型"等关键要素；
2. 基于结构化知识图谱进行多跳推理，从实体→关系→社区描述逐层还原地质成因链；
3. 在图谱证据与文本证据冲突时，依据权威文档优先原则作出裁决，并给出可核验的溯源路径。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[A] ACTION · 行动指令
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
请完成以下任务：
针对【用户问题】，综合【知识图谱数据】（权重 W_g=0.7，图谱由经人工确认的权威地质资料构建）
与【原文文本段落】（权重 W_t=0.3，来自补充资料库），给出准确、专业、带溯源标记的结构化答案。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[S] STEPS · 推理步骤（请按此顺序进行内部推理，不要输出推理过程）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 · 问题解析
  - 识别问题中的核心查询对象（矿区名、地层名、构造名、矿种等）
  - 判断问题类型：矿体特征 / 控矿构造 / 大地构造 / 资源量 / 勘查单位 / 成因分析 / 其他

Step 2 · 图谱侧推理（W_g=0.7，优先）
  - 从"社区报告"的 findings 字段逐条扫描，找到与问题直接相关的关键发现
  - 在"实体"中定位核心实体，重点关注 degree 高（⭐核心实体）的节点
  - 在"关系"中沿着与核心实体相连的高权重边（weight≥5）进行多跳检索
  - 提取具体名称（构造名、地层名、矿种名），而非泛化描述

Step 3 · 文本侧补充（W_t=0.3）
  - 从文本段落中寻找与图谱结论一致的原文佐证
  - 若文本中出现图谱未覆盖的具体数值（储量、坐标、品位），予以补充

Step 4 · 冲突裁决
  - 若图谱与文本对同一事实产生矛盾：优先采信图谱结论，将文本异议记入"不确定性"部分
  - 若两路一致：合并输出，双重溯源

Step 5 · 生成输出
  - 按 [F] FORMAT 规定的三段式结构输出，每条关键陈述必须附加引用标记

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[C] CONTEXT · 双路证据上下文
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【用户问题】
{question}

【知识图谱数据】（来源：权威地质资料构建的 GraphRAG 图谱，W_g=0.7）
{graph_context}

【原文文本段落】（来源：补充资料向量库，W_t=0.3）
{text_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[E] EXAMPLES · 引用格式示例（严格遵照此格式，不得自创格式）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 正确示例：
  班台子铁矿矿体严格受班台子单斜构造控制，赋存于吴集组上段含铁岩系中。
  [数据: 图谱实体(Ent_301); 图谱关系(Rel_45); 社区报告(Comm_12)]

  文本段落亦明确记载："矿体赋存于向斜核部的吴集组上段含铁岩系中"。
  [数据: 原文段落(安徽省霍邱县班台子铁矿床详查地质报告, 第23页)]

❌ 错误示例（禁止以下行为）：
  - 引用时省略记录ID：[数据: 图谱实体]  ← 必须包含括号内的ID
  - 自行编造不在证据中的构造名或储量数字
  - 用"根据资料""综上所述"等套话代替具体引用
  - 将宏观构造单元（扬子准地台、华北地台）作为"控矿构造"的答案

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[F] FORMAT · 输出格式约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
强制规则：
1. 首句直接给出核心结论，不以"根据资料""综上"等套话开头
2. 每条关键陈述后附引用标记，格式为：[数据: 来源类型(记录ID)]
   来源类型仅允许：图谱实体 / 图谱关系 / 社区报告 / 原文段落
3. 图谱与文本证据矛盾时，图谱结论为主结论，文本差异记入"不确定性"段
4. 证据不足时，明确说明缺失信息类型，建议核查的文档类型，禁止臆造
5. 使用规范地质术语，数值必须带单位，构造名称完整书写
6. 【完整性要求】问题中每个独立的查询点都必须得到回答，不得遗漏；
   若证据不足以回答某子问题，明确写出"[该问题暂无充分证据]"

请按以下三段式结构输出：

## 结论
[一句话核心答案，直接点明查询对象的核心属性]

## 详细说明

### 图谱侧证据（W_g=0.7）
[基于知识图谱的推理链：社区发现 → 核心实体描述 → 关系路径，每条结论附引用]

### 文本侧证据（W_t=0.3）
[基于原文段落的补充支撑或佐证，注明来源文档与页码]

## 不确定性与建议验证路径
[若存在信息缺口或图谱/文本冲突，说明缺失信息类型与建议核查路径；若无则填写"证据充分，无需额外验证"]
"""


# ══════════════════════════════════════════════════════════════════════════════
# 问题类型专项 Prompt 补丁（RASCEF S-Steps 层动态注入）
# ══════════════════════════════════════════════════════════════════════════════

# 大地构造类：强调区分宏观地台与具体构造单元名
_PATCH_TECTONIC = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[S-专项补丁] 大地构造部位问题 · 推理约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本问题询问矿区所处的【具体大地构造部位名称】。

⚠️ 层次区分（务必遵守）：
  ❌ 宏观构造（不是答案）：扬子准地台、华北地台、特提斯构造域、华南褶皱系
     → 这类术语是板块/地台级，覆盖范围数百公里，不是矿区级别的构造部位
  ✅ 具体构造（正确答案）：XX背斜、XX向斜、XX穹窿、XX复背斜、XX单斜
     → 此类术语直接定位矿区所在的二级构造单元，文档中通常用"位于XX背斜核部/翼部"表述

Step 2 专项操作：
  - 在图谱关系中，优先检索含"位于""赋存于""处于""构造部位"等谓词的关系，找到以"背斜/向斜/穹窿/单斜"结尾的实体
  - 在文本段落中，检索含"大地构造"或"构造部位"附近的具体背斜/向斜名称
  - 若宏观构造单元与具体构造名同时出现，两者均输出，但具体构造名作为主答案
"""

# 控矿构造类：强调矿区内部直接控矿构造
_PATCH_ORE_STRUCTURE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[S-专项补丁] 控矿构造 / 矿区构造问题 · 推理约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本问题询问【直接控制矿体产出的矿区内部具体构造】。

⚠️ 层次区分（务必遵守）：
  ❌ 区域背景构造（不是答案）：距矿区较远的区域性大断裂、省级断裂带
  ✅ 直接控矿构造（正确答案）：矿区范围内以"控矿""赋矿""矿体产于"等关系连接矿区的具体构造名
     → 典型表述：F1断层、XX单斜、XX背斜核部、向斜轴部

Step 2 专项操作：
  - 在图谱关系中，重点检索 weight≥5 且谓词含"控矿""赋矿""产于""受...控制"的关系
  - 在文本段落中，查找"矿体严格受...控制""产于...核部""赋存于...地层"等句式
  - 同时报告褶皱类（主体）和断裂类（次要）控矿构造，分类输出
"""

# 勘察单位类：区分主责单位与历史辅助单位
_PATCH_SURVEY_UNIT = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[S-专项补丁] 勘察单位问题 · 推理约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本问题询问【负责完成主体勘查工作并持有探矿权的责任单位】。

⚠️ 单位区分（务必遵守）：
  ✅ 目标单位：持有探矿权证、承担详查/勘探任务、提交地质报告的主责单位
  ❌ 排除单位：
    - 仅承担区域地质调查或化探普查的单位
    - 探矿权转让前的历史勘查单位（除非明确问历史沿革）
    - 与矿区有业务往来但非勘查主体的单位

Step 2 专项操作：
  - 在图谱关系中，优先检索含"详查""勘探""探矿权持有""提交报告""负责勘查"等谓词的关系
  - 识别实体类型为"勘查单位"且与目标矿区有直接勘查关系的节点
  - 文本段落中查找"工作单位""资质单位""持证单位"等表述
"""

# 资源量类：优先寻找官方数字与矿种分类
_PATCH_RESOURCE = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[S-专项补丁] 资源储量问题 · 推理约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本问题询问矿区的【探明资源储量数字】。

Step 2&3 专项操作：
  - 优先从文本段落中寻找明确的储量数字（万吨/千吨/万克拉等）及其单位
  - 图谱实体描述中若含储量字段，也予以引用
  - 若文档中同时存在多个储量数字（总量/氧化矿/硫化矿/表内/表外），全部列出并标注类别
  - 报告提交年份不同可能导致数字差异，需注明数据来源版本

⚠️ 关于数字歧义：同一矿区可能有"总资源量"和"某类别资源量"两个不同数字同时出现，
   两者均视为有效答案，输出时分类标注。
"""


# 完整性补丁：问题含多个子问题时强制逐条回答
_PATCH_COMPLETENESS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[S-专项补丁] 多维度问题 · 完整性约束
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
本问题包含多个独立的查询维度（如"XX在哪里？采用了哪些方法？"）。

⚠️ 强制要求（务必遵守）：
  - 识别问题中所有独立的查询点（通常以"？""哪些""哪几个""是什么""如何"为分隔）
  - 在【详细说明】中为每个查询点单独列一段，不得合并或遗漏任何一个
  - 若某个查询点在证据中找不到对应信息，明确写"[该查询点暂无证据]"，禁止沉默跳过

Step 5 专项操作：
  - 在输出前自检：数一数问题有几个"？"或几个并列问法，确保【详细说明】段落数与之对应
"""


def _detect_question_type(question: str) -> str:
    """
    根据问题文本判断问题类型，返回对应的专项 Prompt 补丁。
    补丁将动态注入到 MASTER_PROMPT 的 [S] STEPS 段之后、[C] CONTEXT 段之前。
    多个补丁可叠加。
    """
    q = question
    patches = []

    # 大地构造（优先级最高，防止被控矿构造规则误判）
    tectonic_kws = [
        "大地构造", "构造部位", "构造单元", "构造位置", "构造背景",
        "哪个大地", "所处构造", "区域构造",
    ]
    if any(kw in q for kw in tectonic_kws):
        patches.append(_PATCH_TECTONIC)

    # 勘察单位
    survey_kws = [
        "勘察", "勘查", "勘探", "哪个单位", "哪家单位", "哪个队",
        "地质队", "地质大队", "负责勘", "承担了",
    ]
    if any(kw in q for kw in survey_kws):
        patches.append(_PATCH_SURVEY_UNIT)

    # 控矿构造（排在大地构造之后，防止误判）
    ore_struct_kws = [
        "控矿构造", "矿区构造", "区域内有哪些构造", "包含哪些地质构造",
        "发育有哪些构造", "控矿", "赋矿构造",
    ]
    if any(kw in q for kw in ore_struct_kws):
        patches.append(_PATCH_ORE_STRUCTURE)

    # 资源量
    resource_kws = [
        "资源量", "储量", "资源储量", "矿石量", "多少吨", "探明",
    ]
    if any(kw in q for kw in resource_kws):
        patches.append(_PATCH_RESOURCE)

    # 多维度问题完整性补丁：问题含2个及以上问号，或含"哪些...哪些"等并列结构
    question_marks = q.count("？") + q.count("?")
    has_parallel = any(kw in q for kw in ["哪些", "哪几个", "分别", "各自", "各是"])
    if question_marks >= 2 or (question_marks >= 1 and has_parallel):
        patches.append(_PATCH_COMPLETENESS)

    return "\n\n".join(patches)  # 多补丁叠加


# ══════════════════════════════════════════════════════════════════════════════
# 上下文构建器（ContextPackBuilder）
# ══════════════════════════════════════════════════════════════════════════════
class ContextPackBuilder:
    """
    将检索结果组装成结构化的 [C] CONTEXT 证据包。
    充分利用 GraphRAG 全部字段：findings / full_content / weight / frequency / degree
    """

    # 关系权重阈值：低于此值的关系视为弱关联，降低优先级
    REL_WEIGHT_THRESHOLD = 3.0
    # 最多展示的关系数量（避免 prompt 过长）
    MAX_RELATIONS_PER_ENTITY = 8
    # 最多展示的 findings 条数
    MAX_FINDINGS = 10
    # 文本段落最大字符数
    MAX_TEXT_CHARS = 600

    def build_graph_context(self, graph_evidences: list) -> str:
        """
        构建图谱侧上下文，按社区→实体→关系层级组织。
        充分利用：findings / full_content / weight / frequency / degree
        """
        if not graph_evidences:
            return "（无图谱证据——图谱未加载或未命中实体）"

        sections = []

        # ── 按社区分组，优先展示高 rank 社区 ────────────────────────────────
        seen_communities: set = set()
        for ev in graph_evidences:
            if ev.community_id and ev.community_id not in seen_communities:
                seen_communities.add(ev.community_id)
                community_section = self._format_community(ev)
                if community_section:
                    sections.append(community_section)

        # ── 实体+关系部分 ─────────────────────────────────────────────────────
        seen_entities: set = set()
        for ev in graph_evidences:
            if ev.entity_id not in seen_entities:
                seen_entities.add(ev.entity_id)
                entity_section = self._format_entity(ev)
                if entity_section:
                    sections.append(entity_section)

        return "\n\n".join(sections) if sections else "（无图谱证据）"

    def _format_community(self, ev) -> str:
        """格式化社区报告，重点展示 findings 字段（最有价值的结构化发现）。"""
        if not ev.community_summary and not ev.community_id:
            return ""

        lines = [f"### 📊 社区报告 [community_id={ev.community_id}]"]

        if ev.community_summary:
            lines.append(f"**摘要**: {ev.community_summary}")

        # findings 字段（结构化发现，对 LLM 推理最有价值）
        if hasattr(ev, "community_findings") and ev.community_findings:
            lines.append("**关键发现**（逐条参考）:")
            findings = ev.community_findings
            if isinstance(findings, str):
                try:
                    findings = json.loads(findings)
                except Exception:
                    pass
            if isinstance(findings, list):
                for i, f in enumerate(findings[: self.MAX_FINDINGS]):
                    if isinstance(f, dict):
                        summary = f.get("summary", f.get("explanation", str(f)))
                    else:
                        summary = str(f)
                    lines.append(f"  {i + 1}. {summary}")
            else:
                lines.append(f"  {str(findings)[:400]}")

        return "\n".join(lines)

    def _format_entity(self, ev) -> str:
        """格式化实体信息，包含类型、描述、重要性指标与关系列表。"""
        lines = []

        # 实体头部：名称 + 类型 + 重要性标记
        importance_hint = ""
        if hasattr(ev, "entity_degree") and ev.entity_degree:
            if ev.entity_degree > 20:
                importance_hint = " ⭐核心实体"
            elif ev.entity_degree > 10:
                importance_hint = " ★重要实体"

        lines.append(
            f"#### 🔷 实体: {ev.entity_name}"
            f"（类型: {ev.entity_type}{importance_hint}）"
            f" [entity_id={ev.entity_id[:8]}]"
        )
        if ev.entity_description:
            lines.append(f"**描述**: {ev.entity_description}")

        # 关系列表（按权重降序，优先展示强关联）
        if ev.related_relationships:
            rels = ev.related_relationships
            try:
                rels = sorted(rels, key=lambda r: float(r.get("weight", 0)), reverse=True)
            except Exception:
                pass

            strong_rels = [r for r in rels if float(r.get("weight", 0)) >= self.REL_WEIGHT_THRESHOLD]
            weak_rels = [r for r in rels if float(r.get("weight", 0)) < self.REL_WEIGHT_THRESHOLD]

            display_rels = strong_rels[: self.MAX_RELATIONS_PER_ENTITY]
            if not display_rels:
                display_rels = weak_rels[:3]  # 无强关系时展示3条弱关系

            if display_rels:
                lines.append("**相关关系**（weight≥3 优先展示）:")
                for r in display_rels:
                    src = r.get("source", "")
                    tgt = r.get("target", "")
                    desc = r.get("description", "")
                    weight = r.get("weight", "")
                    rid = str(r.get("id", ""))[:8]
                    weight_str = f" (强度:{weight})" if weight else ""
                    lines.append(
                        f"  - {src} → {tgt}{weight_str}: {desc}"
                        f" [rel_id={rid}]"
                    )
                if len(strong_rels) > self.MAX_RELATIONS_PER_ENTITY:
                    lines.append(
                        f"  ... 还有 {len(strong_rels) - self.MAX_RELATIONS_PER_ENTITY} 条强关系未展示"
                    )

        return "\n".join(lines)

    def build_text_context(self, text_evidences: list, sub_questions: list = None) -> str:
        """构建文本侧上下文，按子问题分组展示。"""
        if not text_evidences:
            return "（无文本证据——向量库可能为空或未检索到相关段落）"

        lines = []

        if sub_questions:
            # 按子问题分组展示
            for sq in sub_questions:
                sq_id = sq.get("id", "")
                sq_query = sq.get("primary_query", sq.get("core_intent", ""))
                sq_evidences = [
                    e for e in text_evidences
                    if getattr(e, "sub_question_id", None) == sq_id
                ]
                if sq_evidences:
                    lines.append(f"**子问题 {sq_id}**: {sq_query}")
                    for e in sq_evidences[:3]:
                        lines.append(self._format_text_evidence(e))

            # 未匹配到子问题的证据（初步检索直通结果）
            unmatched = [e for e in text_evidences if not getattr(e, "sub_question_id", None)]
            for e in unmatched[:3]:
                lines.append(self._format_text_evidence(e))
        else:
            # 无子问题分解，直接按相似度顺序展示
            for e in text_evidences[:8]:
                lines.append(self._format_text_evidence(e))

        return "\n\n".join(lines) if lines else "（无文本证据）"

    def _format_text_evidence(self, e) -> str:
        text = getattr(e, "text", "")[: self.MAX_TEXT_CHARS]
        sim = getattr(e, "similarity", 0)
        rerank = getattr(e, "rerank_score", None)

        # 优先使用溯源后的 source_title，其次用 doc_id
        source_title = getattr(e, "source_title", None)
        doc_id = getattr(e, "doc_id", "unknown")
        display_source = source_title if source_title else doc_id

        # 页码信息
        page_info = getattr(e, "page_info", None)
        start_page = getattr(e, "start_page", 0)
        end_page = getattr(e, "end_page", 0)
        if not page_info and start_page and start_page > 0:
            page_info = (
                f"第{start_page}-{end_page}页" if end_page != start_page else f"第{start_page}页"
            )

        source_label = f"{display_source}（{page_info}）" if page_info else display_source
        score_label = f"相似度={sim:.3f}"
        if rerank is not None:
            score_label += f" / Rerank={rerank:.3f}"

        if len(getattr(e, "text", "")) > self.MAX_TEXT_CHARS:
            text += "..."

        return (
            f"📄 【来源: {source_label}】{score_label}\n"
            f"{text}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 答案生成器（AnswerGenerator）
# ══════════════════════════════════════════════════════════════════════════════
class AnswerGenerator:
    """
    步骤12：融合双路证据，调用 LLM 生成最终答案。

    Prompt 构建流程（RASCEF）：
      1. 构建图谱侧 & 文本侧上下文 → 填入 [C] CONTEXT
      2. 检测问题类型，选取专项补丁 → 注入 [S] STEPS 之后
      3. 调用 LLM，返回严格三段式结构化答案
    """

    def __init__(self):
        self._builder = ContextPackBuilder()

    def generate(self, question: str, retrieval_result) -> str:
        """
        生成答案主入口。
        retrieval_result: DualPathRetriever.retrieve() 的返回值
        """
        # 补充社区 findings 字段到图谱证据
        self._enrich_community_findings(retrieval_result)

        # 构建双路上下文（填充 [C] CONTEXT）
        graph_ctx = self._builder.build_graph_context(retrieval_result.graph_evidences)
        text_ctx = self._builder.build_text_context(
            retrieval_result.text_evidences,
            sub_questions=retrieval_result.sub_questions,
        )

        # 检测问题类型，获取专项补丁
        question_patch = _detect_question_type(question)

        # 组装完整 RASCEF Prompt
        base_prompt = MASTER_PROMPT.format(
            question=question,
            graph_context=graph_ctx,
            text_context=text_ctx,
        )

        if question_patch:
            # 将专项补丁插入到 [C] CONTEXT 段之前（紧接 [S] STEPS 之后）
            insert_marker = "[C] CONTEXT · 双路证据上下文"
            if insert_marker in base_prompt:
                prompt = base_prompt.replace(
                    insert_marker,
                    question_patch.strip() + "\n\n" + insert_marker,
                )
            else:
                prompt = base_prompt + "\n" + question_patch
            logger.debug(f"专项补丁已注入: {question_patch[:60].strip()}...")
        else:
            prompt = base_prompt

        logger.debug(f"最终 Prompt 长度: {len(prompt)} 字符")

        # 调用 LLM
        try:
            answer = llm.generate(
                prompt,
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
            )
            return answer
        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return (
                f"[生成失败] {e}\n\n"
                f"已检索到的图谱证据摘要:\n{graph_ctx[:500]}"
            )

    def _enrich_community_findings(self, retrieval_result) -> None:
        """
        从图谱 artifacts 中读取 findings 字段，补充到 GraphEvidence 对象。
        findings 是 community_reports 里最有价值但目前未被充分利用的字段。
        """
        if not retrieval_result.graph_evidences:
            return

        community_reports_df = getattr(retrieval_result, "community_reports_df", None)
        if community_reports_df is None:
            return

        # 建立 community_id → findings/full_content/rank 的映射
        community_map: dict = {}
        for _, row in community_reports_df.iterrows():
            cid = str(row.get("community", row.get("id", "")))
            community_map[cid] = {
                "findings": row.get("findings", ""),
                "full_content": row.get("full_content", ""),
                "rank": row.get("rank", 0),
            }

        # 逐个 GraphEvidence 补充 findings
        for ev in retrieval_result.graph_evidences:
            if ev.community_id and ev.community_id in community_map:
                data = community_map[ev.community_id]
                ev.community_findings = data["findings"]
                ev.community_rank = data["rank"]


# 全局单例
answer_generator = AnswerGenerator()
