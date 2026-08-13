"""
步骤5-6：GraphRAG 知识图谱构建
═══════════════════════════════════════
封装 Microsoft GraphRAG（graphrag 库）并注入地质领域专用配置。
流程：
  步骤5 → Auto Prompt Tuning：自动生成地质实体/关系抽取模板
  步骤6 → GraphRAG Indexing Pipeline：批量抽取 + Leiden 社区聚类 + 社区摘要向量化
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from config.settings import cfg, GRAPHRAG_DIR, PROMPTS_DIR
from src.ingestion.loader import ParsedDocument

logger = logging.getLogger(__name__)


# ── GraphRAG 配置生成器 ───────────────────────────────────────────────────────
class GraphRAGConfigWriter:
    """
    生成 Microsoft GraphRAG 所需的 settings.yaml 配置文件。
    将 Ollama + Qwen2.5 接入 GraphRAG 的 LLM 与 Embedding 通道。
    """

    SETTINGS_TEMPLATE = """\
# GraphRAG Settings - 地质资料智能问答系统
# 自动生成，请勿手动修改（通过 GraphRAGConfigWriter 调整）

encoding_model: cl100k_base
skip_workflows: []

llm:
  api_key: ollama
  type: openai_chat
  model: {llm_model}
  model_supports_json: true
  api_base: {llm_api_base}
  max_tokens: 4096
  temperature: 0.0
  request_timeout: 120.0
  max_retries: 3

parallelization:
  stagger: 0.3
  num_threads: 4

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ollama
    type: openai_embedding
    model: {embedding_model}
    api_base: {llm_api_base}
    max_tokens: 8192
    request_timeout: 120.0

chunks:
  size: 500
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: input

cache:
  type: file
  base_dir: cache

storage:
  type: file
  base_dir: output

reporting:
  type: file
  base_dir: logs

entity_extraction:
  prompt: prompts/entity_extraction.txt
  entity_types:
    - 矿区名称
    - 行政区划
    - 经纬度
    - 矿产资源
    - 地层
    - 构造
    - 岩石
    - 勘查单位
    - 矿物成分
    - 化学成分
    - 矿产资源量
    - 勘查方法
    - 矿权类型
  max_gleanings: 2

summarize_descriptions:
  prompt: prompts/summarize_descriptions.txt
  max_length: 500

claim_extraction:
  enabled: false

community_reports:
  prompt: prompts/community_report.txt
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: true
  num_walks: 10
  walk_length: 40
  window_size: 2
  iterations: 3
  random_seed: 42

umap:
  enabled: false

snapshots:
  graphml: true
  raw_entities: true
  top_level_nodes: true

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_entities: 10
  top_k_relationships: 10
  max_tokens: 12000

global_search:
  max_tokens: 12000
  data_max_tokens: 12000
  map_max_tokens: 1000
  reduce_max_tokens: 2000
  concurrency: 4
"""

    def write(self, workspace: Path):
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "input").mkdir(exist_ok=True)
        (workspace / "prompts").mkdir(exist_ok=True)

        settings_text = self.SETTINGS_TEMPLATE.format(
            llm_model=cfg.graphrag.llm_model,
            llm_api_base=cfg.graphrag.llm_api_base,
            embedding_model=cfg.graphrag.embedding_model,
        )
        (workspace / "settings.yaml").write_text(settings_text, encoding="utf-8")
        logger.info(f"GraphRAG 配置已写入: {workspace / 'settings.yaml'}")


# ── 地质专用 Prompt 模板管理 ──────────────────────────────────────────────────
class GeoPromptManager:
    """管理地质领域的实体抽取、关系抽取、社区报告 Prompt 模板。"""

    ENTITY_EXTRACTION_PROMPT = """\
-目标-
给定一份地质资料文本，识别文本中所有符合指定类型的实体，以及这些实体之间的所有关系。

-步骤-
1. 识别以下类型的实体（entity_types）：{entity_types}
   对每个识别出的实体，提取以下信息：
   - entity_name：实体名称，标准化为地质术语
   - entity_type：实体类型，必须是 entity_types 中的一个
   - entity_description：实体的综合性文本描述，包含属性、特征及其在地质报告中的意义

   输出格式（每条记录以 <REC_SEP> 分隔）：
   ("entity", <entity_name>, <entity_type>, <entity_description>)

2. 识别所有**明确相关**的实体对，以及它们之间的关系：
   - source_entity：源实体名称
   - target_entity：目标实体名称
   - relationship_description：关系的详细描述
   - relationship_strength：1-10 的整数，表示关系强度

   输出格式（每条记录以 <REC_SEP> 分隔）：
   ("relationship", <source_entity>, <target_entity>, <relationship_description>, <relationship_strength>)

3. 将步骤1和步骤2的所有输出合并，以 <REC_SEP> 分隔，结尾加上 <COMPLETE>

-示例-
文本：河南省栾川县福家村铅矿矿区位于黑沟-栾川断裂带与马超营断裂带之间，
赋矿层位为太古宇太华群地层。

输出：
("entity", "福家村铅矿矿区", "矿区名称", "位于河南省栾川县的铅矿矿区")
<REC_SEP>
("entity", "黑沟-栾川断裂带", "构造", "控制矿区的断裂构造")
<REC_SEP>
("entity", "太华群地层", "地层", "矿区主要赋矿地层，属太古宇")
<REC_SEP>
("relationship", "太华群地层", "福家村铅矿矿区", "太华群地层是矿区主要赋矿层位", 9)
<REC_SEP>
("relationship", "黑沟-栾川断裂带", "福家村铅矿矿区", "矿区位于该断裂带附近", 7)
<COMPLETE>

-真实文本-
entity_types: {entity_types}
text: {input_text}

输出：
"""

    COMMUNITY_REPORT_PROMPT = """\
你是一个地质领域专家，负责为知识图谱社区生成综合性描述报告。

给定以下地质知识图谱社区的实体和关系信息：
{input_text}

请生成一份结构化的社区描述报告，包括：
1. 核心地质实体及其属性
2. 实体间的主要关系
3. 该社区代表的地质意义或成矿规律（若可推断）

输出格式为 JSON：
{{
  "title": "社区标题（简洁概括）",
  "summary": "综合摘要（200字以内）",
  "key_entities": ["实体1", "实体2", ...],
  "key_relationships": ["关系描述1", ...],
  "geological_significance": "地质意义说明"
}}
"""

    def write_to_workspace(self, workspace: Path):
        prompt_dir = workspace / "prompts"
        prompt_dir.mkdir(exist_ok=True)

        (prompt_dir / "entity_extraction.txt").write_text(
            self.ENTITY_EXTRACTION_PROMPT, encoding="utf-8"
        )
        (prompt_dir / "community_report.txt").write_text(
            self.COMMUNITY_REPORT_PROMPT, encoding="utf-8"
        )
        logger.info("地质专用 Prompt 模板已写入 workspace/prompts/")


# ── GraphRAG 构建流水线 ───────────────────────────────────────────────────────
class GraphRAGBuilder:
    """
    步骤5-6：GraphRAG 知识图谱构建的高级封装。

    使用方式：
        builder = GraphRAGBuilder()
        builder.prepare_corpus(corpus1_docs)   # 将文档写入 GraphRAG input/
        builder.run_prompt_tuning()            # 步骤5：Auto Prompt Tuning
        builder.run_indexing()                 # 步骤6：构建图谱
    """

    def __init__(self, workspace: Path = GRAPHRAG_DIR):
        self.workspace = workspace
        self._config_writer = GraphRAGConfigWriter()
        self._prompt_manager = GeoPromptManager()

    def setup_workspace(self):
        """初始化 GraphRAG 工作区（首次运行必须调用）。"""
        self._config_writer.write(self.workspace)
        self._prompt_manager.write_to_workspace(self.workspace)
        logger.info(f"GraphRAG 工作区初始化完成: {self.workspace}")

    def prepare_corpus(self, docs: list[ParsedDocument]):
        """将第一文档库写入 GraphRAG 的 input/ 目录（纯文本格式）。"""
        input_dir = self.workspace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # 清空旧文件
        for f in input_dir.glob("*.txt"):
            f.unlink()

        for doc in docs:
            txt_path = input_dir / f"{doc.doc_id}.txt"
            txt_path.write_text(doc.full_text, encoding="utf-8")

        logger.info(f"已将 {len(docs)} 份文档写入 GraphRAG input/")

    def run_prompt_tuning(self, limit: int = 15):
        """
        步骤5：执行 Auto Prompt Tuning。
        从 input/ 中采样文本，自动优化实体抽取 Prompt。
        """
        cmd = [
            sys.executable, "-m", "graphrag", "prompt-tune",
            "--root", str(self.workspace),
            "--config", str(self.workspace / "settings.yaml"),
            "--limit", str(limit),
            "--language", "Chinese",
            "--output", str(self.workspace / "prompts"),
        ]
        logger.info(f"执行 Auto Prompt Tuning: {' '.join(cmd)}")
        self._run_cmd(cmd)

    def run_indexing(self):
        """步骤6：执行 GraphRAG 完整索引流水线（实体抽取 + Leiden + 社区摘要）。"""
        cmd = [
            sys.executable, "-m", "graphrag", "index",
            "--root", str(self.workspace),
            "--config", str(self.workspace / "settings.yaml"),
        ]
        logger.info("执行 GraphRAG 索引构建（此步骤耗时较长）...")
        self._run_cmd(cmd)
        logger.info("GraphRAG 索引构建完成！")

    def load_artifacts(self, output_dir: Optional[Path] = None) -> dict:
        """
        加载 GraphRAG 输出的 Parquet 文件，返回统一格式的数据字典。

        自动兼容两种命名风格：
          新版（您的文件）：entities / relationships / communities / community_reports
          旧版：create_final_entities / create_final_relationships / ...

        output_dir 优先级：
          1. 方法参数传入
          2. cfg.graphrag.external_output_dir（config/settings.py 中配置）
          3. 默认：self.workspace / "output"
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("请安装: pip install pandas pyarrow")

        # 确定输出目录
        if output_dir is None:
            output_dir = cfg.graphrag.external_output_dir or (self.workspace / "output")
        output_dir = Path(output_dir)

        if not output_dir.exists():
            raise FileNotFoundError(f"GraphRAG 输出目录不存在: {output_dir}")

        logger.info(f"从以下目录加载 GraphRAG artifacts: {output_dir}")

        # 两种风格的文件名映射
        # key = 内部标准键名，value = [新版文件名, 旧版文件名前缀]
        FILE_CANDIDATES = {
            "create_final_entities":          ["entities",          "create_final_entities"],
            "create_final_relationships":     ["relationships",     "create_final_relationships"],
            "create_final_communities":       ["communities",       "create_final_communities"],
            "create_final_community_reports": ["community_reports", "create_final_community_reports"],
            # text_units 用于溯源，可选
            "create_final_text_units":        ["text_units",        "create_final_text_units"],
            "documents": ["documents", "create_final_documents"],
        }

        artifacts = {}
        for standard_key, candidates in FILE_CANDIDATES.items():
            df = self._try_load_parquet(output_dir, candidates, pd)
            if df is not None:
                artifacts[standard_key] = df
                logger.info(f"✅ 加载 {standard_key}: {len(df)} 条记录，列: {list(df.columns)}")
            else:
                logger.warning(f"⚠️  未找到 {standard_key}（候选文件名: {candidates}）")

        if not artifacts:
            raise RuntimeError(
                f"在 {output_dir} 中未找到任何 GraphRAG parquet 文件。\n"
                f"请确认目录中存在以下任一文件：entities.parquet 或 create_final_entities.parquet"
            )

        # 打印列名概览，方便调试
        self._print_schema(artifacts)
        return artifacts

    @staticmethod
    def _try_load_parquet(
        output_dir: Path,
        candidates: list[str],
        pd,
    ):
        """按候选文件名列表依次尝试加载 parquet，找到第一个存在的返回 DataFrame。"""
        for name in candidates:
            # 精确匹配
            exact = output_dir / f"{name}.parquet"
            if exact.exists():
                return pd.read_parquet(exact)
            # 模糊匹配（应对带时间戳后缀的文件名）
            matches = list(output_dir.glob(f"{name}*.parquet"))
            if matches:
                return pd.concat([pd.read_parquet(f) for f in matches], ignore_index=True)
        return None

    @staticmethod
    def _print_schema(artifacts: dict):
        """打印各 DataFrame 的列名，帮助快速了解字段结构。"""
        logger.info("── GraphRAG artifacts 字段概览 ──")
        for key, df in artifacts.items():
            logger.info(f"  {key}: {list(df.columns)}")

    @staticmethod
    def _run_cmd(cmd: list[str]):
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"命令执行失败（退出码 {result.returncode}）")
