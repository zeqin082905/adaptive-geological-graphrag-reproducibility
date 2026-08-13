"""
步骤3-4：文本分块器（含跨页语义桥接）
═══════════════════════════════════════
本模块实现专利核心创新点：
  - 步骤3：固定长度 + 重叠窗口分块（L=500 Token，T=100 Token）
  - 步骤4：跨页语义桥接检测与桥接文本块构建

桥接触发规则（专利权利要求5）：
  Rule-A：句末未闭合标点
  Rule-B：括号/引号未闭合
  Rule-C：列表/条目起始标记未闭合
  Rule-D：标题/图表跨页依附
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config.settings import cfg, ChunkConfig
from src.ingestion.loader import ParsedDocument, PageText

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────
@dataclass
class TextChunk:
    """统一的文本块表示，携带完整溯源信息。"""
    chunk_id: str               # 唯一标识符（哈希）
    doc_id: str                 # 来源文档
    text: str                   # 块文本内容
    start_page: int             # 起始页码
    end_page: int               # 结束页码（桥接块跨页时 != start_page）
    chunk_index: int            # 块在文档内的顺序编号
    is_bridge: bool = False     # 是否为跨页桥接块
    triggered_rules: list[str] = field(default_factory=list)  # 触发的桥接规则
    offset_start: int = 0       # 块内字符偏移（起始）
    offset_end: int = 0         # 块内字符偏移（结束）
    embedding: Optional[list[float]] = None  # 向量（向量化后填入）

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "chunk_index": self.chunk_index,
            "is_bridge": self.is_bridge,
            "triggered_rules": self.triggered_rules,
        }


# ── 桥接触发规则集 ────────────────────────────────────────────────────────────
class BridgeTriggerRules:
    """
    专利权利要求5 中定义的四条桥接触发判别规则。
    每条规则返回 (triggered: bool, rule_name: str)。
    """

    # Rule-A：句末闭合标点集合
    SENTENCE_CLOSE_CHARS = set("。！？；．!?;")

    # Rule-C：列表/条目起始标记
    LIST_START_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:\d+[\.、。]|[①②③④⑤⑥⑦⑧⑨⑩]|[a-zA-Z][\.、]|[-•·▪])\s+"
    )
    LIST_END_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:综上|总结|以上|end|END|合计|小结)"
    )

    # Rule-D：标题/图表/跨页提示
    HEADING_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:第[一二三四五六七八九十百\d]+[章节条款]|[一二三四五六七八九十]+、|\d+\.\d+)"
    )
    TABLE_FIG_PATTERN = re.compile(
        r"(?:续表|续图|接上表|接上图|Table\s+cont|Figure\s+cont)", re.IGNORECASE
    )
    NEXT_PAGE_CONTINUATION = re.compile(
        r"^(?:接上|续|承上|如下所示|详见下|下面|以下)",
    )

    @classmethod
    def check_all(cls, tail: str, head: str) -> list[str]:
        """
        对尾部文本 T_tail 和头部文本 T_head 执行所有规则检测。
        返回触发的规则名称列表（空列表表示无需桥接）。
        """
        triggered = []
        if cls._rule_a(tail):
            triggered.append("Rule-A:句末未闭合")
        if cls._rule_b(tail):
            triggered.append("Rule-B:括号引号未闭合")
        if cls._rule_c(tail):
            triggered.append("Rule-C:列表未闭合")
        if cls._rule_d(tail, head):
            triggered.append("Rule-D:标题图表跨页")
        return triggered

    @classmethod
    def _rule_a(cls, tail: str) -> bool:
        """尾部最后一个非空白字符不是句末闭合标点。"""
        stripped = tail.rstrip()
        if not stripped:
            return False
        return stripped[-1] not in cls.SENTENCE_CLOSE_CHARS

    @classmethod
    def _rule_b(cls, tail: str) -> bool:
        """括号或引号开闭数量不匹配。"""
        pairs = [("（", "）"), ("(", ")"), ("「", "」"), ("【", "】"),
                 ("《", "》"), (""", """), ("'", "'")]
        for open_c, close_c in pairs:
            if tail.count(open_c) != tail.count(close_c):
                return True
        return False

    @classmethod
    def _rule_c(cls, tail: str) -> bool:
        """出现列表起始标记但未出现终止标记。"""
        has_start = bool(cls.LIST_START_PATTERN.search(tail))
        has_end = bool(cls.LIST_END_PATTERN.search(tail))
        return has_start and not has_end

    @classmethod
    def _rule_d(cls, tail: str, head: str) -> bool:
        """标题/图表依附规则：尾部有续图/续表提示，或头部有明显承接表达。"""
        tail_trigger = bool(
            cls.HEADING_PATTERN.search(tail[-200:]) or
            cls.TABLE_FIG_PATTERN.search(tail)
        )
        head_trigger = bool(cls.NEXT_PAGE_CONTINUATION.match(head.lstrip()[:100]))
        return tail_trigger or head_trigger


# ── 主分块器 ──────────────────────────────────────────────────────────────────
class GeoChunker:
    """
    地质资料文本分块器。
    先做固定窗口分块（步骤3），再做跨页桥接补充（步骤4）。
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self._cfg = config or cfg.chunk

    # ── 公开接口 ──────────────────────────────────────────────────────────────
    def chunk_document(self, doc: ParsedDocument) -> list[TextChunk]:
        """对单份文档进行分块，返回普通块 + 桥接块的合并列表。"""
        regular_chunks = self._regular_chunking(doc)
        bridge_chunks = self._bridge_chunking(doc)

        all_chunks = regular_chunks + bridge_chunks
        # 重新排序：按页码，桥接块插入对应页之后
        all_chunks.sort(key=lambda c: (c.start_page, c.is_bridge, c.chunk_index))

        logger.info(
            f"[{doc.doc_id}] 普通块: {len(regular_chunks)}, "
            f"桥接块: {len(bridge_chunks)}, 共: {len(all_chunks)}"
        )
        return all_chunks

    def chunk_corpus(self, docs: list[ParsedDocument]) -> list[TextChunk]:
        """对整个文档库进行分块。"""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    # ── 步骤3：固定长度 + 重叠窗口分块 ──────────────────────────────────────
    def _regular_chunking(self, doc: ParsedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in doc.pages:
            text = page.cleaned_text
            if not text.strip():
                continue

            # 按 Token 近似（中文约 1 字 ≈ 1 Token，英文约 0.75 Token）
            # 此处使用字符数作为近似，生产环境可替换为 tiktoken/transformers tokenizer
            L = self._token_to_char(self._cfg.chunk_size)
            T = self._token_to_char(self._cfg.overlap)

            start = 0
            while start < len(text):
                end = min(start + L, len(text))
                chunk_text = text[start:end]
                if chunk_text.strip():
                    chunks.append(TextChunk(
                        chunk_id=self._make_id(doc.doc_id, page.page_num, start),
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        start_page=page.page_num,
                        end_page=page.page_num,
                        chunk_index=chunk_index,
                        is_bridge=False,
                        offset_start=start,
                        offset_end=end,
                    ))
                    chunk_index += 1

                if end >= len(text):
                    break
                start = end - T  # 重叠滑动

        return chunks

    # ── 步骤4：跨页语义桥接 ───────────────────────────────────────────────────
    def _bridge_chunking(self, doc: ParsedDocument) -> list[TextChunk]:
        """
        遍历相邻页面对，检测是否触发桥接规则。
        若触发，生成独立桥接文本块 D = T_tail || T_head。
        """
        bridge_chunks: list[TextChunk] = []
        pages = [p for p in doc.pages if p.cleaned_text.strip()]

        B_prev_chars = self._token_to_char(self._cfg.bridge_prev)
        B_next_chars = self._token_to_char(self._cfg.bridge_next)

        for i in range(len(pages) - 1):
            curr_page = pages[i]
            next_page = pages[i + 1]

            T_tail = curr_page.cleaned_text[-B_prev_chars:]
            T_head = next_page.cleaned_text[:B_next_chars]

            triggered_rules = BridgeTriggerRules.check_all(T_tail, T_head)

            if triggered_rules:
                bridge_text = T_tail + T_head  # D = T_tail || T_head
                bridge_chunks.append(TextChunk(
                    chunk_id=self._make_id(
                        doc.doc_id, curr_page.page_num, is_bridge=True
                    ),
                    doc_id=doc.doc_id,
                    text=bridge_text,
                    start_page=curr_page.page_num,
                    end_page=next_page.page_num,
                    chunk_index=i,
                    is_bridge=True,
                    triggered_rules=triggered_rules,
                ))
                logger.debug(
                    f"[{doc.doc_id}] 桥接触发: 第{curr_page.page_num}→{next_page.page_num}页, "
                    f"规则: {triggered_rules}"
                )

        return bridge_chunks

    # ── 工具方法 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _token_to_char(token_count: int) -> int:
        """
        简单 Token→字符近似（中文语料）。
        生产环境建议替换为 tiktoken: len(enc.encode(text))。
        """
        return int(token_count * 1.5)

    @staticmethod
    def _make_id(doc_id: str, page_num: int, offset: int = 0, is_bridge: bool = False) -> str:
        raw = f"{doc_id}|p{page_num}|o{offset}|br{is_bridge}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
