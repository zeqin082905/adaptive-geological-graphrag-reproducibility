"""
步骤7：查询标准化器（扩充版）
三级标准化流水线 + 关键词精确预匹配
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import llm

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 扩充地质实体词典
# ══════════════════════════════════════════════════════════════════════════════
GEO_ENTITY_DICT = {
    '岩石': [
        '花岗岩','玄武岩','安山岩','流纹岩','英安岩','粗面岩','响岩','霞石正长岩',
        '辉长岩','闪长岩','正长岩','橄榄岩','辉石岩','角闪石岩','蛇纹岩','玢岩',
        '细晶岩','伟晶岩','煌斑岩','脉岩','超基性岩','基性岩','中性岩','酸性岩',
        '石灰岩','白云岩','页岩','砂岩','砾岩','泥岩','粉砂岩','凝灰岩','火山角砾岩',
        '硅质岩','燧石','碧玉岩','铁质岩','锰质岩','磷质岩','盐岩','石膏岩','钙质岩',
        '泥质岩','砂质岩','碳酸盐岩','碎屑岩','生物岩','化学岩',
        '板岩','千枚岩','片岩','片麻岩','大理岩','石英岩','矽卡岩','角岩','糜棱岩',
        '混合岩','变粒岩','麻粒岩','榴辉岩','蓝片岩','碎裂岩','构造角砾岩',
        '绿片岩','角闪岩','云母片岩','石榴片岩','斜长角闪岩',
        '变质岩','沉积岩','火成岩','岩浆岩','侵入岩','喷出岩','火山岩',
    ],
    '矿物': [
        '石英','长石','云母','角闪石','辉石','橄榄石','钙长石','斜长石','黑云母','白云母',
        '正长石','微斜长石','透长石','钾长石','拉长石','中长石','更长石',
        '金云母','锂云母','绢云母','透闪石','阳起石','普通角闪石','蓝闪石',
        '透辉石','顽火辉石','普通辉石','紫苏辉石','镁橄榄石','铁橄榄石',
        '方解石','白云石','文石','菱铁矿','菱锰矿','菱镁矿','菱锌矿','孔雀石','蓝铜矿',
        '黄铁矿','黄铜矿','方铅矿','闪锌矿','磁黄铁矿','斑铜矿','辉铜矿','铜蓝',
        '辉钼矿','辉锑矿','辉砷矿','毒砂','雄黄','雌黄','辰砂','自然金','自然银',
        '自然铜','自然硫','自然铂',
        '磁铁矿','赤铁矿','褐铁矿','钛铁矿','铬铁矿','刚玉','尖晶石','金红石',
        '锡石','软锰矿','硬锰矿','水锰矿','铝土矿',
        '金刚石','石墨','石膏','硬石膏','重晶石','磷灰石','锆石',
        '石榴石','红柱石','蓝晶石','矽线石','十字石','绿帘石','绿泥石','蛇纹石',
        '滑石','高岭石','蒙脱石','伊利石','沸石','蛋白石','玉髓','玛瑙',
        '电气石','绿柱石','黄玉','水晶','紫晶','烟晶',
    ],
    '构造': [
        '断层','断裂','断裂带','断层系统','正断层','逆断层','逆冲断层','走滑断层',
        '平移断层','转换断层','铲形断层','叠瓦状断层','阶梯状断层','地堑','地垒',
        '褶皱','背斜','向斜','穹隆','盆地','复背斜','复向斜','倒转褶皱','平卧褶皱',
        '线状褶皱','穹窿构造','短轴褶皱','单斜','单斜构造',
        '节理','劈理','线理','面理','片理','片麻理','流劈理','破劈理','板劈理','千枚理',
        '推覆构造','滑脱构造','韧性剪切带','脆性剪切带','伸展构造','压缩构造','剪切构造',
        '叠加构造','多期构造','复合构造',
        '不整合','角度不整合','平行不整合','侵入接触','沉积接触','断层接触','整合接触',
        '克拉通','地台','基底','盖层','造山带','裂谷','俯冲带','碰撞带',
        '控矿构造','赋矿构造','导矿构造',
    ],
    '地层': [
        '太古宙','元古宙','古生界','中生界','新生界','太古界','元古界','显生界',
        '寒武系','奥陶系','志留系','泥盆系','石炭系','二叠系',
        '三叠系','侏罗系','白垩系','古近系','新近系','第四系',
        '长城系','蓟县系','青白口系','震旦系','南华系','昆阳群','蓬莱群',
        '五台群','滹沱群','嵩箕群','晋宁群',
        '前寒武系','前寒武纪地层','古生代地层','中生代地层','新生代地层',
        '海相地层','陆相地层','海陆交互相地层',
        '太华群','霍邱群',
    ],
    '矿产': [
        '铁矿','锰矿','铬矿','钒矿','钛矿','钛铁矿',
        '铜矿','铅矿','锌矿','铅锌矿','镍矿','钴矿','钨矿','锡矿','钼矿','锑矿',
        '汞矿','铋矿','铝土矿','镁矿',
        '金矿','银矿','铂矿','钯矿',
        '锂矿','铍矿','铌矿','钽矿','锆矿','铪矿','铷矿','铯矿',
        '稀土矿','轻稀土','重稀土','独居石','氟碳铈矿',
        '煤矿','石油','天然气','油页岩','油砂','煤层气','页岩气',
        '磷矿','硫矿','钾盐','岩盐','芒硝','天然碱','硼矿',
        '石膏','硬石膏','水泥灰岩','高岭土','膨润土',
        '矿产资源','金属矿产','非金属矿产','能源矿产',
    ],
    '地质作用': [
        '风化作用','物理风化','化学风化','生物风化',
        '侵蚀作用','搬运作用','沉积作用','成岩作用','压实作用','胶结作用',
        '岩浆作用','火山作用','侵入作用','喷出作用','岩浆分异作用',
        '变质作用','接触变质作用','区域变质作用','动力变质作用',
        '热液作用','热液蚀变','交代作用','蚀变作用','矽卡岩化',
        '硅化','碳酸盐化','绿泥石化','黄铁矿化','褐铁矿化',
        '成矿作用','岩浆成矿','热液成矿','沉积成矿','变质成矿',
        '构造运动','造山运动','板块运动',
    ],
    '矿床类型': [
        '岩浆型矿床','伟晶岩型矿床','热液型矿床','斑岩型矿床','矽卡岩型矿床',
        '矽卡岩型铁矿','沉积型矿床','层控矿床','变质型矿床','风化壳型矿床',
        'VMS型矿床','MVT型矿床','SEDEX型矿床','砂矿',
        '接触交代型矿床','沉积变质型铁矿',
    ],
}

# ── 同义词映射 ────────────────────────────────────────────────────────────────
GEO_SYNONYM_MAP: dict[str, str] = {
    '青石': '石灰岩', '灰岩': '石灰岩',
    '长岩脉': '闪长玢岩', '闪长玢岩体': '闪长玢岩',
    '花岗体': '花岗岩', '花岗侵入体': '花岗岩',
    '玄武质': '玄武岩',
    '大铁矿': '铁矿床', '铁矿山': '铁矿床',
    'Au矿': '金矿', '金矿点': '金矿',
    '铜矿山': '铜矿床', '铜山': '铜矿床',
    '煤矿层': '煤层', '煤田': '煤矿',
    '钼矿化': '钼矿',
    '大断裂': '断裂带', '主断层': '主断裂',
    '背斜核': '背斜', '向斜核': '向斜',
    '前寒武地层': '前寒武系', '太古代地层': '太古宇',
    'Fe': '铁', 'Cu': '铜', 'Pb': '铅', 'Zn': '锌',
    'Au': '金', 'Ag': '银', 'Mo': '钼', 'W': '钨',
    # 霍邱特定
    '班台子矿': '班台子铁矿',
    '霍邱铁矿': '霍邱县铁矿',
}

# ── 正则复合实体识别规则 ──────────────────────────────────────────────────────
GEO_REGEX_RULES: list[dict] = [
    {
        'name': 'Rule-1:区域矿区',
        'pattern': re.compile(
            r'(华北|华南|滇西|川西|皖西|豫西|陕南|赣北|湘中|霍邱|安徽)'
            r'([铜铁金银铅锌锰钼钨铝铬镍磷]+)'
            r'矿(区|床|山)?'
        ),
        'entity_type': '矿区名称',
        'normalize': lambda m: f"{m.group(1)}{m.group(2)}矿区",
    },
    {
        'name': 'Rule-2:复合岩石名',
        'pattern': re.compile(
            r'(黑云|白云|角闪|辉石|石榴|斜长|钾长)(片麻|大理|石英|角闪|绿片|千枚)岩'
        ),
        'entity_type': '岩石',
        'normalize': lambda m: m.group(0),
    },
    {
        'name': 'Rule-3:地质年代地层',
        'pattern': re.compile(
            r'(前寒武|寒武|奥陶|志留|泥盆|石炭|二叠|三叠|侏罗|白垩|古近|新近|第四|太古|元古)'
            r'(系|纪|代|界|宙|宇)?(地层|组|段|统|群)?'
        ),
        'entity_type': '地层',
        'normalize': lambda m: m.group(0).strip(),
    },
    {
        'name': 'Rule-4:矿化类型',
        'pattern': re.compile(
            r'(黄铁矿化|硅化|褐铁矿化|铜钼矿化|金矿化|铅锌矿化|磁铁矿化|赤铁矿化)'
        ),
        'entity_type': '地质作用',
        'normalize': lambda m: m.group(0),
    },
    {
        'name': 'Rule-5:矿床类型',
        'pattern': re.compile(
            r'(斑岩型|矽卡岩型|热液型|沉积型|变质型|风化型|岩浆型|接触交代型|沉积变质型)'
            r'([铜铁金银铅锌钼钨]+)?矿(床|化)?'
        ),
        'entity_type': '矿产',
        'normalize': lambda m: m.group(0),
    },
    {
        'name': 'Rule-6:控矿构造',
        'pattern': re.compile(
            r'([\u4e00-\u9fff]{2,6})(单斜|背斜|向斜|断裂|断层|褶皱|穹隆|盆地)(构造|带|系)?'
        ),
        'entity_type': '构造',
        'normalize': lambda m: m.group(0),
    },
]


# ── 关键词提取器 ──────────────────────────────────────────────────────────────
class GeoKeywordExtractor:
    """
    从查询中提取地质实体关键词，用于图谱检索前的精确字符串预匹配。
    解决向量模型对中文地质专名语义理解弱的问题。
    """
    _all_terms: set[str] = set()

    def __init__(self):
        if not self._all_terms:
            for terms in GEO_ENTITY_DICT.values():
                self._all_terms.update(terms)

    def extract(self, query: str) -> list[str]:
        found = []
        # 词典精确匹配（长词优先）
        for term in sorted(self._all_terms, key=len, reverse=True):
            if term in query and term not in found:
                found.append(term)
        # 地名+矿类短语
        for m in re.finditer(r'[\u4e00-\u9fff]{2,8}(?:铁矿|铜矿|金矿|银矿|铅矿|锌矿|钼矿|煤矿|矿区|矿床|矿山|矿田)', query):
            if m.group(0) not in found:
                found.append(m.group(0))
        # 地名+构造短语
        for m in re.finditer(r'[\u4e00-\u9fff]{2,6}(?:单斜|背斜|向斜|断裂|断层|褶皱|穹隆|盆地|地堑|地垒)', query):
            if m.group(0) not in found:
                found.append(m.group(0))
        return found


# ── 标准化结果 ────────────────────────────────────────────────────────────────
@dataclass
class NormalizationResult:
    original_query: str
    normalized_query: str
    extracted_entities: list[dict] = field(default_factory=list)
    keyword_terms: list[str] = field(default_factory=list)
    applied_mappings: list[str] = field(default_factory=list)


# ── 查询标准化器 ──────────────────────────────────────────────────────────────
class QueryNormalizer:

    LLM_NORMALIZE_PROMPT = """\
【Role 角色】
你是一位地质矿产领域的术语标准化专家，熟悉中国地质调查局规范用语体系，\
擅长识别地质口语化表述、历史俗称、化学符号简写，并将其映射至标准地质学术名词。

【Action 动作】
对用户输入的地质问题执行术语标准化处理：
1. 识别问题中的口语化词汇、简写或俗称（如"Au矿"→"金矿"，"青石"→"石灰岩"）
2. 将其替换为对应的标准地质术语
3. 保持问题语义、语序及非地质词汇完全不变
4. 仅返回替换后的完整问题文本

【Scope 范围】
✅ 矿产俗称/简写（Au/Cu/Fe等元素符号，大铁矿/煤矿层等俗称）
✅ 岩石俗称（青石、灰岩等）
✅ 地质构造口语化表述（大断裂→断裂带）
✅ 地层年代简写
❌ 地名、单位名称、非地质专业词汇——保持原样不做修改
❌ 不增加任何解释性文字

【Example 示例】
输入："霍邱那个大铁矿受什么断裂控制？"
输出："霍邱铁矿床受什么断裂带控制？"

输入："Au矿和黄铁矿化有啥关系？"
输出："金矿与黄铁矿化有什么关系？"

输入："这个灰岩层位叫啥？"
输出："该石灰岩层位名称是什么？"

【Format 格式】
- 仅输出标准化后的问题文本，不加任何前缀或后缀说明
- 单行纯文本，不使用标点符号以外的任何标记
- 禁止输出"标准化后："等引导性文字

用户原始问题：{query}
标准化后的问题："""

    def __init__(self, use_llm_fallback: bool = True):
        self._synonyms = GEO_SYNONYM_MAP
        self._rules = GEO_REGEX_RULES
        self._use_llm = use_llm_fallback
        self._keyword_extractor = GeoKeywordExtractor()

    def normalize(self, query: str) -> NormalizationResult:
        result = NormalizationResult(original_query=query, normalized_query=query)
        result = self._apply_synonyms(result)
        result = self._apply_regex_rules(result)
        if self._use_llm and self._needs_llm(result):
            result = self._apply_llm_normalization(result)
        result.keyword_terms = self._keyword_extractor.extract(result.normalized_query)
        if result.keyword_terms:
            logger.info(f"关键词提取: {result.keyword_terms}")
        return result

    def _apply_synonyms(self, result: NormalizationResult) -> NormalizationResult:
        text = result.normalized_query
        for src, tgt in self._synonyms.items():
            if src in text:
                text = text.replace(src, tgt)
                result.applied_mappings.append(f"同义词: {src}→{tgt}")
        result.normalized_query = text
        return result

    def _apply_regex_rules(self, result: NormalizationResult) -> NormalizationResult:
        text = result.normalized_query
        for rule in self._rules:
            for match in rule['pattern'].finditer(text):
                normalized_name = rule['normalize'](match)
                result.extracted_entities.append({
                    'name': normalized_name,
                    'type': rule['entity_type'],
                    'original': match.group(0),
                    'rule': rule['name'],
                })
                if normalized_name != match.group(0):
                    text = text.replace(match.group(0), normalized_name)
                    result.applied_mappings.append(f"正则: {match.group(0)}→{normalized_name}")
        result.normalized_query = text
        return result

    def _apply_llm_normalization(self, result: NormalizationResult) -> NormalizationResult:
        try:
            llm_result = llm.generate(
                self.LLM_NORMALIZE_PROMPT.format(query=result.normalized_query),
                temperature=0.0
            )
            if llm_result and llm_result != result.normalized_query:
                result.applied_mappings.append('LLM兜底标准化')
                result.normalized_query = llm_result.strip()
        except Exception as e:
            logger.warning(f"LLM 标准化失败: {e}")
        return result

    @staticmethod
    def _needs_llm(result: NormalizationResult) -> bool:
        if result.applied_mappings:
            return False
        return any(m in result.original_query for m in ['哪里','怎么','为啥','大概','差不多','那个','啥矿'])


# 全局单例
normalizer = QueryNormalizer()
