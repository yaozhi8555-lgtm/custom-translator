from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Quick 模式
# ─────────────────────────────────────────────

@dataclass
class QuickResult:
    """quick 模式：只返回译文"""
    original: str           # 原文
    translation: str        # 译文
    source_lang: str        # 原文语言，比如 "中文" / "英文" / "日文"
    target_lang: str        # 目标语言，比如 "英文" / "中文" / "日文"
    input_tokens: int
    output_tokens: int


# ─────────────────────────────────────────────
# Detailed 模式的子结构
# ─────────────────────────────────────────────

@dataclass
class SentenceComponent:
    """
    句子成分：主谓宾定状补、从句类型、虚拟语气等
    """
    component: str          # 成分名称，比如 "主语" / "定语从句" / "虚拟语气"
    original: str           # 原文对应片段，比如 "If I were you"
    explanation: str        # 语法说明


@dataclass
class VocabItem:
    """
    重点词汇 / 俚语 / 固定用法
    """
    word: str               # 单词或短语，比如 "exhausted" / "under the weather"
    type: str               # 类型：单词 / 俚语 / 固定搭配 / 惯用表达
    meaning: str            # 含义解释
    examples: list[str]     # 例句列表，至少1个，最多3个


# ─────────────────────────────────────────────
# Detailed 模式主结构
# ─────────────────────────────────────────────

@dataclass
class DetailedResult:
    """detailed 模式：完整语言分析"""
    # 基本信息
    original: str                               # 原文
    source_lang: str                            # 原文语言
    target_lang: str                            # 目标语言

    # 译文
    translation: str

    # 句子结构分析
    sentence_structure: list[SentenceComponent]

    # 特殊语法说明（虚拟语气、倒装、省略等），没有时为空字符串
    special_grammar: str

    # 重点词汇/俚语/用法
    vocabulary: list[VocabItem]

    # 整体评注（语气、使用场景、文化背景等）
    overall_note: str

    # 预留：其他语言的参考译文
    # key 是语言名，value 是该语言的直译
    # 现阶段为空字典 {}，以后加日文/韩文等直接往里填
    # 例：{"日文": "今日はとても疲れた", "韩文": "오늘 너무 피곤해"}
    other_languages: dict[str, str] = field(default_factory=dict)

    # Phase5 新增：可观测性字段，用来看清模型在工具调用循环里到底干了什么
    tools_used: list[str] = field(default_factory=list)  # 这次翻译调用了哪些工具
    tool_rounds: int = 0                                 # 循环了几轮（只知道用了哪些工具不够，还要知道绕了几圈才能判断成本）

    # token 统计
    input_tokens: int = 0
    output_tokens: int = 0