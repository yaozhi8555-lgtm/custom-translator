# 阶段2：结构化输出 + 双模式翻译

## 目标
把单一的"返回译文字符串"升级成"按模式返回结构化数据"：

```
quick 模式（phase1已有） →  只返回译文
detailed 模式（phase2新增）→  译文（直译+意译）+ 句子结构分析 + 重点词汇/俚语 + 整体评注
```

---

## 整体改动范围

```
src/translator/
├── schemas.py        ← 新增：定义输出的数据结构
├── prompts.py        ← 新增：集中管理所有 prompt
├── core.py           ← 修改：支持 mode 参数，处理 JSON 解析
└── cli.py            ← 修改：支持 --mode 参数，格式化打印结果
```

---

## 步骤1：定义输出结构（`schemas.py`）

**先想清楚要什么，再去写 prompt 要它。**

`@dataclass` 是 Python 内置模块，不需要 pip 安装：

```python
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


@dataclass
class TranslationLayer:
    """
    翻译的两个层次
    """
    literal: str            # 直译：逐字对应的翻译
    meaning: str            # 意译：更自然、符合目标语言习惯的翻译
    difference: str         # 说明直译和意译的区别


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

    # 翻译层（直译 + 意译）
    translation: TranslationLayer

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

    # token 统计
    input_tokens: int = 0
    output_tokens: int = 0
```

---

## 步骤2：集中管理 prompt（`prompts.py`）

```python
QUICK_SYSTEM_PROMPT = """你是一个专业翻译助手。
请将用户输入的文本准确翻译成{target_lang}。
只输出译文本身，不要添加任何解释、说明或额外内容。"""


DETAILED_SYSTEM_PROMPT = """你是一个专业翻译助手兼语言教学专家。
请将用户输入的文本从{source_lang}翻译成{target_lang}，并提供完整的语言分析。

你必须严格按照以下 JSON 格式输出，不要输出任何其他内容，不要加 markdown 代码块：

{{
  "translation": {{
    "literal": "逐字直译结果",
    "meaning": "更自然的意译结果",
    "difference": "说明直译和意译的区别，以及各自适用的场合"
  }},
  "sentence_structure": [
    {{
      "component": "语法成分名称，如主语/谓语/宾语/定语从句/状语从句/虚拟语气等",
      "original": "原文中对应的片段",
      "explanation": "这个成分的详细语法解释"
    }}
  ],
  "special_grammar": "如果句子中有虚拟语气、倒装、省略、强调句等特殊语法现象，在这里专门说明。没有则返回空字符串",
  "vocabulary": [
    {{
      "word": "挑选出的重点单词、俚语或固定搭配",
      "type": "类型：单词 / 俚语 / 固定搭配 / 惯用表达",
      "meaning": "含义和用法说明",
      "examples": [
        "包含这个词的例句1",
        "包含这个词的例句2"
      ]
    }}
  ],
  "overall_note": "对整句话的整体评注，包括语气、使用场景、文化背景、近义表达的程度差异等",
  "other_languages": {{}}
}}

注意事项：
- sentence_structure 必须覆盖原文的每一个主要成分，不能遗漏
- vocabulary 挑选3-5个最值得学习的词汇或表达，每个词至少给2个例句
- 如果原文中有俚语或固定搭配，优先收录
- other_languages 暂时返回空对象 {{}}"""
```

---

## 步骤3：更新 `core.py`

```python
import json
from .config import OPENROUTER_API_KEY, DEFAULT_MODEL
from .providers.openrouter import OpenRouterProvider
from .schemas import (
    QuickResult, DetailedResult,
    TranslationLayer, SentenceComponent, VocabItem
)
from .prompts import QUICK_SYSTEM_PROMPT, DETAILED_SYSTEM_PROMPT


def translate(
    text: str,
    source_lang: str = "中文",
    target_lang: str = "英文",
    mode: str = "quick",
    model: str = DEFAULT_MODEL
) -> QuickResult | DetailedResult:

    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    if mode == "quick":
        return _translate_quick(provider, text, source_lang, target_lang, model)
    elif mode == "detailed":
        return _translate_detailed(provider, text, source_lang, target_lang, model)
    else:
        raise ValueError(f"未知模式：{mode}，请使用 quick 或 detailed")


def _translate_quick(provider, text, source_lang, target_lang, model) -> QuickResult:
    system_prompt = QUICK_SYSTEM_PROMPT.format(target_lang=target_lang)
    result = provider.chat(
        system_prompt=system_prompt,
        user_message=text,
        model=model
    )
    return QuickResult(
        original=text,
        translation=result["content"],
        source_lang=source_lang,
        target_lang=target_lang,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )


def _translate_detailed(provider, text, source_lang, target_lang, model) -> DetailedResult:
    system_prompt = DETAILED_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )
    result = provider.chat(
        system_prompt=system_prompt,
        user_message=text,
        model=model
    )

    parsed = _parse_json(result["content"])

    # 组装 TranslationLayer
    t = parsed["translation"]
    translation = TranslationLayer(
        literal=t["literal"],
        meaning=t["meaning"],
        difference=t["difference"],
    )

    # 组装 SentenceComponent 列表
    sentence_structure = [
        SentenceComponent(
            component=s["component"],
            original=s["original"],
            explanation=s["explanation"],
        )
        for s in parsed.get("sentence_structure", [])
    ]

    # 组装 VocabItem 列表
    vocabulary = [
        VocabItem(
            word=v["word"],
            type=v["type"],
            meaning=v["meaning"],
            examples=v.get("examples", []),
        )
        for v in parsed.get("vocabulary", [])
    ]

    return DetailedResult(
        original=text,
        source_lang=source_lang,
        target_lang=target_lang,
        translation=translation,
        sentence_structure=sentence_structure,
        special_grammar=parsed.get("special_grammar", ""),
        vocabulary=vocabulary,
        overall_note=parsed.get("overall_note", ""),
        other_languages=parsed.get("other_languages", {}),
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )


def _parse_json(content: str) -> dict:
    """
    解析模型返回的 JSON 字符串。
    模型有时会在 JSON 外面包一层 markdown 代码块，需要先清理掉。
    """
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"模型返回的内容不是有效的 JSON 格式。\n"
            f"原始内容：\n{content}\n"
            f"错误详情：{e}"
        )
```

---

## 步骤4：更新 `cli.py`

新增 `--from` 参数指定原文语言：

```python
import argparse
from .core import translate
from .schemas import QuickResult, DetailedResult


def main():
    parser = argparse.ArgumentParser(description="定制化翻译工具")
    parser.add_argument("text", help="要翻译的文本")
    parser.add_argument("--from", dest="source_lang", default="中文", help="原文语言，默认中文")
    parser.add_argument("--to", default="英文", help="目标语言，默认英文")
    parser.add_argument(
        "--mode",
        default="quick",
        choices=["quick", "detailed"],
        help="翻译模式：quick（仅直译）或 detailed（完整分析），默认 quick"
    )
    args = parser.parse_args()

    try:
        result = translate(
            args.text,
            source_lang=args.source_lang,
            target_lang=args.to,
            mode=args.mode,
        )
        _print_result(result)
    except RuntimeError as e:
        print(f"\n错误：{e}")


def _print_result(result):
    if isinstance(result, QuickResult):
        _print_quick(result)
    elif isinstance(result, DetailedResult):
        _print_detailed(result)


def _print_quick(result: QuickResult):
    print(f"\n译文：{result.translation}")
    print(f"(消耗 tokens：输入 {result.input_tokens} / 输出 {result.output_tokens})")


def _print_detailed(result: DetailedResult):
    LINE = "=" * 55

    print(f"\n{LINE}")
    print(f"原文（{result.source_lang}）：{result.original}")
    print(f"\n【直译】{result.translation.literal}")
    print(f"【意译】{result.translation.meaning}")
    print(f"【区别】{result.translation.difference}")

    print(f"\n--- 句子结构 ---")
    for s in result.sentence_structure:
        print(f"  [{s.component}]  {s.original}")
        print(f"    └─ {s.explanation}")

    if result.special_grammar:
        print(f"\n--- 特殊语法 ---")
        print(f"  {result.special_grammar}")

    if result.vocabulary:
        print(f"\n--- 重点词汇 / 俚语 / 用法 ---")
        for v in result.vocabulary:
            print(f"\n  {v.word}（{v.type}）")
            print(f"  含义：{v.meaning}")
            for ex in v.examples:
                print(f"    · {ex}")

    print(f"\n--- 整体评注 ---")
    print(f"  {result.overall_note}")

    if result.other_languages:
        print(f"\n--- 其他语言参考 ---")
        for lang, trans in result.other_languages.items():
            print(f"  {lang}：{trans}")

    print(f"\n(消耗 tokens：输入 {result.input_tokens} / 输出 {result.output_tokens})")
    print(f"{LINE}\n")


if __name__ == "__main__":
    main()
```

---

## 步骤5：手动验证

```bash
# quick 模式，行为和 phase1 一样
python -m translator.cli "我今天很累" --to 英文

# detailed 模式，中译英
python -m translator.cli "我今天很累" --mode detailed

# detailed 模式，英译中
python -m translator.cli "If I were you, I wouldn't do that." --from 英文 --to 中文 --mode detailed
```

**英译中** detailed 模式预期输出大概长这样：

```
=======================================================
原文（英文）：If I were you, I wouldn't do that.

【直译】如果我是你，我不会那样做。
【意译】换我是你的话，我才不会那么做呢。
【区别】直译保留句子结构，意译更口语化自然，适合日常对话

--- 句子结构 ---
  [条件状语从句]  If I were you
    └─ 虚拟条件句，"were"是虚拟语气标志，表示与现实相反的假设
  [主句主语]  I
    └─ 第一人称单数
  [主句谓语]  wouldn't do
    └─ would not + 动词原形，虚拟语气主句结构，表示假设情况下的结果
  [宾语]  that
    └─ 指代上文提到的某件事

--- 特殊语法 ---
  虚拟语气（Subjunctive Mood）：条件句使用 "If + 主语 + were"，
  主句使用 "would/could/might + 动词原形"，表达与现实相反的假设。
  注意：虚拟语气中所有人称都用 "were"，不用 "was"。

--- 重点词汇 / 俚语 / 用法 ---

  If I were you（惯用表达）
  含义：表示"如果我处于你的位置"，常用于给出建议，语气委婉
    · If I were you, I'd apologize to her.
    · If I were you, I'd take the job offer.

  wouldn't（固定搭配）
  含义：would not 的缩写，在此处表示虚拟语气下的否定意愿
    · I wouldn't say that if I were you.
    · She wouldn't agree to those terms.

--- 整体评注 ---
  这是一个经典的虚拟条件句，在英语中极为常见，
  用于给别人提建议时既委婉又有力。
  类似表达还有 "In your shoes, I would..." / "Were I you..."（更正式）。

(消耗 tokens：输入 185 / 输出 312)
=======================================================
```

**如果 JSON 解析失败**：`_parse_json` 会把模型返回的原始内容打印出来，对照 prompt 要求的格式看哪里不匹配，再调整 prompt。这是 prompt engineering 最核心的调试方式。

---

## 步骤6：提交代码

```bash
git add .
git commit -m "阶段2: 结构化输出，quick/detailed 双模式，直译意译，句子成分，词汇分析"
```

---

## 步骤7：更新文档

把 `CLAUDE.md` 里的当前阶段改为 phase2 完成，phase3 待开始。

---

## Phase2 学习重点

| 知识点 | 在哪里体现 |
|---|---|
| 结构化输出 | `schemas.py` 定义数据形状，prompt 要求返回对应 JSON |
| Prompt 设计 | `prompts.py` 里 `{{}}` 转义、JSON schema 内嵌在 prompt 里 |
| JSON 解析防御性处理 | `_parse_json()` 清理 markdown 代码块再解析 |
| 多层嵌套数据结构 | `DetailedResult` 包含 `TranslationLayer`、`SentenceComponent` 列表、`VocabItem` 列表 |
| 类型判断 | `cli.py` 里用 `isinstance()` 区分两种结果类型 |
| 预留扩展位置 | `other_languages` 字段、`source_lang` 参数 |
