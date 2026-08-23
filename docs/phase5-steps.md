# 阶段5：自定义工具调用（Tool Use）

## 目标

把 Phase3/4 里"每次翻译都固定查词典+历史"的**被动检索**，升级为"让模型自己决定要不要查、查什么"的**主动决策**：

```
Phase3/4（被动）：
  用户输入 → 固定查词典 → 固定查历史 → 注入 prompt → 翻译

Phase5（主动）：
  用户输入 → 告诉模型"你有这些工具" → 模型自己判断：
    "这句话有个多义词，我要查词典"      → 调用 lookup_dictionary
    "这个术语我好像翻译过，查一下"       → 调用 search_history
    "好了，信息够了，开始翻译"           → 生成最终译文
```

这是迈向 Agent 的关键一步——**把决策权还给模型**，而不是程序写死每一步。

---

## 什么是 Tool Use

Tool Use（工具调用）也叫 Function Calling，是让 LLM 在生成回复时能够"暂停"，调用外部函数，拿到结果后再继续生成的能力。

整个流程是一个**循环**（Loop）：

```
第1轮：
  你 → 模型："帮我翻译这句话，你有这些工具可用：[工具列表]"
  模型 → 你："我要调用 lookup_dictionary，参数 {word: '冷漠'}"

第2轮：
  你 → 模型："lookup_dictionary 结果：冷漠: indifferent; cold"
  模型 → 你："我还要调用 search_history，参数 {text: '他的态度很冷漠'}"

第3轮：
  你 → 模型："search_history 结果：（空，无相似历史）"
  模型 → 你："信息够了。译文是：His attitude is cold and indifferent."

循环结束：模型不再调用工具，返回最终结果。
```

---

## 一个关键决策：只在 detailed 模式启用工具

| | quick 模式 | detailed 模式 |
|---|---|---|
| 检索方式 | 保持 Phase4 的**固定检索** | Phase5 的**工具调用** |
| 请求次数 | 1 次 | 2-4 次 |
| 适用场景 | 快速翻译整段文字 | 单句深度学习 |

**理由**：quick 模式追求快和省，固定检索一次就够；detailed 模式追求准确和深度，值得多花 token 让模型自主决策。这也符合项目一开始"控制成本"的目标。

---

## 成本影响分析（重要）

工具调用的 token 消耗**不是线性增长，而是累积增长**：

```
第1轮 input：system prompt + 用户输入            ≈ 300 tokens
第2轮 input：上面全部 + 模型回复 + 工具结果1      ≈ 550 tokens
第3轮 input：上面全部 + 模型回复 + 工具结果2      ≈ 800 tokens
────────────────────────────────────────────────
总 input ≈ 1650 tokens（而 Phase4 只需 300）
```

因为每一轮都要把**完整的消息历史**发送过去，模型才知道之前发生了什么。

**应对措施**：
1. quick 模式不启用工具（上面已决策）
2. 提供 `--no-tools` 开关，随时可以退回固定检索模式
3. `max_rounds` 限制最多循环轮数
4. 观察 `tools_used` 和 token 统计，评估工具是否真的带来价值

---

## 整体改动范围

```
src/translator/
├── tools/                    ← 新增
│   ├── __init__.py
│   ├── definitions.py        # 工具的 JSON Schema
│   └── executor.py           # 工具执行器（含容错处理）
├── providers/
│   ├── base.py               ← 修改：拆成 chat() 和 chat_with_tools()
│   └── openrouter.py         ← 修改：实现两个方法
├── core.py                   ← 修改：detailed 模式改用工具调用循环
├── cli.py                    ← 修改：加 --no-tools 开关
└── schemas.py                ← 修改：DetailedResult 加 tools_used 字段
```

> **前置确认**：`cli.py` 里应已包含 Phase4 修复的 `detect_source_lang()` 自动语言检测，以及 `source_lang` / `target_lang` 的 `None` 兜底逻辑。如果还没加，先补上再进行 Phase5。

---

## 步骤1：定义工具 Schema（`tools/definitions.py`）

Tool Schema 是告诉模型"这个工具叫什么、干什么用、要什么参数"的说明书，遵循 OpenAI function calling 标准（OpenRouter 兼容）：

```python
LOOKUP_DICTIONARY = {
    "type": "function",
    "function": {
        "name": "lookup_dictionary",
        "description": (
            "在 CC-CEDICT 词典中查找词条的准确含义。"
            "使用场景：遇到多义词、专业术语、不确定含义的词，"
            "或者需要确认某个词在特定语境下的准确释义时。"
            "不要对简单常用词（如'我''今天''好'）调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "要查询的词语，通常是2-4个汉字，或一个英文单词/短语",
                }
            },
            "required": ["word"],
        },
    },
}

SEARCH_HISTORY = {
    "type": "function",
    "function": {
        "name": "search_history",
        "description": (
            "在翻译历史中检索语义相似的句子。"
            "使用场景：需要保持术语翻译一致性时，"
            "或当前内容可能和之前翻译过的内容相关时。"
            "对全新的、不太可能重复的内容不需要调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要检索的文本，通常是当前需要翻译的原文",
                },
                "target_lang": {
                    "type": "string",
                    "description": "目标语言，如'英文'或'中文'，用于筛选同语言历史记录",
                },
            },
            "required": ["text", "target_lang"],
        },
    },
}

ALL_TOOLS = [LOOKUP_DICTIONARY, SEARCH_HISTORY]
```

**`description` 是整个 Schema 里最重要的部分**——模型完全靠它判断何时调用工具。注意上面两个描述里都写了**反面例子**（"不要对简单常用词调用"、"全新内容不需要调用"），这能显著减少模型的无效调用，直接影响 token 消耗。

---

## 步骤2：实现工具执行器（`tools/executor.py`）

关键设计：**工具失败不能让整个流程崩溃**，要把错误作为结果返回给模型，让模型自己决定怎么应对。

```python
from ..dictionary.store import lookup
from ..history.retriever import retrieve_similar
from ..config import DICT_DB_PATH


def execute(tool_name: str, tool_args: dict) -> str:
    """
    执行工具，返回字符串结果（会作为消息发回给模型）。

    重要：这个函数永远不抛异常。
    任何错误都被捕获并转成描述性的字符串返回给模型，
    让模型知道"这个工具没用上"，然后自己决定怎么继续。
    这是 Agent 系统的核心容错原则：工具失败 ≠ Agent 崩溃。
    """
    try:
        if tool_name == "lookup_dictionary":
            return _execute_lookup_dictionary(tool_args)
        elif tool_name == "search_history":
            return _execute_search_history(tool_args)
        else:
            return f"错误：不存在名为 {tool_name} 的工具。请使用已提供的工具。"

    except Exception as e:
        # 任何异常都转成给模型看的说明，不向上抛
        return (
            f"工具 {tool_name} 执行失败：{type(e).__name__}: {e}。"
            f"请在没有这个工具结果的情况下继续完成翻译。"
        )


def _execute_lookup_dictionary(args: dict) -> str:
    word = args.get("word", "").strip()
    if not word:
        return "参数错误：word 不能为空"

    # 模型已经决定了查哪个词，直接精确查找，不需要滑动窗口
    entries = lookup(str(DICT_DB_PATH), word)

    if not entries:
        return f"词典中未收录「{word}」。这可能是新词、专有名词或音译词。"

    lines = [f"词典查询结果（{word}）："]
    for entry in entries:
        defs = "; ".join(entry.definitions)
        lines.append(f"  {entry.simplified}: {defs}")
    return "\n".join(lines)


def _execute_search_history(args: dict) -> str:
    text = args.get("text", "").strip()
    target_lang = args.get("target_lang", "").strip()

    if not text:
        return "参数错误：text 不能为空"
    if not target_lang:
        return "参数错误：target_lang 不能为空"

    records = retrieve_similar(text, target_lang)

    if not records:
        return "翻译历史中没有找到语义相似的句子。这是一次全新的翻译。"

    lines = ["历史翻译检索结果（相似度从高到低）："]
    for r in records:
        lines.append(f"  原文：{r['original']}")
        lines.append(f"  译文：{r['translation']}")
        lines.append(f"  相似度：{r['similarity']}")
    return "\n".join(lines)
```

注意几个"空结果"的返回措辞——不是简单返回"没找到"，而是给模型有用的**信号**（"这可能是新词、专有名词或音译词"），帮助模型判断下一步怎么做。

---

## 步骤3：拆分 Provider 接口

Phase4 的 `chat()` 如果硬塞 `tools` 和 `messages` 参数，会出现"传了 messages 之后 system_prompt 和 user_message 就被忽略"的怪异设计。拆成两个方法，职责清晰。

**`providers/base.py`**：

```python
from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_message: str, model: str) -> dict:
        """
        单轮对话，不带工具。Phase1-4 的用法。

        返回：
        {
            "content": str,
            "input_tokens": int,
            "output_tokens": int,
        }
        """
        pass

    @abstractmethod
    def chat_with_tools(self, messages: list, tools: list, model: str) -> dict:
        """
        带工具的多轮对话。Phase5 的用法。

        messages: 完整的消息历史列表，调用方负责维护
        tools: 工具 schema 列表

        返回：
        {
            "content": str | None,        # 有工具调用时通常为 None
            "tool_calls": list | None,    # 无工具调用时为 None
            "input_tokens": int,
            "output_tokens": int,
        }
        """
        pass
```

**`providers/openrouter.py`**，把异常处理抽成共用装饰器，避免两个方法重复写：

```python
import functools
import openai
from openai import OpenAI
from .base import BaseProvider


def handle_api_errors(func):
    """把 openai 的原始异常转成人话 RuntimeError，两个方法共用。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except openai.AuthenticationError:
            raise RuntimeError("API Key 错误，请检查 .env 里的 OPENROUTER_API_KEY")
        except openai.RateLimitError:
            raise RuntimeError("请求太频繁或余额不足，请稍后再试")
        except openai.APIConnectionError:
            raise RuntimeError("网络连接失败，请检查网络后重试")
        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenRouter 返回错误：{e.status_code} - {e.message}")
    return wrapper


class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    @handle_api_errors
    def chat(self, system_prompt: str, user_message: str, model: str) -> dict:
        """单轮对话，Phase1-4 的实现保持不变。"""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    @handle_api_errors
    def chat_with_tools(self, messages: list, tools: list, model: str) -> dict:
        """带工具的多轮对话。"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,   # JSON 字符串
                }
                for tc in message.tool_calls
            ]

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
```

`@handle_api_errors` 是**装饰器**——把异常处理逻辑抽出来，加在函数上方一行就能复用，不用在每个方法里重复写 try/except。这是 Python 里很常用的设计模式。

---

## 步骤4：更新 `schemas.py`

```python
@dataclass
class DetailedResult:
    original: str
    source_lang: str
    target_lang: str
    translation: TranslationLayer
    sentence_structure: list[SentenceComponent]
    special_grammar: str
    vocabulary: list[VocabItem]
    overall_note: str
    other_languages: dict[str, str] = field(default_factory=dict)

    # Phase5 新增：可观测性字段
    tools_used: list[str] = field(default_factory=list)   # 调用了哪些工具
    tool_rounds: int = 0                                   # 循环了几轮

    input_tokens: int = 0
    output_tokens: int = 0
```

`tool_rounds` 是新加的——只知道"用了哪些工具"不够，还要知道"绕了几轮"才能判断成本和效率。

---

## 步骤5：实现工具调用循环（`core.py`）

这是 Phase5 最核心的代码，注意循环状态管理和失败处理：

```python
import json
from .tools.definitions import ALL_TOOLS
from .tools.executor import execute as execute_tool


MAX_TOOL_ROUNDS = 5   # 最多循环几轮，防止模型陷入无限调用


def _translate_detailed(
    provider, text, source_lang, target_lang, model,
    use_tools: bool = True,      # 新增：可以关闭工具调用
) -> DetailedResult:
    """
    detailed 模式翻译。
    use_tools=True  → Phase5 的工具调用循环（模型自主决策）
    use_tools=False → Phase4 的固定检索（成本更低）
    """
    if use_tools:
        return _translate_detailed_with_tools(
            provider, text, source_lang, target_lang, model
        )
    else:
        return _translate_detailed_fixed(
            provider, text, source_lang, target_lang, model
        )


def _translate_detailed_with_tools(
    provider, text, source_lang, target_lang, model
) -> DetailedResult:
    """带工具调用循环的 detailed 翻译。"""

    system_prompt = DETAILED_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # 循环状态
    final_content = None          # 明确初始化，不依赖循环变量泄漏
    tools_used = []
    rounds_completed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # ── 工具调用循环 ─────────────────────────────────
    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        rounds_completed = round_num

        result = provider.chat_with_tools(
            messages=messages,
            tools=ALL_TOOLS,
            model=model,
        )

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        # 没有工具调用 → 模型认为信息够了，这就是最终结果
        if not result["tool_calls"]:
            final_content = result["content"]
            break

        # 有工具调用 → 把模型的回复加进历史
        messages.append({
            "role": "assistant",
            "content": result["content"],
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in result["tool_calls"]
            ],
        })

        # 逐个执行工具，把结果加进历史
        for tool_call in result["tool_calls"]:
            tool_name = tool_call["name"]

            # 参数解析也要容错——模型有可能生成不合法的 JSON
            try:
                tool_args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                tool_args = {}
                tool_result = (
                    f"参数解析失败，收到的不是合法 JSON：{tool_call['arguments']}。"
                    f"请重新调用并确保参数格式正确。"
                )
            else:
                print(f"  [第{round_num}轮·工具调用] {tool_name}({tool_args})")
                tool_result = execute_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            })

            if tool_name not in tools_used:
                tools_used.append(tool_name)

    # ── 循环结束，检查是否拿到最终结果 ──────────────────
    if final_content is None:
        raise RuntimeError(
            f"模型在 {MAX_TOOL_ROUNDS} 轮工具调用后仍未给出最终翻译结果。\n"
            f"已调用的工具：{tools_used}\n"
            f"已消耗 tokens：输入 {total_input_tokens} / 输出 {total_output_tokens}\n"
            f"可能原因：prompt 未明确要求模型在信息足够后输出最终结果，"
            f"或模型陷入了重复调用同一工具的循环。"
        )

    parsed = _parse_json(final_content)

    # 组装返回结构
    t = parsed["translation"]
    translation = TranslationLayer(
        literal=t["literal"],
        meaning=t["meaning"],
        difference=t["difference"],
    )
    sentence_structure = [
        SentenceComponent(**s) for s in parsed.get("sentence_structure", [])
    ]
    vocabulary = [
        VocabItem(**v) for v in parsed.get("vocabulary", [])
    ]

    # 存入历史
    record_id = history_store.save(
        source_lang=source_lang,
        target_lang=target_lang,
        original=text,
        translation=translation.meaning,
        mode="detailed",
        model_used=model,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        full_result=parsed,
    )
    add_history_to_index(record_id, text)

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
        tools_used=tools_used,
        tool_rounds=rounds_completed,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def _translate_detailed_fixed(provider, text, source_lang, target_lang, model) -> DetailedResult:
    """
    Phase4 的固定检索版本，保留下来作为 --no-tools 的实现。
    代码就是 Phase4 的 _translate_detailed，原样保留即可。
    """
    # ... Phase4 的实现，此处省略
    pass
```

**关键点说明**：

- `final_content = None` **在循环外明确初始化**——如果循环跑满都没拿到结果，`final_content` 仍是 `None`，触发明确的报错，而不是引用未定义变量
- 报错信息包含 `tools_used`、token 消耗、可能原因——排查问题时这些信息很关键
- `json.loads()` 也包了 try/except——模型生成的参数不一定是合法 JSON

---

## 步骤6：更新 `translate()` 和 `cli.py`

**`core.py` 的 `translate()` 加 `use_tools` 参数**：

```python
def translate(
    text: str,
    source_lang: str = "中文",
    target_lang: str = "英文",
    mode: str = "quick",
    model: str = DEFAULT_MODEL,
    use_tools: bool = True,       # 新增
) -> QuickResult | DetailedResult:

    if not source_lang:
        source_lang = "中文"
    if not target_lang:
        target_lang = "英文"

    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    if mode == "quick":
        # quick 模式不使用工具，保持 Phase4 的固定检索
        return _translate_quick(provider, text, source_lang, target_lang, model)
    elif mode == "detailed":
        return _translate_detailed(
            provider, text, source_lang, target_lang, model,
            use_tools=use_tools,
        )
    else:
        raise ValueError(f"未知模式：{mode}")
```

**`cli.py` 加 `--no-tools` 开关和工具信息展示**：

```python
def main():
    parser = argparse.ArgumentParser(description="定制化翻译工具")
    parser.add_argument("text", help="要翻译的文本")
    parser.add_argument("--from", dest="source_lang", default=None,
                        help="原文语言，默认自动检测")
    parser.add_argument("--to", default=None,
                        help="目标语言，默认自动检测（中↔英互译）")
    parser.add_argument("--mode", default="quick",
                        choices=["quick", "detailed"])
    parser.add_argument("--no-tools", action="store_true",
                        help="detailed 模式下关闭工具调用，改用固定检索（更省 token）")
    args = parser.parse_args()

    source_lang = args.source_lang or detect_source_lang(args.text)
    target_lang = args.to or ("英文" if source_lang == "中文" else "中文")

    print(f"[语言检测] {source_lang} → {target_lang}")

    try:
        result = translate(
            args.text,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=args.mode,
            use_tools=not args.no_tools,
        )
        _print_result(result)
    except RuntimeError as e:
        print(f"\n错误：{e}")
```

`_print_detailed` 里加工具信息：

```python
def _print_detailed(result: DetailedResult):
    LINE = "=" * 55
    print(f"\n{LINE}")

    if result.tools_used:
        tools_str = " + ".join(result.tools_used)
        print(f"[工具调用] {tools_str}（共 {result.tool_rounds} 轮）")
    else:
        print(f"[工具调用] 无（模型判断不需要外部信息）")

    print(f"原文（{result.source_lang}）：{result.original}")
    # ... 其余和 Phase4 相同
```

---

## 步骤7：更新 `DETAILED_SYSTEM_PROMPT`

明确告诉模型有工具可用，以及**必须在调用完工具后输出最终 JSON**（这一点如果不强调，模型可能一直调工具不给结果）：

```python
DETAILED_SYSTEM_PROMPT = """你是一个专业翻译助手兼语言教学专家。
请将用户输入的文本从{source_lang}翻译成{target_lang}，并提供完整的语言分析。

## 可用工具
- lookup_dictionary：查询词典。遇到多义词、专业术语、不确定含义的词时使用。
- search_history：检索翻译历史。需要保持术语一致性时使用。

## 工作流程
1. 判断是否需要调用工具获取额外信息。简单常见的句子可以直接翻译，不必调用。
2. 如果需要，调用相应工具。工具调用失败时不要重试同一个工具，直接基于已有信息继续。
3. **获得足够信息后，必须输出最终的 JSON 结果，不要继续调用工具。**

## 输出格式
最终结果必须是以下 JSON 格式，不要输出其他内容，不要加 markdown 代码块：

{{
  "translation": {{
    "literal": "逐字直译结果",
    "meaning": "更自然的意译结果",
    "difference": "直译和意译的区别及各自适用场合"
  }},
  "sentence_structure": [
    {{
      "component": "语法成分名称",
      "original": "原文对应片段",
      "explanation": "详细语法解释"
    }}
  ],
  "special_grammar": "特殊语法现象说明，没有则空字符串",
  "vocabulary": [
    {{
      "word": "重点词汇",
      "type": "单词/俚语/固定搭配/惯用表达",
      "meaning": "含义说明",
      "examples": ["例句1", "例句2"]
    }}
  ],
  "overall_note": "整体评注",
  "other_languages": {{}}
}}"""
```

第3条那句加粗的要求非常关键——不写这一句，模型很容易陷入"一直调工具但不给最终答案"的循环，然后触发 `MAX_TOOL_ROUNDS` 报错。

---

## 步骤8：基础验证

```bash
# 场景1：多义词 → 期待模型调用 lookup_dictionary
python -m translator.cli "他的态度很冷漠" --mode detailed

# 场景2：术语一致性 → 先翻一次，再翻相似的，期待第二次调用 search_history
python -m translator.cli "我今天很累" --mode detailed
python -m translator.cli "我今天非常疲惫" --mode detailed

# 场景3：简单句 → 期待模型不调用任何工具
python -m translator.cli "今天天气很好" --mode detailed

# 场景4：对比成本 → 同一句话开关工具各跑一次，对比 token
python -m translator.cli "他的态度很冷漠" --mode detailed
python -m translator.cli "他的态度很冷漠" --mode detailed --no-tools
```

验证清单：
- [x] 多义词场景模型主动调用 `lookup_dictionary`
- [ ] 相似句子第二次翻译时模型主动调用 `search_history` —— **未通过**，调用不稳定，见下
- [x] 简单句子模型直接翻译，显示"[工具调用] 无"
- [x] `--no-tools` 开关正常工作，退回 Phase4 行为
- [x] token 消耗对比：有工具 vs 无工具的差距有多大
- [x] `tool_rounds` 正确记录了循环轮数

---

## 验证过程中的发现与待办

### 已修复：`lookup_dictionary` 对英文输入必然失效

CC-CEDICT 的 SQLite 索引只按中文（简体/繁体）建列，英文只存在 `definitions` 列里，
而 `lookup()` 查的是 `WHERE simplified = ? OR traditional = ?`，
所以任何英文查询（连 `bus` 这种基础词）都返回"未收录"。

而原本的工具描述里写着「或一个英文单词/短语」，等于在主动诱导模型做注定失败的调用。
英译中每次都会白白多绕一轮请求。

**修复（方案A）**：工具描述改为明确声明只支持中文词条、英文原文不要调用。
实测效果：同一句英译中，输入 token 从 2212 降到 1068，**降幅 52%**，一行执行逻辑都没改。

> 这条是 Phase5 最直观的一课：**工具的"说明书"本身就是成本的一部分**。
> description 里多写一句话，就能让模型持续做无效调用。

### 待第二轮设计：`search_history` 的定位需要重新想

**现象**：模型几乎不主动调用 `search_history`。日常口语句（"我今天很累"）不调，
甚至历史里存着高度相似记录的句子（"他说话总是打太极"）也不调。

**表层原因**：原描述强调"术语一致性"，而测试用的是没有任何术语的口语句，
模型判断"不需要"——从描述字面看这个判断是对的。

**深层原因（更值得记住）**：检索类工具天然会被模型低估。
查词典是"我知道这个词我拿不准"，有明确的未知；
查历史是"我不知道以前有没有翻过"，**模型无法预判收益**，而代价是确定的一整轮请求。
收益不确定、成本确定，模型倾向跳过。

**顺带算出的一笔账**：`search_history` 全部在本地跑（sentence-transformers + ChromaDB），
不花一分 API 费用；但"让模型决定要不要查历史"本身要消耗一整轮请求（约 1000+ 输入 token）。
**用付费的 token 去决定要不要做一件免费的事，经济上是亏的。**

**更根本的设计质疑（第二轮重点）**：句子级"相似"检索本身可能就是个错误的抽象。

```
我今天很累     → I'm very tired today.
我今天非常疲惫  → I'm extremely tired today.     相似度 0.982
```

两句的全部差异就在"很 / 非常"上，而这正是必须译出区别的地方。
把前者当参考注入，模型容易被锚定，反而抹平强度差异。相似度越高越危险——差异全在细微处。
当前阈值 `HISTORY_SIMILARITY_THRESHOLD = 0.85` 更是相当宽松。

这个问题在 CAT 工具（Trados、memoQ）里早有成熟答案：**把"翻译记忆"和"术语库"拆成两件事**。

| | 翻译记忆库 TM | 术语库 Glossary |
|---|---|---|
| 存什么 | 整句：原文 ↔ 译文 | 词条：Stable LatentMoE → 稳定潜在专家混合 |
| 何时有用 | 只有高度匹配才有价值（业界通常 95%+），用于重复率高的文档 | 任何时候，跨句子、跨文档 |
| 模糊匹配怎么办 | 标注相似度**给人看**，由人判断，不自动采用 | 不适用 |

**真正需要保持一致的是术语，不是句子。** 同一份技术文档里 `Stable LatentMoE` 出现 20 次
必须译法统一，这价值巨大；而"我今天很累"和"我今天非常疲惫"译得像不像，那不叫一致性，叫串味。

**第二轮的设计方向**：
1. 新增**术语库**：从历史翻译中抽取专有名词/技术术语单独建表，翻译时匹配术语注入 prompt。
   这才是一致性的正确载体，也和开发计划里"老词库 + 定制 prompt"的思路一脉相承。
2. **句子级历史提高门槛**：阈值提到 0.95+，只处理"几乎重复"的情况（同一份文档翻两遍），
   并在 prompt 里明确标注"仅供参考，语境不同请自行判断"。
3. **重新定位历史记录的价值**：它的主要用途可能本来就不是喂给模型，
   而是**成本统计**和**用户回看**——这两个用途已经在用，而且很扎实。

**本轮暂时处理（方案A）**：只修正工具描述，让它准确说明"什么时候查历史才有价值"
（专有名词/术语/同批文档 → 查；日常口语 → 不查，并说明原因是怕抹掉细微差异）。
不改检索逻辑，留给第二轮。

**方案A 的实测结果：没能改变行为。** 改描述前后，模型对口语句、习语句、
含技术术语的同批文档句一律不调用（唯一例外是偶尔会调，见下条"随机性"）。
说明这不是描述写得好不好的问题，而是架构层面的问题——这反而让第二轮的判断更有底气。

### 成本结论（Phase5 最重要的实证）

**1. 成本结构是三段式的**

```
基础翻译（--no-tools）              ≈  570 tokens
+ 开启工具的"入场费"（schema+说明）  ≈  550 tokens   ← 不管用不用都付
+ 每多绕一轮工具调用                ≈ 1350 tokens   ← 整个消息历史重发一遍
```

实测倍数：触发工具的句子约 **4.3 倍**，完全没触发工具的句子也要 **2 倍**。

**2. 工具 schema 每轮都要重发，是笔容易被忽略的固定开销**

`ALL_TOOLS` 序列化后约 471 tokens（lookup_dictionary 210 + search_history 260）。
两轮循环就是 942 tokens——占某次实测总输入的 **38%**，纯粹是在反复告诉模型"你有什么工具"。
模型无状态，第二轮哪怕已经用过工具，也得再收一遍完整菜单。

由此得出一条写工具描述的准则：**描述里每加一句话，都要问"它省下的比它花掉的多吗"**。
修英译中那次是正面案例：加约 50 token/轮的说明，换来省掉一整轮（约 1144 token），净赚。
反过来，为追求"描述完美"塞进几百 token 的细则、而模型本来判断就没错，就是纯亏。

**3. 最贵的不是执行工具，是多一轮请求**

工具执行全在本地（SQLite + ChromaDB + 本地 embedding 模型），一分 API 费用不花。
贵的是为了把工具结果告诉模型，必须把之前所有对话重发一遍。

**4. 输入 token 可预测，输出 token 不可预测**

| | 可预测性 |
|---|---|
| 输入 | **可预测**：prompt 长度、schema 大小、轮数都能事先算出来 |
| 输出 | **不可预测**：同一句话重复跑，实测波动 1.9～2.8 倍 |

输出波动的成因按影响排序：句子本身值不值得展开讲（主因）> 采样随机性 > 开工具会多出生成工具调用的部分。

**实践含义**：做成本控制时能优化的只有输入侧——精简 prompt、砍掉没用的工具、减少循环轮数。
输出侧只能靠 prompt 约束（如限定 vocabulary 最多3条、例句最多2条），但无法保证。
做预算时输出侧应按"平均值 × 安全系数"估，不要按最好情况算。

**5. 实验1 结论：description 的作用是"授权调用"，不是"防止乱调"**

把 `lookup_dictionary` 的描述改成极简的 `"查词典"`（schema 从 210 tokens 降到 58），
结果与本文档步骤9 的原始假设**相反**：模型没有开始乱调，而是**彻底不调了**。

| 描述版本 | schema | 习语句"打太极"是否调用 | 样本 |
|---|---|---|---|
| 详细版 | 210 tokens | 调用 | 5/5 |
| 极简版「查词典」 | 58 tokens | 不调用 | 0/4 |
| 改回详细版 | 210 tokens | 调用 | 1/1 |

改回去行为立即恢复，变量可控、结果可复现。

**原因**：`"查词典"` 只说明了工具**是什么**，没说明**什么时候用它值得**。
模型面对收益不明的工具，默认选择相信自己的知识。这不是判断失误，是缺乏调用的理由。

> 记法要反过来：工具描述的主要作用是**授权调用**——
> 告诉模型"在这些情况下，花一轮请求来调我是划算的"。

**这种失效是静默的**：不报错、不超支、译文照样能看，
你不会发现工具从来没被触发过，却还在为它每轮付 schema 的钱。
相比之下"乱调工具"至少会体现在账单上，反而容易察觉。

**三方对比后的关键结论：描述糟糕的工具，比不加这个工具还差。**

| 方案 | 输入 tokens | 实际拿到的参考信息 |
|---|---|---|
| `--no-tools` 固定检索 | 576 | 词典 + 历史（程序强制注入） |
| 极简描述 | 1009 | **什么都没有**（工具从不触发） |
| 详细描述 | 2335 | 词典结果（模型按需取用） |

极简版比 `--no-tools` 多花 75% 的钱，拿到的信息却是零——
既没有固定检索的可靠性，也没有按需检索的能力，纯粹在为一个永不触发的工具付广告费。

**6. 实验3 结论：容错设计需要代码和 prompt 配套，缺一不可**

把 `DICT_DB_PATH` 指向不存在的目录，制造真实的工具执行故障，端到端跑一次：

| 观察点 | 结果 |
|---|---|
| 程序是否崩溃 | 没崩。`sqlite3.OperationalError` 被 `executor.execute()` 捕获，转成文本返回 |
| 模型是重试还是继续 | 继续。只调用一次，没有重试 |
| 最终结果是否可用 | 完全可用，译文与词典正常时完全一致，各部分分析都完整 |

工具返回的那句话是**两段式**的：

```
工具 lookup_dictionary 执行失败：OperationalError: unable to open database file。
请在没有这个工具结果的情况下继续完成翻译。
```

前半句是诊断信息（给开发者排查用），后半句是行动指令（给模型用）。

**关键结论：代码层的容错和 prompt 层的容错必须配套。**
- 只有代码兜底、没在 prompt 里写"失败不要重试"：模型可能反复调用直到撞满 `MAX_TOOL_ROUNDS`
- 只有 prompt 规则、代码却往上抛异常：程序在模型有机会读到规则之前就已经崩了

**7. 模型的工具调用决策带随机性，成本因此不可预测**

同一句"他说话总是打太极"，多次运行中有时调用 `search_history`、有时不调。
后果是同一句话可能花 1123 token，也可能花 2467 token，事先无法预知。
对需要控制成本的项目来说，这是纯模型决策方案的一个真实代价。

---

## 步骤9：进阶实验（本阶段最有价值的部分）

Phase5 真正的学习点不是"跑通代码"，而是**观察模型如何决策**。做完基础验证后，试试这几个实验：

### 实验1：description 的影响

把 `LOOKUP_DICTIONARY` 的 description 改成模糊版本：

```python
"description": "查词典"     # 极简版
```

然后重跑场景1和场景3，观察：
- 模型是否还能准确判断何时该调用？
- 是否开始对简单词也调用工具（浪费 token）？

改回详细版本再跑一次，对比差异。**这是 prompt engineering 最直观的一课**。

### 实验2：工具数量的影响

临时把 `ALL_TOOLS` 改成只有一个工具：

```python
ALL_TOOLS = [LOOKUP_DICTIONARY]     # 去掉 SEARCH_HISTORY
```

观察模型的行为变化——工具变少后，它是否更倾向于调用剩下的这个？

### 实验3：故意制造工具失败

临时把 `DICT_DB_PATH` 改成一个不存在的路径，触发工具执行失败，观察：
- 程序是否正常处理了失败（不崩溃）？
- 模型收到错误信息后，是重试还是继续翻译？
- 最终结果是否仍然可用？

这个实验验证的是**容错设计是否真的有效**。

### 实验4：生僻词测试

```bash
python -m translator.cli "这个人很龟毛" --mode detailed
python -m translator.cli "他说话总是打太极" --mode detailed
```

俚语和方言表达，观察模型是否主动查词典，以及查到的结果是否改善了翻译质量。

---

## 步骤10：提交代码与更新文档

```bash
git add .
git commit -m "阶段5: Tool Use工具调用，模型自主决策，容错处理，--no-tools开关"
```

然后更新 `docs/翻译软件开发计划.md`，补充完成记录和实验观察结果。

---

## Phase5 学习重点

| 知识点 | 在哪里体现 |
|---|---|
| Tool Schema 设计 | `definitions.py`：description 里写正面+反面使用场景 |
| 工具调用循环 | `core.py`：循环 + messages 历史维护 + 明确退出条件 |
| 循环状态管理 | `final_content = None` 初始化，避免变量泄漏和静默失败 |
| 容错设计 | `executor.py` 永不抛异常；JSON 参数解析也包 try/except |
| 接口拆分 | `chat()` vs `chat_with_tools()`，避免参数互斥的怪异设计 |
| 装饰器复用 | `@handle_api_errors` 抽取共用异常处理 |
| 可观测性 | `tools_used` + `tool_rounds` + 每轮调用打印 |
| 成本意识 | 消息历史累积增长；`--no-tools` 开关；quick 模式不启用工具 |

---

## Phase5 → Phase6 的升级方向

Phase5 的工具是**代码里的普通函数**——工具在哪执行、谁执行，完全由你的程序控制。

Phase6 引入 **MCP（Model Context Protocol）**：把这些工具封装成独立的 **MCP Server**，翻译软件作为 **MCP Client** 调用。

好处是工具变成了**可独立运行、独立维护、可被其他程序复用的服务**。比如你的词典 MCP Server 建好之后，Claude Code、Claude Desktop 或者别人的程序都能直接接入使用——这就是企业级 Agent 系统里工具层的标准设计方式。
