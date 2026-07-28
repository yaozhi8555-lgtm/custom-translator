# 阶段1：核心直译功能

## 目标
跑通最小链路：`CLI 输入 → 读取配置 → 调用 OpenRouter → 打印译文 + 消耗的 token`

先不做语法分析、JSON 结构化输出（那是阶段2），先把最基础的"调用能通、代码结构对"打扎实。

---

## 步骤1：文件结构

```
src/translator/
├── __init__.py
├── config.py          # 读取.env、管理默认模型等配置
├── providers/
│   ├── __init__.py
│   ├── base.py        # 抽象基类，定义所有provider必须实现的接口
│   └── openrouter.py  # OpenRouter具体实现
├── core.py            # 核心翻译逻辑，对外暴露的translate()函数
└── cli.py             # 命令行入口
```

**为什么这么分**：`providers/` 单独一个包，是为了以后阶段3加别的 provider 时，只需新增一个文件，不用改 `core.py` 里调用它的代码。这是"面向接口编程"的练习。

---

## 步骤2：Provider 抽象接口（`base.py`）

```python
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_message: str, model: str) -> dict:
        """
        调用模型，返回统一格式：
        {
            "content": "模型返回的文本",
            "input_tokens": int,
            "output_tokens": int
        }
        """
        pass
```

关键点：不管底层是 OpenAI 格式还是 Anthropic 格式，返回给 `core.py` 的永远是这个统一的 dict 结构。以后换 provider，`core.py` 完全不用改。

---

## 步骤3：OpenRouter Provider 实现（`openrouter.py`）

```python
from openai import OpenAI
from .base import BaseProvider

class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def chat(self, system_prompt: str, user_message: str, model: str) -> dict:
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
```

要点：
- `response.usage` 是成本控制的起点，从这里开始就把 token 数据取出来，阶段8做统计时直接有数据可用
- 异常处理先留白，等实际跑起来遇到第一个真实报错（API key 错误、网络超时等）再针对性补上

---

## 步骤4：配置管理（`config.py`）

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"  # 后面可做成可配置

if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY，请检查 .env 文件")
```

提醒：去 OpenRouter 官网确认当前可用模型名称和价格，不同模型价格差异较大。

---

## 步骤5：核心翻译函数（`core.py`）

```python
from .config import OPENROUTER_API_KEY, DEFAULT_MODEL
from .providers.openrouter import OpenRouterProvider

SYSTEM_PROMPT_QUICK = """你是一个专业翻译助手。
请将用户输入的文本准确翻译成{target_lang}。
只输出译文本身，不要添加任何解释、说明或额外内容。"""

def translate(text: str, target_lang: str = "英文", model: str = DEFAULT_MODEL) -> dict:
    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)
    system_prompt = SYSTEM_PROMPT_QUICK.format(target_lang=target_lang)
    result = provider.chat(system_prompt=system_prompt, user_message=text, model=model)
    return result
```

要点：system prompt 故意设计得很朴素，只要求纯译文，不掺任何解析——先验证调用链路通不通，避免一开始就分不清是 prompt 问题还是代码问题。

---

## 步骤6：CLI 入口（`cli.py`）

```python
import argparse
from .core import translate

def main():
    parser = argparse.ArgumentParser(description="定制化翻译工具")
    parser.add_argument("text", help="要翻译的文本")
    parser.add_argument("--to", default="英文", help="目标语言，默认英文")
    args = parser.parse_args()

    result = translate(args.text, target_lang=args.to)

    print(f"\n译文: {result['content']}")
    print(f"\n(消耗 tokens: 输入 {result['input_tokens']} / 输出 {result['output_tokens']})")

if __name__ == "__main__":
    main()
```

运行方式：
```bash
python -m translator.cli "我今天很累" --to 英文
```

---

## 步骤7：手动验证清单

- [ ] 输入一句中文，能正确收到英文译文
- [ ] 输入一句英文，`--to 中文` 能正确收到中文译文
- [ ] 打印出来的 token 数量是否合理
- [ ] 故意输错 API key，观察程序抛出什么样的报错（先记下来，不用管，步骤8处理）

---

## 步骤8：补异常处理

根据步骤7实际遇到的报错类型，回来给 `chat()` 方法加 try/except——针对 API key 错误、网络超时、rate limit 超限等，分别给出对应的用户提示。

**建议先跑出真实报错再处理**，这样能对症下药，而不是提前假设一堆可能的异常类型。

---

## 步骤9：提交代码

```bash
git add .
git commit -m "阶段1: 核心直译功能，OpenRouter provider抽象层"
```
