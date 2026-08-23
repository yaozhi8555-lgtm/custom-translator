import functools

import openai
from openai import OpenAI

from .base import BaseProvider


def handle_api_errors(func):
    """
    装饰器：把 openai 的原始异常转成人话 RuntimeError。

    Phase1 时这段 try/except 是直接写在 chat() 里的，现在多了一个 chat_with_tools()，
    两个方法要处理的异常完全一样。抽成装饰器后，加一行 @handle_api_errors 就能复用，
    不用把同样的 try/except 抄两遍。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)

        except openai.AuthenticationError:
            raise RuntimeError("API Key 错误，请检查 .env 文件里的 OPENROUTER_API_KEY 是否正确")

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
        """单轮对话，Phase1-4 的实现，逻辑保持不变，只是异常处理挪到装饰器里了。"""
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
        """带工具的多轮对话。messages 由调用方维护，这里只负责发出去、把结果翻译成统一格式。"""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        message = response.choices[0].message

        # 把 openai SDK 的对象转成普通 dict，避免 core.py 依赖 SDK 的内部类型
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,   # 注意：这是 JSON 字符串，不是 dict
                }
                for tc in message.tool_calls
            ]

        return {
            "content": message.content,
            "tool_calls": tool_calls,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
