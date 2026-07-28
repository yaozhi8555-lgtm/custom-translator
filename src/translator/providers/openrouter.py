import openai
from openai import OpenAI
from .base import BaseProvider

class OpenRouterProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def chat(self, system_prompt: str, user_message: str, model: str) -> dict:
        try:
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

        except openai.AuthenticationError:
            raise RuntimeError("API Key 错误，请检查 .env 文件里的 OPENROUTER_API_KEY 是否正确")

        except openai.RateLimitError:
            raise RuntimeError("请求太频繁或余额不足，请稍后再试")

        except openai.APIConnectionError:
            raise RuntimeError("网络连接失败，请检查网络后重试")

        except openai.APIStatusError as e:
            raise RuntimeError(f"OpenRouter 返回错误：{e.status_code} - {e.message}")