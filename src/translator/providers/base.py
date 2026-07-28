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