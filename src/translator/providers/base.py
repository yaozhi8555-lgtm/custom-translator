from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_message: str, model: str) -> dict:
        """
        单轮对话，不带工具。Phase1-4 的用法。

        返回统一格式：
        {
            "content": "模型返回的文本",
            "input_tokens": int,
            "output_tokens": int
        }
        """
        pass

    @abstractmethod
    def chat_with_tools(self, messages: list, tools: list, model: str) -> dict:
        """
        带工具的多轮对话。Phase5 的用法。

        messages: 完整的消息历史列表，由调用方（core.py 的循环）负责维护和累加
        tools:    工具 schema 列表，来自 tools/definitions.py 的 ALL_TOOLS

        返回统一格式：
        {
            "content": str | None,        # 模型要调用工具时，这里通常是 None
            "tool_calls": list | None,    # 模型不调用工具时，这里是 None
            "input_tokens": int,
            "output_tokens": int
        }

        content 和 tool_calls 是"二选一"的关系，正是这一点驱动了 core.py 里循环的分叉：
        有 tool_calls 就执行工具再来一轮，没有就说明模型给出最终答案了。
        """
        pass
