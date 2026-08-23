"""
工具执行器。

definitions.py 是"说明书"，这里是"真正干活的人"：
模型说要调用某个工具，程序在这里真的去查词典、查历史，把结果变成一段文本还给模型。

核心设计原则：这里的 execute() 永远不抛异常。
工具失败不应该让整个翻译流程崩掉，而是把错误变成一句给模型看的说明，
让模型自己决定"没有这个信息我还能不能继续翻译"。
"""

from ..config import DICT_DB_PATH
from ..dictionary.store import lookup
from ..history.retriever import retrieve_similar


def execute(tool_name: str, tool_args: dict) -> str:
    """
    执行工具，返回字符串结果（这段字符串会作为一条消息发回给模型）。

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
            # 模型有可能凭空编出一个不存在的工具名
            return f"错误：不存在名为 {tool_name} 的工具。请使用已提供的工具。"

    except Exception as e:
        # 任何异常都转成给模型看的说明，不向上抛
        return (
            f"工具 {tool_name} 执行失败：{type(e).__name__}: {e}。"
            f"请在没有这个工具结果的情况下继续完成翻译。"
        )


def _execute_lookup_dictionary(args: dict) -> str:
    """查词典。模型已经决定了查哪个词，直接精确查找，不需要 Phase3 的滑动窗口。"""
    word = args.get("word", "").strip()
    if not word:
        return "参数错误：word 不能为空"

    entries = lookup(str(DICT_DB_PATH), word)

    if not entries:
        # 空结果也要给模型有用的信号，而不是干巴巴一句"没找到"
        return f"词典中未收录「{word}」。这可能是新词、专有名词或音译词。"

    lines = [f"词典查询结果（{word}）："]
    for entry in entries:
        defs = "; ".join(entry.definitions)
        lines.append(f"  {entry.simplified}: {defs}")
    return "\n".join(lines)


def _execute_search_history(args: dict) -> str:
    """检索语义相似的历史翻译，帮助模型保持术语一致性。"""
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
