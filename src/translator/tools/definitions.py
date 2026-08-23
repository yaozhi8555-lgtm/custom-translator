"""
工具的 JSON Schema 定义。

这里不写任何执行逻辑，只写"说明书"——告诉模型每个工具叫什么、能干什么、要传什么参数。
模型完全靠这份说明书来判断"我现在该不该调用这个工具"，所以 description 写得好不好，
直接决定了模型会不会乱调工具（乱调 = 白花 token）。

格式遵循 OpenAI function calling 标准，OpenRouter 兼容这套格式。
"""

LOOKUP_DICTIONARY = {
    "type": "function",
    "function": {
        "name": "lookup_dictionary",
        "description": (
            "在 CC-CEDICT 词典中查找【中文词条】的准确含义。"
            "使用场景：遇到多义词、专业术语、不确定含义的中文词，"
            "或者需要确认某个中文词在特定语境下的准确释义时。"
            "不要对简单常用词（如'我''今天''好'）调用此工具。"
            "【重要】此词典只按中文词条建立索引，无法查询英文单词或短语。"
            "翻译英文原文时不要调用此工具，调用了也一定查不到结果。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "要查询的中文词语，通常是2-4个汉字。不接受英文输入。",
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
            "在翻译历史中检索语义相似的句子，看看同类内容以前是怎么翻的。"
            "使用场景：原文含专有名词、产品名、技术术语、机构名等需要固定译法的内容时；"
            "或原文看起来是某份文档、某个系列的一部分，很可能和之前翻译过的内容同批时。"
            "不要对日常口语、一次性的普通短句调用——这类句子即使找到相似历史，"
            "也容易把上次的译法生搬过来，反而抹掉当前句子的细微差异。"
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

# 发给模型的完整工具清单
ALL_TOOLS = [LOOKUP_DICTIONARY, SEARCH_HISTORY]
