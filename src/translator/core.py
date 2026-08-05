import json
from .config import OPENROUTER_API_KEY, DEFAULT_MODEL
from .providers.openrouter import OpenRouterProvider
from .schemas import (
    QuickResult, DetailedResult,
    SentenceComponent, VocabItem
)
from .prompts import (
    QUICK_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    inject_dict_context
)
from .dictionary.retriever import retrieve, format_for_prompt


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
    # 查词典，注入上下文
    dict_entries = retrieve(text)
    dict_context = format_for_prompt(dict_entries)
    base_prompt = QUICK_SYSTEM_PROMPT.format(target_lang=target_lang)
    system_prompt = inject_dict_context(base_prompt, dict_context)

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
    # 查词典，注入上下文
    dict_entries = retrieve(text)
    dict_context = format_for_prompt(dict_entries)
    base_prompt = DETAILED_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )
    system_prompt = inject_dict_context(base_prompt, dict_context)

    result = provider.chat(
        system_prompt=system_prompt,
        user_message=text,
        model=model
    )

    parsed = _parse_json(result["content"])

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
        translation=parsed["translation"],
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