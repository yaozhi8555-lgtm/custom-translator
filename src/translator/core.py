import json
from json_repair import repair_json
from .config import OPENROUTER_API_KEY, DEFAULT_MODEL
from .providers.openrouter import OpenRouterProvider
from .schemas import (
    QuickResult, DetailedResult,
    SentenceComponent, VocabItem
)
from .prompts import (
    QUICK_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    DETAILED_TOOLS_SYSTEM_PROMPT,
    inject_dict_context
)
from .dictionary.retriever import retrieve, format_for_prompt
from .history import store as history_store
from .history.retriever import (
    retrieve_similar as retrieve_history,
    add_to_index as add_history_to_index,
    format_for_prompt as history_format_for_prompt,
)
from .tools.definitions import ALL_TOOLS
from .tools.executor import execute as execute_tool

# 程序启动时初始化历史数据库
history_store.init_db()

# 工具调用最多循环几轮，防止模型陷入"一直查工具就是不给答案"的死循环
MAX_TOOL_ROUNDS = 5


def translate(
    text: str,
    source_lang: str = "中文",
    target_lang: str = "英文",
    mode: str = "quick",
    model: str = DEFAULT_MODEL,
    use_tools: bool = True,
) -> QuickResult | DetailedResult:

    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    if mode == "quick":
        # quick 模式不启用工具，保持 Phase4 的固定检索：追求快和省，一次请求就够
        return _translate_quick(provider, text, source_lang, target_lang, model)
    elif mode == "detailed":
        return _translate_detailed(
            provider, text, source_lang, target_lang, model,
            use_tools=use_tools,
        )
    else:
        raise ValueError(f"未知模式：{mode}，请使用 quick 或 detailed")


def _translate_quick(provider, text, source_lang, target_lang, model) -> QuickResult:
    # 1. 检索词典
    dict_entries = retrieve(text)
    dict_context = format_for_prompt(dict_entries)

    # 2. 检索历史（新增）
    history_records = retrieve_history(text, target_lang)
    history_context = history_format_for_prompt(history_records)

    # 3. 组装 prompt（词典上下文 + 历史上下文）
    base_prompt = QUICK_SYSTEM_PROMPT.format(target_lang=target_lang)
    system_prompt = inject_dict_context(base_prompt, dict_context)
    system_prompt = inject_dict_context(system_prompt, history_context)

    result = provider.chat(
        system_prompt=system_prompt,
        user_message=text,
        model=model
    )
    quick_result = QuickResult(
        original=text,
        translation=result["content"],
        source_lang=source_lang,
        target_lang=target_lang,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )

    # 4. 存入历史（新增）
    record_id = history_store.save(
        source_lang=source_lang,
        target_lang=target_lang,
        original=text,
        translation=result["content"],
        mode="quick",
        model_used=model,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )
    add_history_to_index(record_id, text)

    return quick_result


def _translate_detailed(
    provider, text, source_lang, target_lang, model,
    use_tools: bool = True,
) -> DetailedResult:
    """
    detailed 模式的分发入口。

    use_tools=True  → Phase5 的工具调用循环，模型自己决定查什么
    use_tools=False → Phase4 的固定检索，程序写死查词典+查历史，成本更低
    """
    if use_tools:
        return _translate_detailed_with_tools(
            provider, text, source_lang, target_lang, model
        )
    else:
        return _translate_detailed_fixed(
            provider, text, source_lang, target_lang, model
        )


def _translate_detailed_fixed(provider, text, source_lang, target_lang, model) -> DetailedResult:
    """Phase4 的固定检索实现，原样保留，作为 --no-tools 的实现。"""
    # 1. 检索词典
    dict_entries = retrieve(text)
    dict_context = format_for_prompt(dict_entries)

    # 2. 检索历史（新增）
    history_records = retrieve_history(text, target_lang)
    history_context = history_format_for_prompt(history_records)

    # 3. 组装 prompt（词典上下文 + 历史上下文）
    base_prompt = DETAILED_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )
    system_prompt = inject_dict_context(base_prompt, dict_context)
    system_prompt = inject_dict_context(system_prompt, history_context)

    result = provider.chat(
        system_prompt=system_prompt,
        user_message=text,
        model=model
    )

    parsed = _parse_json(result["content"])

    return _build_and_save_detailed(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        model=model,
        parsed=parsed,
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )


def _translate_detailed_with_tools(provider, text, source_lang, target_lang, model) -> DetailedResult:
    """
    Phase5：带工具调用循环的 detailed 翻译。

    和固定检索最大的区别是：这里不预先查任何东西，
    只把"你有这些工具"告诉模型，让它自己决定查不查、查什么、查几次。
    """
    system_prompt = DETAILED_TOOLS_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )

    # messages 是一个会不断变长的列表：模型无状态，每轮都要把完整对话重发一遍
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # 循环状态：在循环外明确初始化，不依赖循环内的变量泄漏
    final_content = None
    tools_used = []
    rounds_completed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for round_num in range(1, MAX_TOOL_ROUNDS + 1):
        rounds_completed = round_num

        result = provider.chat_with_tools(
            messages=messages,
            tools=ALL_TOOLS,
            model=model,
        )

        # token 要累加，不是覆盖——每一轮都是一次真实的计费请求
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        # 没有工具调用 → 模型认为信息够了，这就是最终答案，退出循环
        if not result["tool_calls"]:
            final_content = result["content"]
            break

        # 有工具调用 → 先把模型这轮的回复原样追加进 messages，
        # 否则下一轮模型会看不到自己刚才说过要调用工具，对不上号
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

        # 逐个执行工具，把结果也追加进 messages
        for tool_call in result["tool_calls"]:
            tool_name = tool_call["name"]

            # 模型给的 arguments 是 JSON 字符串，不保证合法，解析也要容错
            try:
                tool_args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                tool_result = (
                    f"参数解析失败，收到的不是合法 JSON：{tool_call['arguments']}。"
                    f"请重新调用并确保参数格式正确。"
                )
            else:
                print(f"  [第{round_num}轮·工具调用] {tool_name}({tool_args})")
                tool_result = execute_tool(tool_name, tool_args)

            # tool 角色的消息必须带 tool_call_id，模型靠它把结果和自己的请求对应起来
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result,
            })

            if tool_name not in tools_used:
                tools_used.append(tool_name)

    # 循环跑满都没拿到最终结果，说明模型陷在工具调用里出不来了
    if final_content is None:
        raise RuntimeError(
            f"模型在 {MAX_TOOL_ROUNDS} 轮工具调用后仍未给出最终翻译结果。\n"
            f"已调用的工具：{tools_used}\n"
            f"已消耗 tokens：输入 {total_input_tokens} / 输出 {total_output_tokens}\n"
            f"可能原因：prompt 未明确要求模型在信息足够后输出最终结果，"
            f"或模型陷入了重复调用同一工具的循环。"
        )

    parsed = _parse_json(final_content)

    return _build_and_save_detailed(
        text=text,
        source_lang=source_lang,
        target_lang=target_lang,
        model=model,
        parsed=parsed,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        tools_used=tools_used,
        tool_rounds=rounds_completed,
    )


def _build_and_save_detailed(
    text, source_lang, target_lang, model, parsed,
    input_tokens, output_tokens,
    tools_used=None, tool_rounds=0,
) -> DetailedResult:
    """
    把解析好的 JSON 组装成 DetailedResult 并存入历史。

    抽出来是因为固定检索和工具调用两条路径走到这一步之后要做的事完全一样，
    区别只在于工具模式会多带 tools_used / tool_rounds 两个可观测性字段。
    """
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

    detailed_result = DetailedResult(
        original=text,
        source_lang=source_lang,
        target_lang=target_lang,
        translation=parsed["translation"],
        sentence_structure=sentence_structure,
        special_grammar=parsed.get("special_grammar", ""),
        vocabulary=vocabulary,
        overall_note=parsed.get("overall_note", ""),
        other_languages=parsed.get("other_languages", {}),
        tools_used=tools_used or [],
        tool_rounds=tool_rounds,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # 存入历史
    record_id = history_store.save(
        source_lang=source_lang,
        target_lang=target_lang,
        original=text,
        translation=parsed["translation"],
        mode="detailed",
        model_used=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        full_result=parsed,  # detailed模式存完整JSON备查
    )
    add_history_to_index(record_id, text)

    return detailed_result


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
    except json.JSONDecodeError:
        # 严格解析失败时，用 json_repair 兜底修复常见的格式错误
        # （比如字符串内嵌未转义的引号、缺逗号等），修复失败会返回 {}
        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
        raise RuntimeError(
            f"模型返回的内容不是有效的 JSON 格式，自动修复也失败了。\n"
            f"原始内容：\n{content}"
        )