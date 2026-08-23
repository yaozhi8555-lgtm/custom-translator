import argparse
from .core import translate
from .schemas import QuickResult, DetailedResult
import re

def main():
    parser = argparse.ArgumentParser(description="定制化翻译工具")
    parser.add_argument("text", help="要翻译的文本")
    parser.add_argument("--from", dest="source_lang", default=None, help="原文语言，默认中文")
    parser.add_argument("--to", default=None, help="目标语言，默认自动检测（中文↔英文互译）")
    parser.add_argument(
        "--mode",
        default="quick",
        choices=["quick", "detailed"],
        help="翻译模式：quick（仅直译）或 detailed（完整分析），默认 quick"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="detailed 模式下关闭工具调用，改用固定检索（更省 token）"
    )
    args = parser.parse_args()

   # 自动检测原文语言
    source_lang = args.source_lang or detect_source_lang(args.text)

    # 自动推断目标语言（没指定就反向互译）
    if args.to:
        target_lang = args.to
    else:
        target_lang = "英文" if source_lang == "中文" else "中文"

    try:
        result = translate(
            args.text,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=args.mode,
            use_tools=not args.no_tools,   # argparse 把 --no-tools 存成 True，含义要反过来
        )
        _print_result(result)
    except RuntimeError as e:
        print(f"\n错误：{e}")


def _print_result(result):
    if isinstance(result, QuickResult):
        _print_quick(result)
    elif isinstance(result, DetailedResult):
        _print_detailed(result)


def _print_quick(result: QuickResult):
    print(f"\n译文：{result.translation}")
    print(f"(消耗 tokens：输入 {result.input_tokens} / 输出 {result.output_tokens})")


def _print_detailed(result: DetailedResult):
    LINE = "=" * 55

    # Phase5：把模型在循环里干了什么显示出来，否则这部分完全是黑箱
    if result.tools_used:
        tools_str = " + ".join(result.tools_used)
        print(f"\n[工具调用] {tools_str}（共 {result.tool_rounds} 轮）")
    elif result.tool_rounds:
        print(f"\n[工具调用] 无（模型判断不需要外部信息）")

    print(f"\n原文（{result.source_lang}）：{result.original}")
    print(f"\n【译文】{result.translation}")

    print(f"\n--- 句子结构 ---")
    for s in result.sentence_structure:
        print(f"  [{s.component}]  {s.original}")
        print(f"    └─ {s.explanation}")

    if result.special_grammar:
        print(f"\n--- 特殊语法 ---")
        print(f"  {result.special_grammar}")

    if result.vocabulary:
        print(f"\n--- 重点词汇 / 俚语 / 用法 ---")
        for v in result.vocabulary:
            print(f"\n  {v.word}（{v.type}）")
            print(f"  含义：{v.meaning}")
            for ex in v.examples:
                print(f"    · {ex}")

    print(f"\n--- 整体评注 ---")
    print(f"  {result.overall_note}")

    if result.other_languages:
        print(f"\n--- 其他语言参考 ---")
        for lang, trans in result.other_languages.items():
            print(f"  {lang}：{trans}")

    print(f"\n(消耗 tokens：输入 {result.input_tokens} / 输出 {result.output_tokens})")
    print(f"{LINE}\n")

def detect_source_lang(text: str) -> str:
    """
    简单判断：文本里有中文字符就是中文，否则是英文。
    不需要第三方库，覆盖日常中英互译场景足够用。
    """
    chinese_chars = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
    return "中文" if chinese_chars.search(text) else "英文"


if __name__ == "__main__":
    main()