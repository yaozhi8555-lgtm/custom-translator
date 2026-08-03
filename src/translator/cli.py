import argparse
from .core import translate
from .schemas import QuickResult, DetailedResult


def main():
    parser = argparse.ArgumentParser(description="定制化翻译工具")
    parser.add_argument("text", help="要翻译的文本")
    parser.add_argument("--from", dest="source_lang", default="中文", help="原文语言，默认中文")
    parser.add_argument("--to", default="英文", help="目标语言，默认英文")
    parser.add_argument(
        "--mode",
        default="quick",
        choices=["quick", "detailed"],
        help="翻译模式：quick（仅直译）或 detailed（完整分析），默认 quick"
    )
    args = parser.parse_args()

    try:
        result = translate(
            args.text,
            source_lang=args.source_lang,
            target_lang=args.to,
            mode=args.mode,
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


if __name__ == "__main__":
    main()