import argparse
from .core import translate


def main():
    parser = argparse.ArgumentParser(description="定制化翻译工具")
    parser.add_argument("text", help="要翻译的文本")
    parser.add_argument("--to", default="英文", help="目标语言，默认英文")
    args = parser.parse_args()

    try:
        result = translate(args.text, target_lang=args.to)
        print(f"\n译文: {result['content']}")
        print(f"\n(消耗 tokens: 输入 {result['input_tokens']} / 输出 {result['output_tokens']})")

    except RuntimeError as e:
        print(f"\n错误：{e}")


if __name__ == "__main__":
    main()