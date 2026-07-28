"""custom-translator 入口文件 —— 验证环境可用"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("Openrouter_API_KEY"),
)


def main():
    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "用中文回复：Hello, how are you?"}
        ],
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()