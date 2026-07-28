import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"  # 后面可做成可配置

if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY，请检查 .env 文件")