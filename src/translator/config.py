import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 原有配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"  # 后面可做成可配置

if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY，请检查 .env 文件")

# 新增：词典配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录
DICT_PATH = BASE_DIR / "data" / "cedict_ts.u8"           # 词典文件路径
DICT_DB_PATH = BASE_DIR / "data" / "dictionary.db"       # SQLite索引路径

# 检索配置
MAX_DICT_RESULTS = 5        # 每次最多检索几个词条注入 prompt
MIN_WORD_LENGTH = 2         # 最短检索词长度，过滤掉单字噪音