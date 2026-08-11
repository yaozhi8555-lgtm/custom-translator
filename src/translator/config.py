import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 原有配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"  # 后面可做成可配置

if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY，请检查 .env 文件")

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录
DICT_PATH = BASE_DIR / "data" / "cedict_ts.u8"           # 词典文件路径
DICT_DB_PATH = BASE_DIR / "data" / "dictionary.db"       # SQLite索引路径

# Phase4 新增
HISTORY_DB_PATH = BASE_DIR / "data" / "history.db"      # 翻译历史 SQLite
CHROMA_PATH = str(BASE_DIR / "data" / "chroma")         # ChromaDB 存储路径

# Embedding 配置
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 本地多语言模型

# 检索配置
MAX_DICT_RESULTS = 5          # 词典最多注入几个词条
MAX_HISTORY_RESULTS = 3       # 历史最多注入几条相似翻译
MIN_WORD_LENGTH = 2
HISTORY_SIMILARITY_THRESHOLD = 0.85   # 历史检索的相似度阈值，低于这个分数的结果不用