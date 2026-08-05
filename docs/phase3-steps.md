# 阶段3：融合老词库（RAG 检索源）

## 目标

把开源词典接入翻译流程，作为 AI 翻译前的**参考上下文**：

```
用户输入文本
    ↓
从词典里检索相关词条（尤其是多义词、专业术语）
    ↓
把词条作为上下文注入 prompt
    ↓
AI 结合词典依据翻译，而非纯凭训练记忆
```

词库不是替代 AI，而是给 AI **补充可查证的参考依据**，让翻译结果更准确、更稳定。

---

## 为什么这是 RAG

RAG = Retrieval-Augmented Generation（检索增强生成）。

Phase3 是最基础形态的 RAG：
- **检索**：从词典里找和输入文本相关的词条
- **增强**：把词条塞进 prompt
- **生成**：AI 带着这些参考信息生成翻译

Phase4 会在这个基础上升级：把翻译历史也做成可检索的语料，用向量检索替代简单的关键词匹配，实现更智能的"找相似"能力。

---

## 词典选型：CC-CEDICT

使用 **CC-CEDICT**，理由：
- 开源免费，可商用（Creative Commons 协议）
- 中英双语，契合当前主要翻译方向
- 收录约 12 万条词条，覆盖日常 + 专业词汇
- 格式简单，解析容易
- 下载地址：https://www.mdbg.net/chinese/dictionary?page=cedict

词条格式示例：
```
# 注释行以 # 开头
中文 中文 [zhong1 wen2] /Chinese language/Mandarin/
虚拟语气 虚拟语气 [xu1 ni3 yu3 qi4] /subjunctive mood (grammar)/
```
每行格式：`繁体 简体 [拼音] /释义1/释义2/`

---

## 整体改动范围

```
src/translator/
├── dictionary/           ← 新增：词典相关模块
│   ├── __init__.py
│   ├── loader.py         # 解析 CC-CEDICT 文件，读入内存
│   ├── store.py          # 管理本地索引（SQLite精确查找）
│   └── retriever.py      # 检索逻辑：输入文本 → 相关词条列表
├── prompts.py            ← 修改：新增带词典上下文的 prompt 模板
├── core.py               ← 修改：翻译前先检索词典，把结果注入 prompt
└── config.py             ← 修改：新增词典路径等配置项

data/                     ← 新增（放在项目根目录，不进 git）
└── cedict_ts.u8          # 下载的词典原始文件

.gitignore                ← 修改：忽略 data/ 目录
```

---

## 步骤1：下载词典，更新 .gitignore

在项目根目录创建 `data/` 文件夹：
```bash
mkdir data
```

去 https://www.mdbg.net/chinese/dictionary?page=cedict 下载最新的词典压缩包，解压后把 `cedict_ts.u8` 放进 `data/` 目录。

在 `.gitignore` 里加上（词典文件较大，不应提交到 git）：
```
data/
```

---

## 步骤2：更新 `config.py`，加入词典配置

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 原有配置
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

if not OPENROUTER_API_KEY:
    raise ValueError("未找到 OPENROUTER_API_KEY，请检查 .env 文件")

# 新增：词典配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 项目根目录
DICT_PATH = BASE_DIR / "data" / "cedict_ts.u8"           # 词典文件路径
DICT_DB_PATH = BASE_DIR / "data" / "dictionary.db"       # SQLite索引路径

# 检索配置
MAX_DICT_RESULTS = 5        # 每次最多检索几个词条注入 prompt
MIN_WORD_LENGTH = 2         # 最短检索词长度，过滤掉单字噪音
```

`Path(__file__)` 会自动计算出当前文件的绝对路径，然后 `.parent.parent.parent` 往上走三层到项目根目录——这样不管项目放在哪个位置，路径都能自动算对，不用写死。

---

## 步骤3：实现 `loader.py`，解析词典文件

```python
from dataclasses import dataclass


@dataclass
class DictEntry:
    """单条词典记录"""
    traditional: str    # 繁体
    simplified: str     # 简体
    pinyin: str         # 拼音
    definitions: list[str]  # 释义列表

    def to_context_string(self) -> str:
        """
        转成可以直接塞进 prompt 的字符串格式
        例：中文 [zhong1 wen2]: Chinese language; Mandarin
        """
        defs = "; ".join(self.definitions)
        return f"{self.simplified} [{self.pinyin}]: {defs}"


def parse_cedict(file_path: str) -> list[DictEntry]:
    """
    解析 CC-CEDICT 文件，返回所有词条列表。
    文件较大（约 12 万条），解析一次后应存入 SQLite，不要每次重新解析。
    """
    entries = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # 跳过注释行和空行
            if not line or line.startswith("#"):
                continue

            entry = _parse_line(line)
            if entry:
                entries.append(entry)

    return entries


def _parse_line(line: str) -> DictEntry | None:
    """
    解析单行词条。
    格式：繁体 简体 [拼音] /释义1/释义2/
    """
    try:
        # 拆分拼音部分：找到 [ 和 ] 的位置
        bracket_start = line.index("[")
        bracket_end = line.index("]")

        # 繁简体在 [ 之前，用空格分割
        chars_part = line[:bracket_start].strip().split()
        traditional = chars_part[0]
        simplified = chars_part[1]

        pinyin = line[bracket_start + 1:bracket_end]

        # 释义在 ] 之后，用 / 分割，过滤空字符串
        defs_part = line[bracket_end + 1:].strip()
        definitions = [d for d in defs_part.split("/") if d.strip()]

        return DictEntry(
            traditional=traditional,
            simplified=simplified,
            pinyin=pinyin,
            definitions=definitions,
        )
    except (ValueError, IndexError):
        # 格式不对的行直接跳过，不影响整体
        return None
```

---

## 步骤4：实现 `store.py`，建立 SQLite 索引

词典文件每次从头解析太慢（12万条，大概需要几秒），用 SQLite 做一个本地索引，**只建立一次，之后直接查数据库**。

```python
import sqlite3
from pathlib import Path
from .loader import DictEntry, parse_cedict


def build_index(dict_path: str, db_path: str) -> None:
    """
    把词典文件解析后存入 SQLite，建立索引。
    只需要运行一次，之后直接用 db_path 查询。
    """
    print(f"开始建立词典索引，这需要几秒钟...")
    entries = parse_cedict(dict_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            traditional TEXT,
            simplified TEXT,
            pinyin TEXT,
            definitions TEXT      -- 用 | 分隔多个释义
        )
    """)

    # 建立索引，加快查询速度
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_simplified ON entries(simplified)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traditional ON entries(traditional)")

    # 批量插入（比逐条插入快很多）
    cursor.executemany(
        "INSERT INTO entries (traditional, simplified, pinyin, definitions) VALUES (?, ?, ?, ?)",
        [
            (e.traditional, e.simplified, e.pinyin, "|".join(e.definitions))
            for e in entries
        ]
    )

    conn.commit()
    conn.close()
    print(f"词典索引建立完成，共收录 {len(entries)} 条词条。")


def index_exists(db_path: str) -> bool:
    """检查索引是否已建立"""
    return Path(db_path).exists()


def lookup(db_path: str, word: str) -> list[DictEntry]:
    """
    精确查找某个词，同时查简体和繁体。
    返回匹配的 DictEntry 列表。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT traditional, simplified, pinyin, definitions FROM entries "
        "WHERE simplified = ? OR traditional = ?",
        (word, word)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        DictEntry(
            traditional=row[0],
            simplified=row[1],
            pinyin=row[2],
            definitions=row[3].split("|"),
        )
        for row in rows
    ]
```

---

## 步骤5：实现 `retriever.py`，检索逻辑

这是 Phase3 的核心：**给定一段输入文本，找出里面值得查词典的词，返回相关词条**。

```python
from .store import lookup, build_index, index_exists
from .loader import DictEntry
from ..config import DICT_PATH, DICT_DB_PATH, MAX_DICT_RESULTS, MIN_WORD_LENGTH


def ensure_index() -> None:
    """
    确保索引存在，不存在则自动建立。
    在程序启动时调用一次即可。
    """
    if not index_exists(str(DICT_DB_PATH)):
        if not DICT_PATH.exists():
            raise FileNotFoundError(
                f"词典文件不存在：{DICT_PATH}\n"
                f"请从 https://www.mdbg.net/chinese/dictionary?page=cedict 下载后放入 data/ 目录"
            )
        build_index(str(DICT_PATH), str(DICT_DB_PATH))


def retrieve(text: str) -> list[DictEntry]:
    """
    从文本中提取候选词，检索词典，返回相关词条。

    检索策略：滑动窗口分词
    中文没有空格，用"滑动窗口"从长到短切出候选词：
    例如"虚拟语气"会生成：
      ["虚拟语气", "虚拟语", "虚拟", "拟语气", "拟语", "语气"]
    然后逐个去词典里查，查到了就收录，查不到就跳过。
    """
    ensure_index()

    candidates = _extract_candidates(text)
    results = []
    seen = set()   # 避免同一个词条重复出现

    for word in candidates:
        if len(word) < MIN_WORD_LENGTH:
            continue

        entries = lookup(str(DICT_DB_PATH), word)
        for entry in entries:
            key = entry.simplified
            if key not in seen:
                seen.add(key)
                results.append(entry)

        if len(results) >= MAX_DICT_RESULTS:
            break

    return results[:MAX_DICT_RESULTS]


def _extract_candidates(text: str) -> list[str]:
    """
    用滑动窗口从文本里切出候选词，优先长词。
    窗口大小从4到2，长词优先匹配。
    """
    candidates = []
    for window_size in range(4, 1, -1):  # 4字、3字、2字
        for i in range(len(text) - window_size + 1):
            word = text[i:i + window_size]
            if word not in candidates:
                candidates.append(word)
    return candidates


def format_for_prompt(entries: list[DictEntry]) -> str:
    """
    把检索到的词条格式化成适合注入 prompt 的字符串。
    """
    if not entries:
        return ""

    lines = ["以下是词典中找到的相关词条，请在翻译时参考："]
    for entry in entries:
        lines.append(f"- {entry.to_context_string()}")

    return "\n".join(lines)
```

**为什么用滑动窗口而不是分词库**：
- 分词库（如 jieba）需要额外安装，增加复杂度
- 词典本身就是查询的依据，查到了说明是真实词汇，查不到就跳过，天然起到了"过滤噪音"的作用
- Phase4 升级到向量检索后，这个策略会被替换，现在用简单方案就够

---

## 步骤6：更新 `prompts.py`，加入词典上下文模板

在原有 prompt 的基础上，新增带词典上下文的版本：

```python
# 原有的 prompt 保持不变
QUICK_SYSTEM_PROMPT = """..."""
DETAILED_SYSTEM_PROMPT = """..."""


# 新增：带词典上下文的 prompt 包装函数
def inject_dict_context(base_prompt: str, dict_context: str) -> str:
    """
    把词典检索结果注入到 system prompt 里。
    如果没有检索到词条，直接返回原始 prompt 不做修改。
    """
    if not dict_context:
        return base_prompt

    return base_prompt + f"\n\n{dict_context}"
```

设计说明：不是重写整个 prompt，而是在原有 prompt **后面追加**词典上下文。这样：
- 原有 prompt 的结构和要求保持不变
- 词典上下文作为补充信息附在后面
- 没有词条时完全不影响原有逻辑

---

## 步骤7：更新 `core.py`，翻译前先检索词典

只需要改 `_translate_quick` 和 `_translate_detailed` 两个函数，在调用 provider 之前先查词典：

```python
from .dictionary.retriever import retrieve, format_for_prompt
from .prompts import (
    QUICK_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    inject_dict_context
)

def _translate_quick(provider, text, source_lang, target_lang, model) -> QuickResult:
    # 新增：查词典
    dict_entries = retrieve(text)
    dict_context = format_for_prompt(dict_entries)

    # 原有逻辑：组装 prompt，注入词典上下文
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
    # 新增：查词典
    dict_entries = retrieve(text)
    dict_context = format_for_prompt(dict_entries)

    # 原有逻辑：组装 prompt，注入词典上下文
    base_prompt = DETAILED_SYSTEM_PROMPT.format(
        source_lang=source_lang,
        target_lang=target_lang,
    )
    system_prompt = inject_dict_context(base_prompt, dict_context)

    # 后续解析逻辑完全不变
    result = provider.chat(...)
    ...
```

---

## 步骤8：手动验证

**第一次运行会自动建立索引（需要几秒）**，之后就是正常速度。

```bash
# 测试多义词：确认词典参考是否有效
python -m translator.cli "他的态度很冷淡" --mode detailed

# 测试专业术语：虚拟语气
python -m translator.cli "If I were you, I wouldn't do that." --from 英文 --to 中文 --mode detailed

# 测试普通句子：确认没有词典词条时也能正常工作
python -m translator.cli "今天天气很好" --mode quick
```

验证清单：
- [ ] 第一次运行时看到"开始建立词典索引..."提示，完成后正常翻译
- [ ] 之后再运行不再重复建立索引，速度正常
- [ ] 含多义词的句子，翻译结果是否更准确
- [ ] 词典文件不存在时，报错提示是否清晰（引导去下载）
- [ ] quick 和 detailed 模式都能正常工作

---

## 步骤9：提交代码

```bash
git add .
git commit -m "阶段3: 融合CC-CEDICT词库，滑动窗口检索，词典上下文注入prompt"
```

---

## 步骤10：更新文档

phase3 完成后更新 `docs/翻译软件开发计划.md`，补充完成记录和踩坑笔记。

---

## Phase3 学习重点

| 知识点 | 在哪里体现 |
|---|---|
| RAG 最基础形态 | 检索（词典查询）→ 增强（注入prompt）→ 生成（AI翻译） |
| 数据解析 | `loader.py` 解析 CC-CEDICT 固定格式 |
| SQLite 索引 | `store.py` 一次建立，反复查询，批量插入性能优化 |
| 滑动窗口分词 | `retriever.py` 不依赖分词库，用词典本身做过滤 |
| Prompt 组合 | `inject_dict_context()` 在原有 prompt 后追加上下文，不破坏原结构 |
| 防御性设计 | 词典文件不存在时友好报错；词条为空时不影响原有流程 |

## Phase3 → Phase4 的升级方向

Phase3 用的是**精确匹配**（词条简体/繁体完全等于查询词才算命中），有两个局限：
1. 查不到"近义词"或"相关词"
2. 翻译历史无法被检索利用

Phase4 会引入 **embedding 向量检索**（ChromaDB），把词典和翻译历史都向量化，用"语义相似度"替代"精确字符匹配"，实现真正意义上的 RAG。
