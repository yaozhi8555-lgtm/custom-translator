# 阶段4：RAG 向量检索

## 目标

把 Phase3 的**精确匹配**升级为**语义相似度检索**，同时引入翻译历史作为第二个检索源：

```
Phase3：输入"冷淡" → 精确匹配词典 → 只能找到"冷淡"本身
Phase4：输入"冷淡" → 语义检索     → 能找到"冷淡"及语义相近的"冷漠""疏远"等词条
                                   → 还能找到历史里翻译过的类似句子，保持术语一致性
```

这也是真正意义上的 RAG（Retrieval-Augmented Generation）：
- **Retrieval**：用 embedding 向量做语义检索
- **Augmented**：把检索结果注入 prompt
- **Generation**：AI 带着上下文生成翻译

---

## 一个重要决策：Embedding 模型选哪个

有两个方向，各有取舍：

| | 本地模型（推荐） | API（OpenRouter） |
|---|---|---|
| 代表 | `paraphrase-multilingual-MiniLM-L12-v2` | `text-embedding-ada-002` |
| 费用 | 免费，只需一次下载（约 400MB） | 按 token 计费 |
| 速度 | 本地推理，无网络延迟 | 需要网络请求 |
| 质量 | 良好，支持中英双语 | 更高，但差距在此场景不明显 |
| 离线 | 支持 | 不支持 |

**推荐使用本地模型**，原因：
- 你选 OpenRouter 的初衷就是控制成本，embedding 按量计费很容易累积
- 词典向量化一次就要处理 12 万条，用 API 费用较高
- `paraphrase-multilingual-MiniLM-L12-v2` 对中英双语支持很好，够用

---

## 整体改动范围

```
src/translator/
├── embeddings/               ← 新增：embedding 模型封装
│   ├── __init__.py
│   └── embedder.py
├── history/                  ← 新增：翻译历史存储与检索
│   ├── __init__.py
│   ├── store.py              # SQLite 存结构化历史记录
│   └── retriever.py          # ChromaDB 语义检索历史
├── dictionary/
│   └── retriever.py          ← 修改：精确匹配基础上加语义检索
├── core.py                   ← 修改：翻译后存历史；用语义检索增强 prompt
└── config.py                 ← 修改：新增 embedding 和 ChromaDB 配置

data/
├── cedict_ts.u8              # Phase3 已有
├── dictionary.db             # Phase3 已有
├── history.db                ← 新增：翻译历史 SQLite
└── chroma/                   ← 新增：ChromaDB 向量数据（不进 git）
```

`.gitignore` 新增：
```
data/chroma/
```

---

## 步骤1：安装新依赖

```bash
pip install chromadb sentence-transformers
pip freeze > requirements.txt
```

`sentence-transformers` 首次安装会下载 PyTorch 依赖，体积较大（约 500MB），耐心等待。
`chromadb` 是本地向量数据库，数据存在 `data/chroma/` 目录下，不需要单独启动服务。

---

## 步骤2：更新 `config.py`

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

# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DICT_PATH = BASE_DIR / "data" / "cedict_ts.u8"
DICT_DB_PATH = BASE_DIR / "data" / "dictionary.db"

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
```

---

## 步骤3：实现 `embedder.py`

这是整个 Phase4 的基础，所有向量化操作都走这里：

```python
from sentence_transformers import SentenceTransformer
from ..config import EMBEDDING_MODEL

# 模块级别加载模型，只加载一次（首次使用时会下载模型文件）
_model = None

def get_model() -> SentenceTransformer:
    """
    懒加载：第一次调用时才加载模型，之后复用同一个实例。
    避免每次翻译都重新加载模型（加载一次约需 2-3 秒）。
    """
    global _model
    if _model is None:
        print(f"正在加载 embedding 模型：{EMBEDDING_MODEL}（首次使用需要下载）")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text: str) -> list[float]:
    """
    把一段文本转成向量。
    返回一个浮点数列表，长度固定（取决于模型，MiniLM 是 384 维）。
    """
    model = get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def embed_batch(texts: list[str], batch_size: int = 256) -> list[list[float]]:
    """
    批量向量化，比逐条调用 embed() 快很多。
    词典向量化时使用这个方法。
    batch_size：每批处理多少条，内存不够时调小这个值。
    """
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,   # 显示进度条，词典12万条需要一段时间
        convert_to_numpy=True,
    )
    return vectors.tolist()
```

**为什么用"懒加载"**：模型加载需要 2-3 秒，如果在模块顶层直接 `_model = SentenceTransformer(...)` 的话，只要 import 了这个模块就会触发加载，哪怕这次翻译根本用不到 embedding（比如 quick 模式）。懒加载可以让模型只在真正需要时才初始化。

---

## 步骤4：实现 `history/store.py`，存翻译历史

```python
import sqlite3
import json
from datetime import datetime
from ...config import HISTORY_DB_PATH


def init_db() -> None:
    """建表，如果表已存在则跳过。程序启动时调用一次。"""
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            source_lang TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            original    TEXT NOT NULL,
            translation TEXT NOT NULL,    -- quick模式存直译；detailed模式存意译
            mode        TEXT NOT NULL,    -- quick / detailed
            model_used  TEXT NOT NULL,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            full_result TEXT              -- detailed模式把整个JSON存这里备查
        )
    """)
    conn.commit()
    conn.close()


def save(
    source_lang: str,
    target_lang: str,
    original: str,
    translation: str,
    mode: str,
    model_used: str,
    input_tokens: int,
    output_tokens: int,
    full_result: dict = None,
) -> int:
    """
    存一条翻译记录，返回这条记录的 id（后面 ChromaDB 用这个 id 关联）。
    """
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO translations
           (timestamp, source_lang, target_lang, original, translation,
            mode, model_used, input_tokens, output_tokens, full_result)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            source_lang, target_lang,
            original, translation,
            mode, model_used,
            input_tokens, output_tokens,
            json.dumps(full_result, ensure_ascii=False) if full_result else None,
        )
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_by_id(record_id: int) -> dict | None:
    """根据 id 取回完整记录，供 ChromaDB 检索结果回查原文用。"""
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT original, translation, source_lang, target_lang, mode, timestamp "
        "FROM translations WHERE id = ?",
        (record_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "original": row[0],
        "translation": row[1],
        "source_lang": row[2],
        "target_lang": row[3],
        "mode": row[4],
        "timestamp": row[5],
    }
```

---

## 步骤5：实现 `history/retriever.py`，语义检索历史

```python
import chromadb
from ...config import CHROMA_PATH, MAX_HISTORY_RESULTS, HISTORY_SIMILARITY_THRESHOLD
from ...embeddings.embedder import embed
from .store import get_by_id, init_db


def _get_collection():
    """获取 ChromaDB 的翻译历史集合。"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="translation_history",
        metadata={"hnsw:space": "cosine"},  # 用余弦相似度衡量距离
    )


def add_to_index(record_id: int, original: str) -> None:
    """
    把一条翻译记录的原文向量化后存入 ChromaDB。
    record_id 是 SQLite 里的 id，用于关联回去查完整记录。
    """
    collection = _get_collection()
    vector = embed(original)
    collection.add(
        ids=[str(record_id)],
        embeddings=[vector],
        documents=[original],  # 存原文方便调试时直接看
    )


def retrieve_similar(text: str, target_lang: str) -> list[dict]:
    """
    检索和当前输入语义相似的历史翻译。
    只返回相似度超过阈值的结果，并且只返回同一目标语言的记录。
    """
    collection = _get_collection()

    # 如果历史为空，直接返回
    if collection.count() == 0:
        return []

    vector = embed(text)
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(MAX_HISTORY_RESULTS * 2, collection.count()),  # 多取一些，过滤后再截断
        include=["distances", "documents"],
    )

    similar = []
    for doc_id, distance in zip(
        results["ids"][0],
        results["distances"][0],
    ):
        # ChromaDB 余弦距离：0 表示完全相同，2 表示完全相反
        # 转成相似度：1 - distance/2，值越高越相似
        similarity = 1 - distance / 2

        if similarity < HISTORY_SIMILARITY_THRESHOLD:
            continue

        record = get_by_id(int(doc_id))
        if record and record["target_lang"] == target_lang:
            record["similarity"] = round(similarity, 3)
            similar.append(record)

    return similar[:MAX_HISTORY_RESULTS]


def format_for_prompt(records: list[dict]) -> str:
    """把相似历史记录格式化成适合注入 prompt 的字符串。"""
    if not records:
        return ""

    lines = ["以下是你之前翻译过的相似句子，请参考保持术语一致性："]
    for r in records:
        lines.append(f"- 原文：{r['original']}")
        lines.append(f"  译文：{r['translation']}")
    return "\n".join(lines)
```

**余弦距离和余弦相似度的换算**：ChromaDB 返回的是距离（越小越相似），`1 - distance/2` 把它转成相似度（越大越相似）。`HISTORY_SIMILARITY_THRESHOLD = 0.85` 意味着只有非常相似的句子才会被采用，避免注入不相关的历史干扰 AI。

---

## 步骤6：升级 `dictionary/retriever.py`，加入语义检索

Phase3 只有精确匹配，Phase4 在此基础上加一层语义检索作为补充：

```python
import chromadb
from ..config import (
    CHROMA_PATH, DICT_DB_PATH, DICT_PATH,
    MAX_DICT_RESULTS, MIN_WORD_LENGTH
)
from ..embeddings.embedder import embed, embed_batch
from .store import lookup, build_index, index_exists
from .loader import DictEntry, parse_cedict


DICT_COLLECTION_NAME = "dictionary"


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=DICT_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_vector_index() -> None:
    """
    把词典里所有词条的释义向量化，存入 ChromaDB。
    只需运行一次，之后复用。
    对释义（英文）做向量化，而不是对汉字——英文语义更适合用 embedding 检索。
    12万条预计需要 10-20 分钟，请耐心等待。
    """
    print("开始建立词典向量索引（只需一次，请耐心等待）...")
    entries = parse_cedict(str(DICT_PATH))
    collection = _get_collection()

    # 如果已有数据就跳过
    if collection.count() > 0:
        print(f"词典向量索引已存在（{collection.count()} 条），跳过重建。")
        return

    # 准备数据
    ids = []
    documents = []   # 用释义做向量化
    metadatas = []

    for i, entry in enumerate(entries):
        definition_text = "; ".join(entry.definitions)
        ids.append(str(i))
        documents.append(definition_text)
        metadatas.append({
            "simplified": entry.simplified,
            "traditional": entry.traditional,
        })

    # 分批向量化并存入（ChromaDB 单次最多存 5461 条）
    batch_size = 1000
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]
        batch_vectors = embed_batch(batch_docs)

        collection.add(
            ids=batch_ids,
            embeddings=batch_vectors,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        print(f"  已处理 {end}/{len(documents)} 条...")

    print(f"词典向量索引建立完成，共 {len(entries)} 条。")


def retrieve(text: str) -> list[DictEntry]:
    """
    两阶段检索：
    1. Phase3 的精确匹配（快，准）
    2. Phase4 的语义检索（能找到近义词）
    合并去重后返回。
    """
    results = []
    seen_simplified = set()

    # 阶段1：精确匹配（沿用 Phase3 的滑动窗口逻辑）
    exact_results = _exact_retrieve(text)
    for entry in exact_results:
        if entry.simplified not in seen_simplified:
            seen_simplified.add(entry.simplified)
            results.append(entry)

    # 阶段2：语义检索，补充精确匹配没找到的相关词
    if len(results) < MAX_DICT_RESULTS:
        semantic_results = _semantic_retrieve(text, top_k=MAX_DICT_RESULTS * 2)
        for entry in semantic_results:
            if entry.simplified not in seen_simplified:
                seen_simplified.add(entry.simplified)
                results.append(entry)
            if len(results) >= MAX_DICT_RESULTS:
                break

    return results[:MAX_DICT_RESULTS]


def _exact_retrieve(text: str) -> list[DictEntry]:
    """Phase3 的滑动窗口精确匹配，保持不变。"""
    results = []
    seen = set()
    candidates = _extract_candidates(text)

    for word in candidates:
        if len(word) < MIN_WORD_LENGTH:
            continue
        entries = lookup(str(DICT_DB_PATH), word)
        for entry in entries:
            if entry.simplified not in seen:
                seen.add(entry.simplified)
                results.append(entry)
        if len(results) >= MAX_DICT_RESULTS:
            break

    return results


def _semantic_retrieve(text: str, top_k: int = 5) -> list[DictEntry]:
    """用 embedding 做语义检索，返回释义语义相近的词条。"""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    vector = embed(text)
    results = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "documents"],
    )

    entries = []
    for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
        definitions = [d.strip() for d in doc.split(";") if d.strip()]
        entries.append(DictEntry(
            traditional=meta["traditional"],
            simplified=meta["simplified"],
            definitions=definitions,
        ))
    return entries


def _extract_candidates(text: str) -> list[str]:
    """滑动窗口，Phase3 保持不变。"""
    candidates = []
    for window_size in range(4, 1, -1):
        for i in range(len(text) - window_size + 1):
            word = text[i:i + window_size]
            if word not in candidates:
                candidates.append(word)
    return candidates


def format_for_prompt(entries: list[DictEntry]) -> str:
    if not entries:
        return ""
    lines = ["以下是词典中找到的相关词条，请在翻译时参考："]
    for entry in entries:
        defs = "; ".join(entry.definitions)
        lines.append(f"- {entry.simplified}: {defs}")
    return "\n".join(lines)
```

---

## 步骤7：更新 `core.py`，翻译后存历史

改动点：翻译完成后把结果存入历史，同时检索历史注入 prompt：

```python
from .history import store as history_store
from .history.retriever import (
    retrieve_similar as retrieve_history,
    add_to_index as add_history_to_index,
    format_for_prompt as history_format_for_prompt,
)
from .prompts import inject_dict_context

# 程序启动时初始化历史数据库
history_store.init_db()


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
```

`_translate_detailed` 同理，`translation` 字段存意译结果（`result.translation.meaning`），`full_result` 字段存完整 JSON。

---

## 步骤8：手动验证

**第一次运行**会触发两件事（都只做一次）：
1. 加载 embedding 模型（约 2-3 秒，首次需下载）
2. 建立词典向量索引（12万条，约 10-20 分钟）

建议先单独跑一次索引建立，确认没问题再开始翻译：

```bash
# 在 Python 里手动触发索引建立
python -c "from translator.dictionary.retriever import build_vector_index; build_vector_index()"
```

然后正式验证：

```bash
# 测试语义检索效果：输入"冷漠"，看能不能同时找到"冷淡""疏远"
python -m translator.cli "他的态度很冷漠" --mode detailed

# 翻译两次相似句子，第二次应该能检索到第一次的历史
python -m translator.cli "我今天很累" --mode quick
python -m translator.cli "我今天非常疲惫" --mode quick
```

验证清单：
- [x] embedding 模型正常加载
- [x] 词典向量索引建立完成，数量约 12 万（实际 124,750 条）
- [x] 语义检索能找到精确匹配之外的相关词条
- [x] 翻译历史被正确存入 SQLite 和 ChromaDB
- [x] 第二次翻译相似句子时，prompt 里能看到历史参考

> **验证过程中发现并修复的问题**：
>
> 1. **`dictionary/store.py` 的 `index_exists()` 只判断数据库文件存不存在，不判断表里有没有数据**。
>    实际情况是 `dictionary.db` 文件早就存在（可能之前建索引时中途中断），但 `entries` 表是空的，
>    导致 `build_index()` 一直被跳过，精确匹配永远查不到任何词条。
>    已修复为真正检查 `entries` 表的行数（`SELECT COUNT(*)`），并重新触发建索引，
>    现在精确匹配能正确命中「态度」「冷漠」等常见词。
>
> 2. **`history/retriever.py` 的 `retrieve_similar()` 会把重复的 (原文, 译文) 记录一起返回**。
>    起因是同一句话被反复翻译时，SQLite/ChromaDB 里会存下多条内容相同的历史记录
>    （这是预期行为——每次真实调用都要如实记账，方便后续做 token 用量统计），
>    但检索出来直接注入 prompt 时会把重复内容也塞进去，浪费 token、增加噪音。
>    已修复为在检索阶段按 `(原文, 译文)` 去重（去重发生在按 `MAX_HISTORY_RESULTS` 截断之前，
>    避免重复项占用真正该保留的名额）；SQLite/ChromaDB 里的原始记录本身不做任何清理或合并。
>
> 3. **词典语义检索质量还有优化空间（暂不处理，留给后续阶段）**：
>    词典向量化时用的是词条的英文释义做 embedding，但查询时用的是完整的中文原句做 embedding，
>    属于跨语言检索，语义空间没有完全对齐，召回质量不如预期（比如输入"冷漠"，
>    期望召回"冷淡""疏远"，实际召回的是"不服气""嫌弃"等相关度较低的词）。

---

## 步骤9：提交代码

```bash
git add .
git commit -m "阶段4: RAG向量检索，sentence-transformers embedding，词典+历史双源检索"
```

---

## 步骤10：更新文档

phase4 完成后更新 `docs/翻译软件开发计划.md`，补充完成记录和踩坑笔记。

---

## Phase4 学习重点

| 知识点 | 在哪里体现 |
|---|---|
| Embedding 原理 | `embedder.py`：文本→向量，语义相近的文本向量距离更近 |
| 懒加载模式 | `get_model()` 的全局变量 + 首次调用才初始化 |
| ChromaDB 基本用法 | `add()`、`query()`、`PersistentClient` |
| 余弦相似度 | `1 - distance/2` 的换算逻辑 |
| 两阶段检索 | 精确匹配优先，语义检索补充，合并去重 |
| 批量向量化 | `embed_batch()` + 分批写入 ChromaDB |
| 相似度阈值过滤 | `HISTORY_SIMILARITY_THRESHOLD` 避免低质量历史干扰 |
| 多源 RAG | 词典 + 历史两个检索源，分别格式化后注入同一个 prompt |

## Phase4 → Phase5 的升级方向

Phase4 的检索还是"被动的"——每次翻译都固定查词典和历史，AI 自己没有决策权。

Phase5 引入 **Tool Use（工具调用）**：让 AI 自己判断"我需要查词典吗？""我需要查历史吗？""还是两个都查？"——把检索的决策权还给模型，这是迈向 Agent 的关键一步。
