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
    _ensure_sqlite_index()
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


def _ensure_sqlite_index() -> None:
    """确保 SQLite 索引存在"""
    if not index_exists(str(DICT_DB_PATH)):
        if not DICT_PATH.exists():
            raise FileNotFoundError(
                f"词典文件不存在：{DICT_PATH}\n"
                f"请从 https://www.mdbg.net/chinese/dictionary?page=cedict 下载后放入 data/ 目录"
            )
        build_index(str(DICT_PATH), str(DICT_DB_PATH))


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