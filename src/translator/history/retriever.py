import chromadb
from ..config import CHROMA_PATH, MAX_HISTORY_RESULTS, HISTORY_SIMILARITY_THRESHOLD
from ..embeddings.embedder import embed
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