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
        print(f"正在加载 embedding 模型：{EMBEDDING_MODEL}（正在加载 embedding 模型）")
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