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