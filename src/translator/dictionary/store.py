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
            definitions TEXT      -- 用 | 分隔多个释义
        )
    """)

    # 建立索引，加快查询速度
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_simplified ON entries(simplified)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_traditional ON entries(traditional)")

    # 批量插入（比逐条插入快很多）
    cursor.executemany(
        "INSERT INTO entries (traditional, simplified, definitions) VALUES (?, ?, ?)",
        [
            (e.traditional, e.simplified,  "|".join(e.definitions))
            for e in entries
        ]
    )

    conn.commit()
    conn.close()
    print(f"词典索引建立完成，共收录 {len(entries)} 条词条。")


def index_exists(db_path: str) -> bool:
    """
    检查索引是否已建立。
    不能只看文件存不存在——sqlite3.connect() 哪怕数据库是空的也会先创建出这个文件，
    之前就是因为只判断"文件存在"，导致 entries 表其实是空的，也被误判为"已建立"。
    这里改成真正查表里有没有数据。
    """
    path = Path(db_path)
    if not path.exists():
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
        )
        if cursor.fetchone() is None:
            return False  # 表都还没建

        cursor.execute("SELECT COUNT(*) FROM entries")
        count = cursor.fetchone()[0]
        return count > 0
    finally:
        conn.close()


def lookup(db_path: str, word: str) -> list[DictEntry]:
    """
    精确查找某个词，同时查简体和繁体。
    返回匹配的 DictEntry 列表。
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT traditional, simplified,  definitions FROM entries "
        "WHERE simplified = ? OR traditional = ?",
        (word, word)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        DictEntry(
            traditional=row[0],
            simplified=row[1],
            definitions=row[2].split("|"),
        )
        for row in rows
    ]