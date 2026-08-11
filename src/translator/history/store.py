import sqlite3
import json
from datetime import datetime
from ..config import HISTORY_DB_PATH


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