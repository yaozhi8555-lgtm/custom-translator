from dataclasses import dataclass


@dataclass
class DictEntry:
    traditional: str        # 保留，用于检索繁体输入
    simplified: str         # 主要显示字段
    definitions: list[str]  # 释义

    def to_context_string(self) -> str:
        # 去掉拼音，只显示简体 + 释义
        defs = "; ".join(self.definitions)
        return f"{self.simplified}: {defs}"

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
    try:
        bracket_start = line.index("[")
        bracket_end = line.index("]")

        chars_part = line[:bracket_start].strip().split()
        traditional = chars_part[0]
        simplified = chars_part[1]

        defs_part = line[bracket_end + 1:].strip()
        definitions = [d for d in defs_part.split("/") if d.strip()]

        return DictEntry(
            traditional=traditional,
            simplified=simplified,
            definitions=definitions,
        )
    except (ValueError, IndexError):
        return None