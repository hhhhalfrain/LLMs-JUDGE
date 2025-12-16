from __future__ import annotations
import json
import os
from typing import Any, Dict, List


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_book_dirs(books_root: str) -> List[str]:
    """books_root 下每个子文件夹对应一本书"""
    out = []
    for name in os.listdir(books_root):
        p = os.path.join(books_root, name)
        if os.path.isdir(p):
            out.append(p)
    return sorted(out)
