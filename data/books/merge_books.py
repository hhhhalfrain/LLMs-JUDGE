from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import difflib


# =========================
# Hard-coded config (按你要求：不使用命令行参数)
# =========================
BOOKS_DIR = Path("")                 # 书籍文件夹根目录
INDEX_XLSX = BOOKS_DIR / "index.xlsx"          # Excel 书单
OUT_JSON = BOOKS_DIR / "merged_books.json"     # 输出
FUZZY_CUTOFF = 0.92                            # 默认开启模糊匹配 & 阈值


# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    """用于匹配的标准化：NFKC + 去首尾 + 压缩空白"""
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def loose_norm(s: str) -> str:
    """更“松”的匹配串：只保留字母数字（用于模糊匹配打分）"""
    s = norm(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_chapter_list(obj: Any) -> List[Any]:
    """
    兼容 chapters.json 的常见结构：
    - list
    - {"chapter":[...]} 或 {"chapters":[...]}
    """
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("chapter"), list):
            return obj["chapter"]
        if isinstance(obj.get("chapters"), list):
            return obj["chapters"]
    raise ValueError("chapters.json 格式不支持：必须是 list 或包含 chapter/chapters 的 dict。")


def pick_column(df: pd.DataFrame, candidates: List[str], required: bool = False) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}  # display -> real
    for name in candidates:
        if name in cols:
            return cols[name]
    if required:
        raise ValueError(f"index.xlsx 缺少必要列：{candidates}；当前列={list(df.columns)}")
    return None


def row_to_metadata(row: Dict[str, Any],
                    col_title: str,
                    col_author: Optional[str],
                    col_rating: Optional[str],
                    col_count: Optional[str],
                    col_intro: Optional[str]) -> Dict[str, Any]:
    def get(col: Optional[str]) -> Any:
        if not col:
            return None
        v = row.get(col)
        return None if (pd.isna(v) if v is not None else True) else v

    md = {
        "title": get(col_title),
        "author": get(col_author),
        "goodreads_rating": get(col_rating),
        "ratings_count": get(col_count),
        "intro": get(col_intro),
    }
    return md


def best_folder_match(book_title: str, folders: List[str], cutoff: float) -> Tuple[Optional[str], float]:
    """
    返回 (最佳匹配folder名 or None, 分数)
    - 先尝试 exact norm 匹配
    - 否则用 difflib 在 loose_norm 上模糊匹配
    """
    t_norm = norm(book_title)
    for f in folders:
        if norm(f) == t_norm:
            return f, 1.0

    t_loose = loose_norm(book_title)
    scored = []
    for f in folders:
        s = difflib.SequenceMatcher(a=t_loose, b=loose_norm(f)).ratio()
        scored.append((s, f))
    scored.sort(reverse=True, key=lambda x: x[0])

    if not scored:
        return None, 0.0

    best_score, best_folder = scored[0]
    if best_score < cutoff:
        return None, best_score

    # 防止“两个差不多像”的歧义（否则容易错配）
    if len(scored) >= 2 and scored[1][0] >= cutoff and abs(scored[0][0] - scored[1][0]) < 0.01:
        raise RuntimeError(
            f"模糊匹配出现歧义：'{book_title}' 同时接近 '{scored[0][1]}'(score={scored[0][0]:.3f}) "
            f"和 '{scored[1][1]}'(score={scored[1][0]:.3f}). "
            f"请重命名文件夹或提高 FUZZY_CUTOFF。"
        )

    return best_folder, best_score


def main() -> None:
    if not BOOKS_DIR.exists():
        raise FileNotFoundError(f"BOOKS_DIR 不存在：{BOOKS_DIR}")
    if not INDEX_XLSX.exists():
        raise FileNotFoundError(f"INDEX_XLSX 不存在：{INDEX_XLSX}")

    # 列名兼容：你的截图是中文“书名/作者/数/intro”+“GoodreadRatings”
    df = pd.read_excel(INDEX_XLSX)

    col_title = pick_column(df, ["Title"], required=True)
    col_author = pick_column(df, ["Author"], required=True)
    col_rating = pick_column(df, ["GoodreadsRatings"], required=True)
    col_count = pick_column(df, ["RatingsCount"], required=True)
    col_intro = pick_column(df, ["intro"], required=True)

    # 只以 xlsx 为权威书单：遍历 xlsx 的每一行书
    rows = df.to_dict(orient="records")

    # books_dir 下真实存在的文件夹列表（但会被“xlsx 权威列表”筛掉：多余的忽略）
    folders = sorted([p.name for p in BOOKS_DIR.iterdir() if p.is_dir()])

    used_folders = set()
    merged: List[Dict[str, Any]] = []
    missing: List[str] = []

    for r in rows:
        title = r.get(col_title)
        if title is None or (pd.isna(title) if hasattr(pd, "isna") else False):
            continue
        title = norm(str(title))
        if not title:
            continue

        # 默认开启模糊匹配（exact 优先 + fuzzy fallback）
        folder, score = best_folder_match(title, folders, FUZZY_CUTOFF)
        if folder is None:
            missing.append(title)
            continue

        if folder in used_folders:
            raise RuntimeError(
                f"同一个文件夹被多个 xlsx 书名匹配到：folder='{folder}'。"
                f"请检查 index.xlsx 是否有重复书名，或文件夹命名过于相似。"
            )
        used_folders.add(folder)

        chapters_path = BOOKS_DIR / folder / "chapters.json"
        if not chapters_path.exists():
            raise FileNotFoundError(f"找到书籍文件夹但缺少 chapters.json：{chapters_path}")

        chap_obj = read_json(chapters_path)
        chapters = extract_chapter_list(chap_obj)

        metadata = row_to_metadata(
            row=r,
            col_title=col_title,
            col_author=col_author,
            col_rating=col_rating,
            col_count=col_count,
            col_intro=col_intro,
        )
        # # 可选：把匹配信息也写进 metadata（英文 key）
        # metadata["matched_folder"] = folder
        # metadata["match_score"] = float(score)

        merged.append({
            "chapter": chapters,
            "metadata": metadata,
        })

    # 约束 1：匹配不到就报错（而不是默默跳过）
    if missing:
        msg = "\n".join([f"- {t}" for t in missing])
        raise RuntimeError(
            "以下书名在 books/ 下找不到可匹配的文件夹（只以 xlsx 书单为准，匹配失败直接报错）：\n"
            f"{msg}\n\n"
            f"当前 FUZZY_CUTOFF={FUZZY_CUTOFF}。你可以：\n"
            f"1) 把文件夹名改得更像 xlsx 书名；\n"
            f"2) 或调低 FUZZY_CUTOFF（在脚本顶部改）。"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {OUT_JSON}")
    print(f"[OK] Books merged (from xlsx only): {len(merged)}")
    print(f"[INFO] Extra folders under books/ ignored: {max(0, len(folders) - len(used_folders))}")


if __name__ == "__main__":
    main()
