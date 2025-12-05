import json
import os
from typing import List, Dict, Any, Optional
import math

import pandas as pd


# ===== 运行配置：书单 & Goodreads 评分 =====

CANONICAL_TITLES: List[str] = [
    "Fourth Wing (The Empyrean, #1)",
    # "Iron Flame (The Empyrean, #2)",
    # "Divine Rivals (Letters of Enchantment, #1)",
    "Yours Truly (Part of Your World, #2)",
    # "The Frozen River",
    # "Knockemout 3 Things We Left Behind",
    "The Covenant of Water",
    "The Women",
    "Funny Story",
    "Just for the Summer",
    "The God of the Woods",
    "All the Colors of the Dark",
    "Quicksilver (Fae & Alchemy, #1)",
    "James",
    # "Onyx Storm (The Empyrean, #3)",

    "Yellowface",
    # "The Pumpkin Spice Café (Dream Harbor, #1)",
    "The Husbands",
    # "Society of Lies",
    # "The Blue Hour",
    "The Widow's Husband's Secret Lie",
    "Incidents Around the House",
    "A Tempest of Tea (Blood and Tea, #1)",
    "Only If You're Lucky",
    "The Night We Lost Him",
    "A Novel Love Story",
    # "The Christmas Tree Farm (Dream Harbor, #3)",
    # "No One Can Know",
    "I Hope This Finds You Well",
    "The Striker (Gods of the Game, #1)",
]

# Goodreads 评分（按上面书名一一对应）
GOODREADS_SCORES: Dict[str, float] = {
    "Fourth Wing (The Empyrean, #1)": 4.57,
    "Iron Flame (The Empyrean, #2)": 4.36,
    "Divine Rivals (Letters of Enchantment, #1)": 4.15,
    "Yours Truly (Part of Your World, #2)": 4.30,
    "The Frozen River": 4.38,
    "Knockemout 3 Things We Left Behind": 4.39,
    "The Covenant of Water": 4.41,
    "The Women": 4.59,
    "Funny Story": 4.19,
    "Just for the Summer": 4.35,
    "The God of the Woods": 4.11,
    "All the Colors of the Dark": 4.25,
    "Quicksilver (Fae & Alchemy, #1)": 4.50,
    "James": 4.45,
    "Onyx Storm (The Empyrean, #3)": 4.21,

    "Yellowface": 3.73,
    "The Pumpkin Spice Café (Dream Harbor, #1)": 3.36,
    "The Husbands": 3.51,
    "Society of Lies": 3.52,
    "The Blue Hour": 3.28,
    "The Widow's Husband's Secret Lie": 3.33,
    "Incidents Around the House": 3.58,
    "A Tempest of Tea (Blood and Tea, #1)": 3.50,
    "Only If You're Lucky": 3.60,
    "The Night We Lost Him": 3.59,
    "A Novel Love Story": 3.62,
    "The Christmas Tree Farm (Dream Harbor, #3)": 3.67,
    "No One Can Know": 3.67,
    "I Hope This Finds You Well": 3.74,
    "The Striker (Gods of the Game, #1)": 3.78,
}


BASE_RESULT_DIR = "../data/result"

# 输出目录和文件名
OUTPUT_DIR = "../data/result"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "book_ratings_summary.xlsx")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "book_ratings_summary.csv")


# ===== 基础工具：数据清洗与统计 =====

def slugify(s: str) -> str:
    """把 education_level 等字符串变成安全的列名。"""
    import re
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_book_records(book_title: str) -> pd.DataFrame:
    """
    读取某一本书的 interview_records.json，并抽取：
    book_name, age, sex, education_level, score
    """
    folder = os.path.join(BASE_RESULT_DIR, book_title)
    json_path = os.path.join(folder, "interview_records.json")

    if not os.path.exists(json_path):
        print(f"[WARN] 未找到文件：{json_path}（跳过 {book_title}）")
        return pd.DataFrame(
            columns=["book_name", "age", "sex", "education_level", "score"]
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows: List[Dict[str, Any]] = []

    for rec in data:
        profile = rec.get("profile", {})
        book_name = profile.get("book_name", book_title)
        age = profile.get("age", None)
        sex = profile.get("sex", None)
        edu = profile.get("education_level", None)

        # 优先用 parsed_score，没有就从 raw_info.response 里兜底解析
        score = rec.get("parsed_score", None)
        if score is None:
            raw_info = rec.get("raw_info", {})
            resp = raw_info.get("response")
            if isinstance(resp, str):
                try:
                    resp_json = json.loads(resp)
                    score = resp_json.get("score", None)
                except Exception:
                    score = None

        rows.append(
            {
                "book_name": book_name,
                "age": age,
                "sex": sex,
                "education_level": edu,
                "score": score,
            }
        )

    df = pd.DataFrame(rows)

    # 确保 score 为数值，并去掉缺失
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])

    # age 也转为数值，方便分段
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    return df


def compute_statistics(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    对所有书的合并数据计算：
    - 每本书整体均分
    - 按年龄段 / sex / education_level 的均分
    """

    edu_levels = sorted(
        [e for e in all_data["education_level"].dropna().unique()]
    )

    results = []

    for book_name, df_b in all_data.groupby("book_name"):
        row: Dict[str, Any] = {
            "book_name": book_name,
            "n_reviews": len(df_b),
            "mean_score_overall": df_b["score"].mean(),
        }

        # 年龄段
        def age_mean(min_age: int, max_age: Optional[int]) -> float:
            if max_age is None:
                mask = df_b["age"] >= min_age
            else:
                mask = (df_b["age"] >= min_age) & (df_b["age"] <= max_age)
            return df_b.loc[mask, "score"].mean()

        row["mean_score_age_10_30"] = age_mean(10, 30)
        row["mean_score_age_31_60"] = age_mean(31, 60)
        row["mean_score_age_61_plus"] = age_mean(61, None)

        # 性别
        for sex in ["Male", "Female"]:
            mask = df_b["sex"] == sex
            row[f"mean_score_sex_{sex.lower()}"] = df_b.loc[mask, "score"].mean()

        # 教育程度
        for edu in edu_levels:
            mask = df_b["education_level"] == edu
            col_name = f"mean_score_edu_{slugify(edu)}"
            row[col_name] = df_b.loc[mask, "score"].mean()

        results.append(row)

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values("book_name").reset_index(drop=True)
    return summary_df


# ===== 相关性工具：秩 / Spearman / Kendall τ-b =====

def _compute_ranks(values: List[float]) -> List[float]:
    """
    根据数值计算秩，ties 使用平均秩。
    返回的秩列表与 values 同长度。
    """
    indexed = list(enumerate(values))  # (idx, value)
    indexed.sort(key=lambda x: x[1])

    n = len(values)
    ranks = [0.0] * n
    pos = 1  # 秩从 1 开始

    i = 0
    while i < n:
        j = i
        # 找到这一段 ties
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        start_pos = pos
        end_pos = pos + (j - i)
        avg_rank = (start_pos + end_pos) / 2.0
        for k in range(i, j + 1):
            original_idx = indexed[k][0]
            ranks[original_idx] = avg_rank
        pos = end_pos + 1
        i = j + 1

    return ranks


def spearman_rho(x: List[float], y: List[float]) -> float:
    """
    Spearman 秩相关系数 ρs：对 x,y 转秩，再做 Pearson 相关。
    """
    if len(x) != len(y):
        raise ValueError("x 和 y 的长度必须相同")

    rx = _compute_ranks(x)
    ry = _compute_ranks(y)

    n = len(x)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        # 某一侧全同值，相关性无意义
        return float("nan")

    rho = num / (den_x * den_y)
    return rho


def kendall_tau_b(x: List[float], y: List[float]) -> float:
    """
    Kendall's τ-b（带 ties 修正）：
      τ_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))
    """
    if len(x) != len(y):
        raise ValueError("x 和 y 的长度必须相同")

    n = len(x)
    if n < 2:
        return float("nan")

    C = 0
    D = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 or dy == 0:
                # 有 ties，按 τ-b 公式，这对 pair 不计入 C/D
                continue
            if dx * dy > 0:
                C += 1
            elif dx * dy < 0:
                D += 1

    n0 = n * (n - 1) / 2.0

    # x 中 ties
    value_to_count_x: Dict[float, int] = {}
    for v in x:
        value_to_count_x[v] = value_to_count_x.get(v, 0) + 1
    n1 = sum(c * (c - 1) / 2.0 for c in value_to_count_x.values() if c > 1)

    # y 中 ties
    value_to_count_y: Dict[float, int] = {}
    for v in y:
        value_to_count_y[v] = value_to_count_y.get(v, 0) + 1
    n2 = sum(c * (c - 1) / 2.0 for c in value_to_count_y.values() if c > 1)

    den = math.sqrt((n0 - n1) * (n0 - n2))
    if den == 0:
        return float("nan")

    tau_b = (C - D) / den
    return tau_b


def compute_correlations(summary_df: pd.DataFrame) -> Dict[str, Any]:
    """
    基于 summary_df（已经有 mean_score_overall）：
    - 对齐 CANONICAL_TITLES + GOODREADS_SCORES
    - 构造 Goodreads 向量 & 实验均分向量
    - 计算 Spearman ρs & Kendall τ-b
    - 返回用于写 Excel 的 correlation_df 和指标
    """
    summary_df = summary_df.copy()
    summary_df["goodreads_score"] = summary_df["book_name"].map(GOODREADS_SCORES)

    used_titles: List[str] = []
    gr_scores: List[float] = []
    exp_scores: List[float] = []

    for title in CANONICAL_TITLES:
        if title not in GOODREADS_SCORES:
            print(f"[WARN] Goodreads 评分中缺少：{title}")
            continue

        row = summary_df.loc[summary_df["book_name"] == title]
        if row.empty:
            print(f"[WARN] 实验结果中缺少：{title}")
            continue

        exp = float(row["mean_score_overall"].iloc[0])
        gr = float(GOODREADS_SCORES[title])

        if not (math.isnan(exp) or math.isnan(gr)):
            used_titles.append(title)
            exp_scores.append(exp)
            gr_scores.append(gr)

    if len(used_titles) < 2:
        print("[WARN] 可用于相关性计算的书目不足 2 本，返回 NaN。")
        return {
            "used_titles": used_titles,
            "gr_scores": gr_scores,
            "exp_scores": exp_scores,
            "spearman_rho": float("nan"),
            "kendall_tau_b": float("nan"),
            "correlation_df": pd.DataFrame(),
        }

    rho_s = spearman_rho(gr_scores, exp_scores)
    tau_b = kendall_tau_b(gr_scores, exp_scores)

    corr_df = pd.DataFrame({
        "book_name": used_titles,
        "goodreads_score": gr_scores,
        "experiment_mean_score": exp_scores,
    })

    # 把对应的秩也写进去，方便后续肉眼检查局部排序
    corr_df["goodreads_rank"] = _compute_ranks(corr_df["goodreads_score"].tolist())
    corr_df["experiment_rank"] = _compute_ranks(corr_df["experiment_mean_score"].tolist())

    return {
        "used_titles": used_titles,
        "gr_scores": gr_scores,
        "exp_scores": exp_scores,
        "spearman_rho": rho_s,
        "kendall_tau_b": tau_b,
        "correlation_df": corr_df,
    }


# ===== 主流程 =====

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 读取所有书的 interview_records.json
    all_dfs: List[pd.DataFrame] = []

    for title in CANONICAL_TITLES:
        df_book = load_book_records(title)
        if df_book.empty:
            print(f"[INFO] 书《{title}》没有有效数据（或文件缺失）。")
        else:
            print(f"[INFO] 书《{title}》共读取 {len(df_book)} 条记录。")
            all_dfs.append(df_book)

    if not all_dfs:
        print("[ERROR] 没有任何书的数据被成功读取，程序结束。")
        return

    all_data = pd.concat(all_dfs, ignore_index=True)
    all_data["score"] = pd.to_numeric(all_data["score"], errors="coerce")
    all_data["age"] = pd.to_numeric(all_data["age"], errors="coerce")
    all_data = all_data.dropna(subset=["score"])

    # 2. 计算每本书统计
    summary_df = compute_statistics(all_data)
    # 给 summary 补上一列 Goodreads 评分，方便一起看
    summary_df["goodreads_score"] = summary_df["book_name"].map(GOODREADS_SCORES)

    # 3. 计算 Spearman / Kendall 相关性
    corr_result = compute_correlations(summary_df)
    corr_df = corr_result["correlation_df"]
    metrics_df = pd.DataFrame(
        [
            {"metric": "Spearman_rho", "value": corr_result["spearman_rho"]},
            {"metric": "Kendall_tau_b", "value": corr_result["kendall_tau_b"]},
            {"metric": "n_books_used", "value": len(corr_result["used_titles"])},
        ]
    )

    # 4. 写出 Excel（或退回 CSV）
    try:
        with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="summary")
            all_data.to_excel(writer, index=False, sheet_name="raw_data")
            if not corr_df.empty:
                corr_df.to_excel(writer, index=False, sheet_name="correlation_data")
                metrics_df.to_excel(writer, index=False, sheet_name="correlation_metrics")
        print(f"[OK] 汇总结果已写入：{OUTPUT_EXCEL}")
    except ImportError:
        # 没有 openpyxl 时退回 CSV（只写 summary）
        summary_df.to_csv(OUTPUT_CSV, index=False)
        print(
            f"[WARN] 未安装 openpyxl，已改为输出 CSV：{OUTPUT_CSV}"
        )


if __name__ == "__main__":
    main()
