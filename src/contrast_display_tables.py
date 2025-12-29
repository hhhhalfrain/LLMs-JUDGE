# -*- coding: utf-8 -*-
"""
contrast_display_tables.py
------------------------------------------------------------
你这次的 Q1 要求（修正版）：

Q1: 在 persona=1 且 interest=1 的情况下，
    每种 method（aggregation / incremental / summary_based）
    从 round=0..8 的“预测得分曲线”（按书籍聚合后的平均分）是什么样？

因此本脚本会额外输出两张表：
- q1_scores_long  : method × round 的均值/方差/书本数
- q1_scores_pivot : 行=round(0..8)，列=method，单元格=mean_pred_score（方便展示/作图）

仍然保留 corr_by_round / q2_persona / q3_interest（用于你原本论文三问里的 Q2/Q3 对比）。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# ✅ 路径与参数区
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_ID = "4agents exp5"
BATCH_ROOT = PROJECT_ROOT / "runs" / "batch" / BATCH_ID
OUTPUTS_ROOT = BATCH_ROOT / "outputs"

INDEX_XLSX = PROJECT_ROOT / "data" / "books" / "index.xlsx"

OUT_XLSX = BATCH_ROOT / "contrast_display_tables.xlsx"

# 相关性计算展示：round 缺失如何处理
ROUND_POLICY = "carry"  # "carry" / "nan"

# agent 聚合：mean_kept=只用 kept=True 的 agent（否则回退 all）；mean_all=全用
AGENT_AGG = "mean_kept"  # "mean_kept" / "mean_all"

# 分数模式：
# - reconstructed：aggregation/incremental 用章节均值(float)做 base + 讨论增量（减少 ties）
# - raw：直接用 score_history 的整数分
SCORE_MODE = "reconstructed"  # "reconstructed" / "raw"

# 展示 rounds（你的 r=8）
ROUNDS_TO_SHOW = list(range(0, 9))  # 0..8

# Q2/Q3 仍然需要一个对比轮次（例如 round3）
DISC_ROUND_TO_COMPARE = 3


# ============================================================
# ✅ 你在这里手写“要排除的书名”
# ============================================================
EXCLUDED_TITLES = [
    "The Midnight Feast",
    "The Frozen River",
    "Meet Me at the Lake",
    "Fourth Wing (The Empyrean, #1)",
    "The Crash",
    "Home Is Where the Bodies Are",
    "Happy Place",
    "A Fate Inked in Blood (Saga of the Unfated, #1)",
    "We Used to Live Here",
    "The Will of the Many (Hierarchy, #1)",
    "Quicksilver (Fae & Alchemy, #1)",
    "Katabasis",
    "The Pumpkin Spice Café (Dream Harbor, #1)",
    "Beautiful Ugly",
    "The Covenant of Water",
    "Wildfire (Maple Hills, #2)",
    "Great Big Beautiful Life",
    "All the Colors of the Dark",
    "House of Flame and Shadow (Crescent City, #3)",
    "Sunrise on the Reaping",
    "Bury Our Bones in the Midnight Soil",
    "First Lie Wins",
    "First-Time Caller (Heartstrings, #1)",
    "Alchemised",
    "Yellowface",
    "Leather & Lark (The Ruinous Love Trilogy, #2)",
    "Things We Hide from the Light (Knockemout, #2)",
    "Deep End",
    "Intermezzo",
    "The Teacher",
]


# ============================================================
# 数据结构
# ============================================================
@dataclass(frozen=True)
class ExpParams:
    method: str
    persona: int
    disc: int
    interest: int
    r_max: int
    w: int
    na: int
    sd: int
    das: int
    cbs: int


@dataclass(frozen=True)
class ExpKey:
    params: ExpParams

    def as_id(self) -> str:
        p = self.params
        return (
            f"m={p.method}"
            f"__persona={p.persona}"
            f"__disc={p.disc}"
            f"__interest={p.interest}"
            f"__r={p.r_max}"
            f"__w={p.w}"
            f"__na={p.na}"
            f"__sd={p.sd}"
            f"__das={p.das}"
            f"__cbs={p.cbs}"
        )


# ============================================================
# 字符串/列名工具
# ============================================================
def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s)
    return s


def _title_variants(title: str) -> List[str]:
    t = (title or "").strip()
    out = set()
    out.add(_norm(t))

    t2 = re.sub(r"\s*\([^)]*#\d+\)\s*$", "", t).strip()
    if t2 and t2 != t:
        out.add(_norm(t2))

    t3 = re.sub(r"\s*\([^)]*\)\s*$", "", t2).strip()
    if t3 and t3 != t2:
        out.add(_norm(t3))

    return [x for x in out if x]


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low_map = {str(c).strip().lower(): str(c) for c in cols}
    for c in candidates:
        if c.strip().lower() in low_map:
            return low_map[c.strip().lower()]
    for col in cols:
        cl = str(col).strip().lower()
        for c in candidates:
            if c.strip().lower() in cl:
                return str(col)
    return None


# ============================================================
# 解析 experiment 参数
# ============================================================
EXP_RE = re.compile(
    r"m=(?P<method>aggregation|incremental|summary_based)"
    r"__persona=(?P<persona>\d+)"
    r"__disc=(?P<disc>\d+)"
    r"__interest=(?P<interest>\d+)"
    r"__r=(?P<r>\d+)"
    r"__w=(?P<w>\d+)"
    r"__na=(?P<na>\d+)"
    r"__sd=(?P<sd>\d+)"
    r"__das=(?P<das>\d+)"
    r"__cbs=(?P<cbs>\d+)"
)


def _extract_exp_fragment(dirname: str) -> Optional[str]:
    if dirname.startswith("m="):
        return dirname
    if "__m=" in dirname:
        i = dirname.find("__m=")
        return dirname[i + 2 :]
    return None


def parse_exp_params(dirname: str) -> Optional[ExpKey]:
    frag = _extract_exp_fragment(dirname)
    if not frag:
        return None
    m = EXP_RE.search(frag)
    if not m:
        return None
    p = ExpParams(
        method=str(m.group("method")),
        persona=int(m.group("persona")),
        disc=int(m.group("disc")),
        interest=int(m.group("interest")),
        r_max=int(m.group("r")),
        w=int(m.group("w")),
        na=int(m.group("na")),
        sd=int(m.group("sd")),
        das=int(m.group("das")),
        cbs=int(m.group("cbs")),
    )
    return ExpKey(params=p)


# ============================================================
# 读取 index.xlsx（Goodreads）
# ============================================================
def load_goodreads_index(xlsx: Path) -> Tuple[Dict[str, float], pd.DataFrame]:
    if not xlsx.exists():
        raise FileNotFoundError(f"index.xlsx not found: {xlsx}")

    df = pd.read_excel(xlsx)

    title_col = _find_col(df, ["title", "book_name", "name"])
    rating_col = _find_col(df, ["goodreads_rating", "rating", "score"])
    if title_col is None or rating_col is None:
        raise RuntimeError(f"index.xlsx 列名不匹配：{list(df.columns)}")

    mp: Dict[str, float] = {}
    for _, row in df.iterrows():
        t = str(row.get(title_col, "") or "")
        r = _safe_float(row.get(rating_col, float("nan")))
        if not t or not math.isfinite(r):
            continue
        for k in _title_variants(t):
            mp.setdefault(k, float(r))
    return mp, df


# ============================================================
# 从 result.json 提取：每本书、每个实验、每个 round 的预测分
# ============================================================
def _score_from_history(score_history: List[Dict[str, Any]], r: int, policy: str) -> float:
    if not isinstance(score_history, list) or not score_history:
        return float("nan")
    items = []
    for it in score_history:
        rr = int(it.get("round", -999999))
        sc = _safe_float(it.get("score", float("nan")))
        if rr >= 0 and math.isfinite(sc):
            items.append((rr, sc))
    if not items:
        return float("nan")
    items.sort(key=lambda x: x[0])

    if policy == "nan":
        for rr, sc in items:
            if rr == r:
                return float(sc)
        return float("nan")

    last = None
    for rr, sc in items:
        if rr <= r:
            last = sc
        else:
            break
    return float(last) if last is not None else float("nan")


def _agent_base_float(agent: Dict[str, Any], method: str) -> float:
    pre = _safe_float(agent.get("pre_discussion_score", float("nan")))
    if method == "aggregation":
        evs = agent.get("chapter_evals", None)
        if isinstance(evs, list) and evs:
            ss = [_safe_float(x.get("score", float("nan"))) for x in evs if isinstance(x, dict)]
            ss = [x for x in ss if math.isfinite(x)]
            if ss:
                return float(np.mean(ss))
    if method == "incremental":
        steps = agent.get("incremental_steps", None)
        if isinstance(steps, list) and steps:
            ss = [_safe_float(x.get("score", float("nan"))) for x in steps if isinstance(x, dict)]
            ss = [x for x in ss if math.isfinite(x)]
            if ss:
                return float(np.mean(ss))
    if math.isfinite(pre):
        return float(pre)
    return 3.0


def _agent_score_at_round(agent: Dict[str, Any], method: str, r: int) -> float:
    hist = agent.get("score_history", []) or []
    rs = _score_from_history(hist, r, ROUND_POLICY)
    if not math.isfinite(rs):
        return float("nan")

    if SCORE_MODE == "raw":
        return float(rs)

    base = _agent_base_float(agent, method)
    pre = _safe_float(agent.get("pre_discussion_score", base))
    if not math.isfinite(pre) or not math.isfinite(base):
        return float("nan")
    return float(base + (rs - pre))


def _book_pred_score(result_obj: Dict[str, Any], exp: ExpKey, r: int) -> float:
    agents = result_obj.get("agents", []) or []
    if not isinstance(agents, list) or not agents:
        return float("nan")

    if AGENT_AGG == "mean_kept":
        kept = [a for a in agents if bool(a.get("kept", True))]
        pool = kept if kept else agents
    else:
        pool = agents

    ss = []
    for a in pool:
        sc = _agent_score_at_round(a, exp.params.method, r)
        if math.isfinite(sc):
            ss.append(sc)
    if not ss:
        return float("nan")
    return float(np.mean(ss))


# ============================================================
# 扫描 outputs：收集 (book, exp, round)->pred
# ============================================================
def scan_outputs(outputs_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs root not found: {outputs_root}")

    rows = []
    book_rows = []

    excluded_norm = set()
    for t in EXCLUDED_TITLES:
        for k in _title_variants(t):
            excluded_norm.add(k)

    dirs = [p for p in outputs_root.iterdir() if p.is_dir()]
    for d in dirs:
        exp = parse_exp_params(d.name)
        if exp is None:
            continue
        res_path = d / "result.json"
        if not res_path.exists():
            continue

        try:
            obj = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        book_name = str(obj.get("book_name", "") or "").strip()
        if not book_name:
            md = obj.get("metadata", {}) or {}
            book_name = str(md.get("book_name", "") or md.get("title", "") or "").strip()
        if not book_name:
            continue

        book_keys = _title_variants(book_name)
        if any(k in excluded_norm for k in book_keys):
            continue

        book_rows.append({"book": book_name, "book_norm": book_keys[0] if book_keys else _norm(book_name)})

        r_show = [r for r in ROUNDS_TO_SHOW if r <= exp.params.r_max]
        for rr in r_show:
            pred = _book_pred_score(obj, exp, rr)
            if not math.isfinite(pred):
                continue
            p = exp.params
            rows.append({
                "book": book_name,
                "book_norm": book_keys[0] if book_keys else _norm(book_name),
                "method": p.method,
                "persona": p.persona,
                "disc": p.disc,
                "interest": p.interest,
                "r_max": p.r_max,
                "w": p.w,
                "na": p.na,
                "sd": p.sd,
                "das": p.das,
                "cbs": p.cbs,
                "round": int(rr),
                "pred_score": float(pred),
            })

    df_pred = pd.DataFrame(rows)
    df_books = pd.DataFrame(book_rows).drop_duplicates(subset=["book_norm"]).reset_index(drop=True)
    return df_pred, df_books


# ============================================================
# 相关性计算（用于 Q2/Q3 以及 corr_by_round）
# ============================================================
def calc_corr(truth: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.isfinite(truth) & np.isfinite(pred)
    if int(np.sum(m)) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rho, rp = spearmanr(truth[m], pred[m])
    tau, tp = kendalltau(truth[m], pred[m])
    return float(rho), float(rp), float(tau), float(tp)


def build_corr_by_round(df_pred: pd.DataFrame, gr_map: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_pred.copy()
    df["goodreads_rating"] = df["book"].map(lambda x: _safe_float(gr_map.get(_title_variants(x)[0], float("nan"))))

    df_missing = df.loc[~np.isfinite(df["goodreads_rating"].to_numpy()), ["book"]].drop_duplicates().reset_index(drop=True)

    group_cols = ["method", "persona", "disc", "interest", "r_max", "w", "na", "sd", "das", "cbs", "round"]
    out_rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        truth = g["goodreads_rating"].to_numpy(dtype=float)
        pred = g["pred_score"].to_numpy(dtype=float)
        n_eff = int(np.sum(np.isfinite(truth) & np.isfinite(pred)))
        uniq = int(pd.Series(pred[np.isfinite(pred)]).nunique())

        rho, rp, tau, tp = calc_corr(truth, pred)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_books_effective": n_eff,
            "n_unique_scores": uniq,
            "spearman_rho": rho,
            "spearman_p": rp,
            "kendall_tau": tau,
            "kendall_p": tp,
        })
        out_rows.append(row)

    corr_df = pd.DataFrame(out_rows)
    corr_df = corr_df.sort_values(
        by=["method", "persona", "interest", "disc", "r_max", "round"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)
    return corr_df, df_missing


# ============================================================
# ✅ 新版 Q1：persona=1 & interest=1 时，各 method 在 round=0..8 的“得分曲线”
# ============================================================
def build_q1_scores_tables(df_pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    输出：
      q1_scores_long:
        method, round, n_books, mean_pred_score, std_pred_score, min, max
      q1_scores_pivot:
        行=round, 列=method, 值=mean_pred_score
    """
    df = df_pred.copy()
    df = df[(df["persona"] == 1) & (df["interest"] == 1)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg = (
        df.groupby(["method", "round"], dropna=False)
          .agg(
              n_books=("book_norm", "nunique"),
              mean_pred_score=("pred_score", "mean"),
              std_pred_score=("pred_score", "std"),
              min_pred_score=("pred_score", "min"),
              max_pred_score=("pred_score", "max"),
          )
          .reset_index()
    )

    # 补齐 round=0..8（避免某 method 某 round 缺一行）
    methods = sorted(df["method"].unique().tolist())
    full_idx = pd.MultiIndex.from_product([methods, ROUNDS_TO_SHOW], names=["method", "round"])
    agg = agg.set_index(["method", "round"]).reindex(full_idx).reset_index()

    # pivot：展示用（round 行，method 列）
    pivot = agg.pivot(index="round", columns="method", values="mean_pred_score").reset_index()
    pivot.columns.name = None

    # 排序更直观
    agg = agg.sort_values(by=["method", "round"]).reset_index(drop=True)
    pivot = pivot.sort_values(by=["round"]).reset_index(drop=True)
    return agg, pivot


# ============================================================
# Q2/Q3（相关性对比表）
# ============================================================
def build_q2_persona_table(corr_by_round: pd.DataFrame) -> pd.DataFrame:
    r = int(DISC_ROUND_TO_COMPARE)
    base_cols = ["method", "disc", "interest", "r_max", "w", "na", "sd", "das", "cbs", "round"]

    df = corr_by_round[corr_by_round["round"] == r].copy()
    df = df[df["interest"] == 0].copy()

    a = df[df["persona"] == 1].copy().rename(columns={
        "spearman_rho": "rho_p1",
        "spearman_p": "p_p1",
        "kendall_tau": "tau_p1",
        "kendall_p": "tp_p1",
        "n_books_effective": "n_p1",
        "n_unique_scores": "uniq_p1",
    })
    b = df[df["persona"] == 0].copy().rename(columns={
        "spearman_rho": "rho_p0",
        "spearman_p": "p_p0",
        "kendall_tau": "tau_p0",
        "kendall_p": "tp_p0",
        "n_books_effective": "n_p0",
        "n_unique_scores": "uniq_p0",
    })

    m = a.merge(b, on=base_cols, how="inner")
    m["delta_rho_persona"] = m["rho_p1"] - m["rho_p0"]
    m["delta_tau_persona"] = m["tau_p1"] - m["tau_p0"]
    m = m.sort_values(by=["delta_rho_persona"], ascending=False, na_position="last").reset_index(drop=True)
    return m


def build_q3_interest_table(corr_by_round: pd.DataFrame) -> pd.DataFrame:
    r = int(DISC_ROUND_TO_COMPARE)
    base_cols = ["method", "persona", "disc", "r_max", "w", "na", "sd", "das", "cbs", "round"]

    df = corr_by_round[corr_by_round["round"] == r].copy()
    df = df[df["persona"] == 1].copy()

    a = df[df["interest"] == 1].copy().rename(columns={
        "spearman_rho": "rho_i1",
        "spearman_p": "p_i1",
        "kendall_tau": "tau_i1",
        "kendall_p": "tp_i1",
        "n_books_effective": "n_i1",
        "n_unique_scores": "uniq_i1",
    })
    b = df[df["interest"] == 0].copy().rename(columns={
        "spearman_rho": "rho_i0",
        "spearman_p": "p_i0",
        "kendall_tau": "tau_i0",
        "kendall_p": "tp_i0",
        "n_books_effective": "n_i0",
        "n_unique_scores": "uniq_i0",
    })

    m = a.merge(b, on=base_cols, how="inner")
    m["delta_rho_interest"] = m["rho_i1"] - m["rho_i0"]
    m["delta_tau_interest"] = m["tau_i1"] - m["tau_i0"]
    m = m.sort_values(by=["delta_rho_interest"], ascending=False, na_position="last").reset_index(drop=True)
    return m


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    print(f"[INFO] outputs_root={OUTPUTS_ROOT}")
    print(f"[INFO] index_xlsx={INDEX_XLSX}")
    print(f"[INFO] out_xlsx={OUT_XLSX}")
    print(f"[INFO] ROUND_POLICY={ROUND_POLICY} AGENT_AGG={AGENT_AGG} SCORE_MODE={SCORE_MODE}")

    gr_map, _ = load_goodreads_index(INDEX_XLSX)
    df_pred, df_books = scan_outputs(OUTPUTS_ROOT)
    if df_pred.empty:
        raise RuntimeError("扫描不到任何可用的 result.json（或全部被排除/无分数）。")

    # 相关性（Q2/Q3 仍然要）
    corr_by_round, missing_goodreads = build_corr_by_round(df_pred, gr_map)

    # ✅ Q1：得分曲线（persona=1 & interest=1）
    q1_long, q1_pivot = build_q1_scores_tables(df_pred)

    # Q2/Q3
    q2 = build_q2_persona_table(corr_by_round)
    q3 = build_q3_interest_table(corr_by_round)

    excluded_df = pd.DataFrame({"excluded_title": EXCLUDED_TITLES})
    excluded_df["excluded_norm"] = excluded_df["excluded_title"].map(
        lambda x: _title_variants(x)[0] if _title_variants(x) else _norm(x)
    )

    cfg_df = pd.DataFrame([{
        "batch_id": BATCH_ID,
        "round_policy": ROUND_POLICY,
        "agent_agg": AGENT_AGG,
        "score_mode": SCORE_MODE,
        "rounds_to_show": ",".join(map(str, ROUNDS_TO_SHOW)),
        "disc_round_to_compare": int(DISC_ROUND_TO_COMPARE),
        "excluded_count": int(len(EXCLUDED_TITLES)),
        "included_books": int(len(df_books)),
        "pred_rows": int(len(df_pred)),
    }])

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        cfg_df.to_excel(w, sheet_name="config", index=False)
        excluded_df.to_excel(w, sheet_name="excluded_titles", index=False)

        df_books.to_excel(w, sheet_name="books_included", index=False)
        missing_goodreads.to_excel(w, sheet_name="missing_goodreads", index=False)

        # 明细（debug 用）
        df_pred.to_excel(w, sheet_name="pred_long", index=False)

        # 相关性总表
        corr_by_round.to_excel(w, sheet_name="corr_by_round", index=False)

        # ✅ Q1：得分曲线（你要的核心）
        q1_long.to_excel(w, sheet_name="q1_scores_long", index=False)
        q1_pivot.to_excel(w, sheet_name="q1_scores_pivot", index=False)

        # Q2/Q3：对比用
        q2.to_excel(w, sheet_name="q2_persona", index=False)
        q3.to_excel(w, sheet_name="q3_interest", index=False)

    print(f"[DONE] wrote: {OUT_XLSX}")
    print(f"[DONE] included_books={len(df_books)} | pred_rows={len(df_pred)} | corr_rows={len(corr_by_round)}")


if __name__ == "__main__":
    main()
