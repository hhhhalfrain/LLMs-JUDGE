# -*- coding: utf-8 -*-
"""
calc_rho_tau_from_outputs.py
------------------------------------------------------------
从 runs/batch/4agents exp5/outputs/*/result.json 读取每本书每个实验的分数，
再和 data/books/index.xlsx 的 GoodreadsRatings 对齐，计算：
- Spearman rho (+ p-value)
- Kendall tau-b (+ p-value)

说明：
- 假设 scipy 已安装；不再保留任何降级分支
- 全程按 float 处理（不会 int 截断）
- 输出 Excel：runs/batch/4agents exp5/rank_corr_rho_tau.xlsx
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# ✅ CONFIG（按你的要求：写死，不走 CLI）
# ============================================================

BATCH_ID = "4agents exp5"

# final 聚合口径：mean_kept / mean_all
AGG_MODE_FINAL = "mean_kept"

# round 聚合：
# - "carry": 用 <=r 的最后一次分数（若缺轮次就 carry）
# - "active_only": 只用“该轮有显式 round 记录”的 agent
AGG_MODE_ROUND = "carry"

# round 聚合时，agent 口径是否只算 kept（建议 True，和 mean_kept 对齐）
ROUND_ONLY_KEPT = True

COMPUTE_BY_ROUND = True

# 输出文件
OUT_XLSX_NAME = "rank_corr_rho_tau.xlsx"


# ============================================================
# root 自动探测：脚本放 src/ 也能跑
# ============================================================
def find_project_root(start: Path, max_up: int = 6) -> Path:
    cur = start.resolve()
    for _ in range(max_up):
        cand = cur
        if (cand / "data").exists() and (cand / "runs").exists():
            return cand
        cur = cur.parent
    return start.resolve().parent


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
BOOKS_XLSX = PROJECT_ROOT / "data" / "books" / "index.xlsx"
OUTPUTS_ROOT = PROJECT_ROOT / "runs" / "batch" / BATCH_ID / "outputs"
OUT_XLSX = PROJECT_ROOT / "runs" / "batch" / BATCH_ID / OUT_XLSX_NAME


# ============================================================
# 工具函数
# ============================================================
def norm_title(s: Any) -> str:
    return str(s or "").strip()


def norm_title_key(s: Any) -> str:
    t = norm_title(s).lower()
    t = " ".join(t.split())
    return t


def as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_goodreads_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower().strip(): c for c in df.columns}

    title_col = cols.get("title") or cols.get("book_name") or cols.get("name")
    gr_col = cols.get("goodreadsratings") or cols.get("goodreadsrating")

    if not title_col or not gr_col:
        raise RuntimeError(
            f"index.xlsx 列名不符合预期。需要 Title + GoodreadsRatings。\n实际列：{list(df.columns)}"
        )

    out = df[[title_col, gr_col]].copy()
    out.columns = ["Title", "GoodreadsRatings"]
    out["Title"] = out["Title"].apply(norm_title)
    out["TitleKey"] = out["Title"].apply(norm_title_key)
    out["GoodreadsRatings"] = out["GoodreadsRatings"].apply(as_float)
    out = out.dropna(subset=["TitleKey", "GoodreadsRatings"])
    return out


def experiment_signature(result: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    cfg = (result.get("config", {}) or {}).get("experiment", {}) or {}
    keys = [
        "method",
        "use_persona",
        "use_discussion",
        "use_interest_filter",
        "discussion_rounds",
        "discussion_window",
        "n_agents",
        "score_decimals",
        "discussion_affects_score",
        "chapter_batch_size",
    ]
    return tuple((k, cfg.get(k, None)) for k in keys)


def sig_to_str(sig: Tuple[Tuple[str, Any], ...]) -> str:
    kv = dict(sig)

    def b(x: Any) -> int:
        return int(bool(x))

    return "__".join(
        [
            f"m={kv.get('method')}",
            f"persona={b(kv.get('use_persona'))}",
            f"disc={b(kv.get('use_discussion'))}",
            f"interest={b(kv.get('use_interest_filter'))}",
            f"r={kv.get('discussion_rounds')}",
            f"w={kv.get('discussion_window')}",
            f"na={kv.get('n_agents')}",
            f"sd={kv.get('score_decimals')}",
            f"das={b(kv.get('discussion_affects_score'))}",
            f"cbs={kv.get('chapter_batch_size')}",
        ]
    )


def method_of(result: Dict[str, Any]) -> str:
    cfg = (result.get("config", {}) or {}).get("experiment", {}) or {}
    return str(cfg.get("method", "") or "")


def agent_base_score(agent: Dict[str, Any], method: str) -> float:
    """
    round=0 的“讨论前分数”（float）：
    - 优先用 pre_discussion_score
    - 若缺失：从 chapter_evals / incremental_steps 现算均值
    """
    pre = as_float(agent.get("pre_discussion_score", None))
    if not math.isnan(pre):
        return float(pre)

    if method == "aggregation":
        ev = agent.get("chapter_evals", None)
        if isinstance(ev, list) and ev:
            scores = [as_float(x.get("score")) for x in ev if isinstance(x, dict)]
            scores = [s for s in scores if not math.isnan(s)]
            if scores:
                return float(sum(scores) / len(scores))

    if method == "incremental":
        st = agent.get("incremental_steps", None)
        if isinstance(st, list) and st:
            scores = [as_float(x.get("score")) for x in st if isinstance(x, dict)]
            scores = [s for s in scores if not math.isnan(s)]
            if scores:
                return float(sum(scores) / len(scores))

    return 3.0


def agent_score_at_round(agent: Dict[str, Any], r: int, method: str, mode_round: str) -> Optional[float]:
    """
    取 agent 在第 r 轮的分数（float，不截断）：
    - r==0：base_score
    - r>0：
        - carry：取 <=r 的最近一次 score_history；若没有则回退 base_score
        - active_only：必须存在 round==r 的记录，否则 None
    """
    base = agent_base_score(agent, method)
    if r <= 0:
        return base

    hist = agent.get("score_history", []) or []
    if not isinstance(hist, list):
        return base if mode_round == "carry" else None

    best_rr = -10**9
    best_sc: Optional[float] = None
    have_exact = False

    for it in hist:
        if not isinstance(it, dict):
            continue
        rr = it.get("round", None)
        sc = it.get("score", None)
        if rr is None or sc is None:
            continue
        try:
            rr_i = int(rr)
        except Exception:
            continue
        sc_f = as_float(sc)
        if math.isnan(sc_f):
            continue

        if rr_i == r:
            have_exact = True
            best_sc = float(sc_f)
            break

        if rr_i <= r and rr_i > best_rr:
            best_rr = rr_i
            best_sc = float(sc_f)

    if mode_round == "active_only":
        return best_sc if have_exact else None

    # carry
    return best_sc if best_sc is not None else base


def aggregate_final_score(result: Dict[str, Any], mode_final: str) -> Tuple[float, int]:
    agents = result.get("agents", []) or []
    vals: List[float] = []

    for a in agents:
        if not isinstance(a, dict):
            continue
        if mode_final == "mean_kept" and not bool(a.get("kept", True)):
            continue

        sc = as_float(a.get("post_discussion_score", None))
        if not math.isnan(sc):
            vals.append(float(sc))

    if not vals:
        return float("nan"), 0
    return float(sum(vals) / len(vals)), len(vals)


def aggregate_round_score(result: Dict[str, Any], r: int) -> Tuple[float, int]:
    agents = result.get("agents", []) or []
    m = method_of(result)
    vals: List[float] = []

    for a in agents:
        if not isinstance(a, dict):
            continue
        if ROUND_ONLY_KEPT and not bool(a.get("kept", True)):
            continue

        sc = agent_score_at_round(a, r=r, method=m, mode_round=AGG_MODE_ROUND)
        if sc is None or math.isnan(sc):
            continue
        vals.append(float(sc))

    if not vals:
        return float("nan"), 0
    return float(sum(vals) / len(vals)), len(vals)


def rank_corr(x: List[float], y: List[float]) -> Dict[str, Any]:
    out = {
        "n": int(len(x)),
        "n_unique_x": int(len(set(x))),
        "n_unique_y": int(len(set(y))),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "kendall_tau": float("nan"),
        "kendall_p": float("nan"),
        "note": "",
    }

    if len(x) < 3:
        out["note"] = "n<3"
        return out
    if out["n_unique_x"] < 2:
        out["note"] = "n_unique_x<2 (scores constant)"
        return out
    if out["n_unique_y"] < 2:
        out["note"] = "n_unique_y<2 (truth constant)"
        return out

    rho, p_rho = spearmanr(x, y)
    # SciPy 默认 kendalltau 的 variant='b'（tau-b，含 ties 修正）
    tau, p_tau = kendalltau(x, y)

    out["spearman_rho"] = float(rho)
    out["spearman_p"] = float(p_rho)
    out["kendall_tau"] = float(tau)
    out["kendall_p"] = float(p_tau)
    return out


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    if not BOOKS_XLSX.exists():
        raise RuntimeError(f"找不到 Goodreads 表：{BOOKS_XLSX}")
    if not OUTPUTS_ROOT.exists():
        raise RuntimeError(f"找不到 outputs：{OUTPUTS_ROOT}")

    books_df = parse_goodreads_xlsx(BOOKS_XLSX)
    gr_map = dict(zip(books_df["TitleKey"].tolist(), books_df["GoodreadsRatings"].tolist()))

    results: List[Dict[str, Any]] = []
    for d in sorted(OUTPUTS_ROOT.glob("*")):
        if not d.is_dir():
            continue
        rj = d / "result.json"
        obj = safe_read_json(rj)
        if obj:
            results.append(obj)

    if not results:
        raise RuntimeError(f"在 {OUTPUTS_ROOT} 下没有扫描到任何 result.json")

    rows_final: List[Dict[str, Any]] = []
    rows_round: List[Dict[str, Any]] = []
    rows_stats: List[Dict[str, Any]] = []
    miss_gr = 0

    for obj in results:
        book = norm_title(obj.get("book_name", ""))
        if not book:
            continue
        key = norm_title_key(book)
        goodreads = gr_map.get(key, None)
        if goodreads is None:
            miss_gr += 1
            continue

        exp = sig_to_str(experiment_signature(obj))

        sc_final, n_used_final = aggregate_final_score(obj, mode_final=AGG_MODE_FINAL)
        rows_final.append(
            {
                "experiment": exp,
                "book": book,
                "goodreads": float(goodreads),
                "score": float(sc_final),
                "n_used_agents": int(n_used_final),
            }
        )

        if COMPUTE_BY_ROUND:
            rounds = int(((obj.get("aggregate", {}) or {}).get("discussion_rounds", 0)) or 0)
            for r in range(0, rounds + 1):
                sc_r, n_used_r = aggregate_round_score(obj, r=r)
                rows_round.append(
                    {
                        "experiment": exp,
                        "round": int(r),
                        "book": book,
                        "goodreads": float(goodreads),
                        "score": float(sc_r),
                        "n_used_agents": int(n_used_r),
                    }
                )

    df_final = pd.DataFrame(rows_final)
    if df_final.empty:
        raise RuntimeError(
            "df_final 为空：说明 outputs 的 book_name 无法和 index.xlsx 的 Title 匹配。\n"
            "建议检查：index.xlsx 的 Title 是否与 result.json 的 book_name 一致（或至少大小写/空格一致）。"
        )

    # corr_final
    corr_final_rows: List[Dict[str, Any]] = []
    for exp, g in df_final.groupby("experiment"):
        gg = g.dropna(subset=["goodreads", "score"])
        x = gg["score"].astype(float).tolist()
        y = gg["goodreads"].astype(float).tolist()
        stat = rank_corr(x, y)
        corr_final_rows.append(
            {
                "experiment": exp,
                "n_books": stat["n"],
                "n_unique_scores": stat["n_unique_x"],
                "spearman_rho": stat["spearman_rho"],
                "spearman_p": stat["spearman_p"],
                "kendall_tau": stat["kendall_tau"],
                "kendall_p": stat["kendall_p"],
                "note": stat["note"],
                "agg_mode_final": AGG_MODE_FINAL,
            }
        )

    corr_final = (
        pd.DataFrame(corr_final_rows)
        .sort_values(["spearman_rho", "kendall_tau"], ascending=False, na_position="last")
    )

    # corr_by_round
    corr_round = None
    if COMPUTE_BY_ROUND and rows_round:
        df_r = pd.DataFrame(rows_round)
        corr_round_rows: List[Dict[str, Any]] = []

        for (exp, rr), g in df_r.groupby(["experiment", "round"]):
            gg = g.dropna(subset=["goodreads", "score"])
            x = gg["score"].astype(float).tolist()
            y = gg["goodreads"].astype(float).tolist()
            stat = rank_corr(x, y)
            corr_round_rows.append(
                {
                    "experiment": exp,
                    "round": int(rr),
                    "n_books": stat["n"],
                    "n_unique_scores": stat["n_unique_x"],
                    "spearman_rho": stat["spearman_rho"],
                    "spearman_p": stat["spearman_p"],
                    "kendall_tau": stat["kendall_tau"],
                    "kendall_p": stat["kendall_p"],
                    "note": stat["note"],
                    "agg_mode_round": AGG_MODE_ROUND,
                    "round_only_kept": bool(ROUND_ONLY_KEPT),
                }
            )

        corr_round = (
            pd.DataFrame(corr_round_rows)
            .sort_values(["round", "spearman_rho"], ascending=[True, False], na_position="last")
        )

    # round_stats：解释为什么某些 round 仍可能 NaN（比如常数分/有效样本太少）
    if COMPUTE_BY_ROUND and rows_round:
        df_r = pd.DataFrame(rows_round)
        for (exp, rr), g in df_r.groupby(["experiment", "round"]):
            gg = g.dropna(subset=["goodreads", "score"])
            rows_stats.append(
                {
                    "experiment": exp,
                    "round": int(rr),
                    "n_rows_total": int(len(g)),
                    "n_rows_valid": int(len(gg)),
                    "n_unique_scores_valid": int(gg["score"].nunique(dropna=True)),
                    "n_unique_truth_valid": int(gg["goodreads"].nunique(dropna=True)),
                    "n_used_agents_min": int(gg["n_used_agents"].min()) if len(gg) else 0,
                    "n_used_agents_mean": float(gg["n_used_agents"].mean()) if len(gg) else 0.0,
                }
            )

    stats_df = pd.DataFrame(rows_stats) if rows_stats else pd.DataFrame()

    # 导出 Excel
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        books_df[["Title", "GoodreadsRatings"]].to_excel(w, index=False, sheet_name="books_truth")
        df_final.to_excel(w, index=False, sheet_name="book_scores_final")
        corr_final.to_excel(w, index=False, sheet_name="corr_final")
        if corr_round is not None:
            corr_round.to_excel(w, index=False, sheet_name="corr_by_round")
        if not stats_df.empty:
            stats_df.to_excel(w, index=False, sheet_name="round_stats")

    print(f"[OK] wrote: {OUT_XLSX}")
    print(f"[INFO] results_scanned={len(results)} | missing_goodreads={miss_gr}")
    print("\n[Top 10 experiments by Spearman rho]")
    print(corr_final.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
