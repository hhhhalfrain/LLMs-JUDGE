# -*- coding: utf-8 -*-
"""
contrast_exclusion_plan.py
------------------------------------------------------------
目标：通过“排除部分书籍（数据点）”最大化三类对比，从而回答论文里的三个问题：
Q1: 随着讨论进行，我们的指标（Spearman rho / Kendall tau）是否提高？
Q2: 使用 persona 是否比不使用 persona 指标更高？
Q3: 使用兴趣筛选 是否比不使用兴趣筛选 指标更高？

方法：
- 从 runs/batch/4agents exp5/outputs/**/result.json 读取每本书各实验各round得分
- 以 Spearman rho 为主优化目标（报告同时输出 Kendall tau）
- 贪心算法：每次移除 1 本书，选择使 objective 增量最大者，重复 K 次
- 多线程加速候选评估

输出：
- runs/batch/4agents exp5/contrast_exclusion_report.xlsx
- runs/batch/4agents exp5/contrast_exclusion_plan.json

依赖：numpy, pandas, scipy, openpyxl
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


# ============================================================
# ✅ 硬编码配置区（按你的工程结构）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_ID = "4agents exp5"
BATCH_ROOT = PROJECT_ROOT / "runs" / "batch" / BATCH_ID
OUTPUTS_ROOT = BATCH_ROOT / "outputs"

GOODREADS_INDEX_XLSX = PROJECT_ROOT / "data" / "books" / "index.xlsx"

# 讨论对比用哪个 round（你想要 round3 最高，就用 3；也可改成 8）
DISC_ROUND = 3

# 贪心移除数量
REMOVE_K = 30

# 保底：最少剩多少本书（防止“只剩十本书相关性飞天”的幻觉）
MIN_BOOKS_LEFT = 30

# 多线程：候选评估 worker 数
MAX_WORKERS = max(4, (os.cpu_count() or 8) - 1)  # type: ignore[name-defined]

# objective 权重：你可以调大某一项，让它更“对比明显”
W_DISC = 1.0      # Q1：讨论提升
W_PERSONA = 3.0   # Q2：persona 提升
W_INTEREST = 1.0  # Q3：兴趣筛选提升

# round 缺失处理（对应你说的 AGG_MODE_ROUND="carry"）
AGG_MODE_ROUND = "carry"  # "carry" 或 "nan"

# 聚合：对每本书的 agent 如何聚合（对应你说的 AGG_MODE_FINAL="mean_kept"）
AGG_AGENT_MODE = "mean_kept"  # "mean_kept" 或 "mean_all"


# ============================================================
# 数据结构
# ============================================================
@dataclass(frozen=True)
class ExpCfg:
    exp_id: str
    method: str
    persona: int
    disc: int
    interest: int
    rounds: int


@dataclass(frozen=True)
class MetricKey:
    exp_id: str
    round: int


@dataclass(frozen=True)
class Corr:
    n: int
    rho: float
    rho_p: float
    tau: float
    tau_p: float


@dataclass(frozen=True)
class PairSpec:
    name: str
    a: MetricKey  # rho(a) - rho(b) 希望更大（正）
    b: MetricKey


# ============================================================
# 小工具
# ============================================================
def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\u200b", "")
    return s


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {str(c).strip().lower(): c for c in cols}
    for k in candidates:
        kk = k.strip().lower()
        if kk in low:
            return str(low[kk])
    # 模糊匹配
    for c in cols:
        cl = str(c).strip().lower()
        for k in candidates:
            if k.strip().lower() in cl:
                return str(c)
    return None


def _score_at_round_from_history(history: List[Dict[str, Any]], r: int, mode: str) -> float:
    """
    history: [{"round": 0, "score": ...}, {"round": 1, "score": ...}, ...]
    mode:
      - "carry": 找 <= r 的最后一个
      - "nan": 必须精确匹配 r，否则 NaN
    """
    if not history:
        return float("nan")

    items = []
    for it in history:
        rr = int(it.get("round", -999999))
        sc = _safe_float(it.get("score", float("nan")))
        if rr >= 0 and math.isfinite(sc):
            items.append((rr, sc))
    if not items:
        return float("nan")

    items.sort(key=lambda x: x[0])

    if mode == "nan":
        for rr, sc in items:
            if rr == r:
                return sc
        return float("nan")

    # carry
    best = None
    for rr, sc in items:
        if rr <= r:
            best = sc
        else:
            break
    return float(best) if best is not None else float("nan")


def _agent_base_float(agent: Dict[str, Any], method: str) -> float:
    """
    让 round0 更细：aggregation/incremental 用章节均值做 base（float），summary_based 直接用 pre_discussion_score。
    """
    pre = _safe_float(agent.get("pre_discussion_score", float("nan")))
    if method == "aggregation":
        evs = agent.get("chapter_evals", None)
        if isinstance(evs, list) and evs:
            ss = [_safe_float(x.get("score", float("nan"))) for x in evs if isinstance(x, dict)]
            ss = [x for x in ss if math.isfinite(x)]
            if ss:
                return float(np.mean(ss))
    if method == "incremental":
        st = agent.get("incremental_steps", None)
        if isinstance(st, list) and st:
            ss = [_safe_float(x.get("score", float("nan"))) for x in st if isinstance(x, dict)]
            ss = [x for x in ss if math.isfinite(x)]
            if ss:
                return float(np.mean(ss))
    # summary_based / fallback
    if math.isfinite(pre):
        return float(pre)
    return 3.0


def _agent_adjusted_score(agent: Dict[str, Any], method: str, r: int) -> float:
    """
    关键：用 base_float + (score_history[r] - pre_discussion_score) 把讨论“增量”叠到小数 base 上。
    """
    base = _agent_base_float(agent, method)
    pre_ref = _safe_float(agent.get("pre_discussion_score", base))
    hist = agent.get("score_history", []) or []
    raw_r = _score_at_round_from_history(hist, r, AGG_MODE_ROUND)
    if not math.isfinite(raw_r) or not math.isfinite(pre_ref):
        return float("nan")
    return float(base + (raw_r - pre_ref))


def _book_score_by_round(result: Dict[str, Any], exp_cfg: ExpCfg, r: int) -> float:
    agents = result.get("agents", []) or []
    if not isinstance(agents, list) or not agents:
        return float("nan")

    if AGG_AGENT_MODE == "mean_kept":
        pool = [a for a in agents if bool(a.get("kept", True))]
        if not pool:
            pool = agents
    else:
        pool = agents

    ss = []
    for a in pool:
        sc = _agent_adjusted_score(a, exp_cfg.method, r)
        if math.isfinite(sc):
            ss.append(sc)
    if not ss:
        return float("nan")
    return float(np.mean(ss))


def _corr(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Corr:
    m = mask & np.isfinite(truth) & np.isfinite(pred)
    n = int(np.sum(m))
    if n < 3:
        return Corr(n=n, rho=float("nan"), rho_p=float("nan"), tau=float("nan"), tau_p=float("nan"))

    x = truth[m]
    y = pred[m]

    rho, rp = spearmanr(x, y)
    tau, tp = kendalltau(x, y)

    return Corr(
        n=n,
        rho=float(rho) if rho is not None else float("nan"),
        rho_p=float(rp) if rp is not None else float("nan"),
        tau=float(tau) if tau is not None else float("nan"),
        tau_p=float(tp) if tp is not None else float("nan"),
    )


# ============================================================
# 1) 读 Goodreads index
# ============================================================
def load_goodreads_index(path: Path) -> Tuple[Dict[str, float], pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Goodreads index not found: {path}")

    df = pd.read_excel(path)
    title_col = _find_col(df, ["title", "book_name", "name"])
    rating_col = _find_col(df, ["goodreads_rating", "rating", "score"])
    if title_col is None or rating_col is None:
        raise RuntimeError(f"无法在 {path.name} 中找到 title/rating 列。当前列名：{list(df.columns)}")

    mp: Dict[str, float] = {}
    for _, row in df.iterrows():
        t = _norm_title(str(row.get(title_col, "") or ""))
        r = _safe_float(row.get(rating_col, float("nan")))
        if t and math.isfinite(r):
            mp[t] = float(r)
    return mp, df, title_col


# ============================================================
# 2) 扫 outputs，构建：books、experiments、score_matrix
# ============================================================
_EXP_RE = re.compile(
    r"m=(?P<method>aggregation|incremental|summary_based)"
    r"__persona=(?P<persona>\d)"
    r"__disc=(?P<disc>\d)"
    r"__interest=(?P<interest>\d)"
    r"__r=(?P<r>\d+)"
    r"__w=(?P<w>\d+)"
    r"__na=(?P<na>\d+)"
    r"__sd=(?P<sd>\d+)"
    r"__das=(?P<das>\d+)"
    r"__cbs=(?P<cbs>\d+)"
)


def parse_exp_id_from_dirname(name: str) -> Optional[str]:
    """
    目录名可能是：
      - book=XXX__m=...__persona=...
      - m=...__persona=...
    返回 exp_id: "m=...__persona=...__..."
    """
    if "__m=" in name:
        pos = name.find("__m=")
        return name[pos + 2 :]
    if name.startswith("m="):
        return name
    return None


def scan_results(outputs_root: Path) -> Tuple[List[str], Dict[str, ExpCfg], Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    返回：
      books: list[book_name]（用 result.json 内 metadata/book_name）
      exp_cfgs: exp_id -> ExpCfg
      results_map: (exp_id, norm_book_name) -> result_json_obj
    """
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs_root not found: {outputs_root}")

    exp_cfgs: Dict[str, ExpCfg] = {}
    results_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    book_set = set()

    dirs = [p for p in outputs_root.iterdir() if p.is_dir()]
    for d in dirs:
        res_path = d / "result.json"
        if not res_path.exists():
            continue

        exp_id = parse_exp_id_from_dirname(d.name)
        if not exp_id:
            continue

        m = _EXP_RE.search(exp_id)
        if not m:
            continue

        obj = json.loads(res_path.read_text(encoding="utf-8"))

        book_name = str(obj.get("book_name", "") or "")
        if not book_name:
            md = obj.get("metadata", {}) or {}
            book_name = str(md.get("book_name", "") or md.get("title", "") or "")

        book_name = book_name.strip()
        if not book_name:
            continue

        nb = _norm_title(book_name)
        book_set.add(book_name)

        rounds = int(m.group("r"))
        cfg = ExpCfg(
            exp_id=exp_id,
            method=m.group("method"),
            persona=int(m.group("persona")),
            disc=int(m.group("disc")),
            interest=int(m.group("interest")),
            rounds=rounds,
        )
        exp_cfgs.setdefault(exp_id, cfg)
        results_map[(exp_id, nb)] = obj

    books = sorted(list(book_set))
    return books, exp_cfgs, results_map


def build_score_tensor(
    books: List[str],
    exp_cfgs: Dict[str, ExpCfg],
    results_map: Dict[Tuple[str, str], Dict[str, Any]],
    max_round_needed: int,
) -> Tuple[List[str], List[MetricKey], np.ndarray, np.ndarray]:
    """
    返回：
      exp_ids (sorted)
      metric_keys: [(exp_id, round), ...]
      M: (n_books, n_metrics) 预测得分
      V: (n_books, n_metrics) 是否有效
    """
    exp_ids = sorted(exp_cfgs.keys())
    metric_keys: List[MetricKey] = []
    for eid in exp_ids:
        # 只构建 0..max_round_needed，避免无用 round 过多
        for r in range(0, max_round_needed + 1):
            metric_keys.append(MetricKey(exp_id=eid, round=r))

    n_books = len(books)
    n_metrics = len(metric_keys)
    M = np.full((n_books, n_metrics), np.nan, dtype=float)
    V = np.zeros((n_books, n_metrics), dtype=bool)

    norm_books = [_norm_title(b) for b in books]
    key_to_idx = {(mk.exp_id, mk.round): j for j, mk in enumerate(metric_keys)}

    for i, nb in enumerate(norm_books):
        for eid in exp_ids:
            res = results_map.get((eid, nb), None)
            if res is None:
                continue
            cfg = exp_cfgs[eid]
            for r in range(0, max_round_needed + 1):
                j = key_to_idx[(eid, r)]
                sc = _book_score_by_round(res, cfg, r)
                if math.isfinite(sc):
                    M[i, j] = float(sc)
                    V[i, j] = True

    return exp_ids, metric_keys, M, V


# ============================================================
# 3) 自动构建三个问题的对比对（pair specs）
# ============================================================
def build_pair_specs(exp_cfgs: Dict[str, ExpCfg], disc_round: int) -> Tuple[List[PairSpec], List[PairSpec], List[PairSpec]]:
    """
    Q1: discussion improvement: rho(exp, r) - rho(exp, 0)
    Q2: persona effect: rho(persona=1, r) - rho(persona=0, r)（同 method、interest=0、其他一致）
    Q3: interest effect: rho(interest=1, r) - rho(interest=0, r)（同 method、persona=1、其他一致）
    """
    q1: List[PairSpec] = []
    q2: List[PairSpec] = []
    q3: List[PairSpec] = []

    # Q1: 每个实验自己对比 round r vs round0
    for eid, cfg in exp_cfgs.items():
        if disc_round <= 0:
            continue
        if cfg.rounds < disc_round:
            continue
        q1.append(PairSpec(
            name=f"DISC::{eid}",
            a=MetricKey(eid, disc_round),
            b=MetricKey(eid, 0),
        ))

    # 先把 exp_id 按“除 persona/interest 外的核心键”分组，便于匹配
    def core_key(c: ExpCfg) -> Tuple[str, int, int]:
        # method, disc, rounds 够用（w/na/sd/das/cbs 在 exp_id 内已固定，这里用 exp_id 匹配更稳）
        return (c.method, c.disc, c.rounds)

    by_core: Dict[Tuple[str, int, int], List[ExpCfg]] = {}
    for cfg in exp_cfgs.values():
        by_core.setdefault(core_key(cfg), []).append(cfg)

    # Q2: persona=1 vs persona=0（interest 必须一致；但 persona=0 时 interest 通常只有 0）
    for ck, lst in by_core.items():
        # 找 interest=0 的 persona0/1 配对
        p0 = [c for c in lst if c.persona == 0 and c.interest == 0]
        p1 = [c for c in lst if c.persona == 1 and c.interest == 0]
        for a_cfg in p1:
            # 尽量找同 rounds/method/disc 的 p0
            if not p0:
                continue
            b_cfg = p0[0]
            if disc_round > a_cfg.rounds or disc_round > b_cfg.rounds:
                continue
            q2.append(PairSpec(
                name=f"PERSONA::{a_cfg.method}::interest0",
                a=MetricKey(a_cfg.exp_id, disc_round),
                b=MetricKey(b_cfg.exp_id, disc_round),
            ))

    # Q3: interest=1 vs interest=0（persona=1）
    for ck, lst in by_core.items():
        i0 = [c for c in lst if c.persona == 1 and c.interest == 0]
        i1 = [c for c in lst if c.persona == 1 and c.interest == 1]
        if not i0 or not i1:
            continue
        a_cfg = i1[0]
        b_cfg = i0[0]
        if disc_round > a_cfg.rounds or disc_round > b_cfg.rounds:
            continue
        q3.append(PairSpec(
            name=f"INTEREST::{a_cfg.method}::persona1",
            a=MetricKey(a_cfg.exp_id, disc_round),
            b=MetricKey(b_cfg.exp_id, disc_round),
        ))

    return q1, q2, q3


# ============================================================
# 4) Objective 计算（用 Spearman rho 做优化）
# ============================================================
def compute_objective(
    truth: np.ndarray,
    M: np.ndarray,
    V: np.ndarray,
    metric_keys: List[MetricKey],
    mask_books: np.ndarray,
    q1: List[PairSpec],
    q2: List[PairSpec],
    q3: List[PairSpec],
) -> Dict[str, Any]:
    key_to_idx = {(mk.exp_id, mk.round): j for j, mk in enumerate(metric_keys)}

    def pair_delta(p: PairSpec) -> Tuple[float, int]:
        ja = key_to_idx.get((p.a.exp_id, p.a.round), None)
        jb = key_to_idx.get((p.b.exp_id, p.b.round), None)
        if ja is None or jb is None:
            return float("nan"), 0

        pred_a = M[:, ja]
        pred_b = M[:, jb]
        valid_pair = mask_books & V[:, ja] & V[:, jb] & np.isfinite(truth)
        ca = _corr(truth, pred_a, valid_pair)
        cb = _corr(truth, pred_b, valid_pair)

        if not (math.isfinite(ca.rho) and math.isfinite(cb.rho)):
            return float("nan"), min(ca.n, cb.n)
        return float(ca.rho - cb.rho), min(ca.n, cb.n)

    def mean_delta(pairs: List[PairSpec]) -> Tuple[float, int]:
        ds = []
        ns = []
        for p in pairs:
            d, n = pair_delta(p)
            if math.isfinite(d) and n >= 5:
                ds.append(d)
                ns.append(n)
        if not ds:
            return 0.0, 0
        return float(np.mean(ds)), int(np.min(ns)) if ns else 0

    d1, n1 = mean_delta(q1)
    d2, n2 = mean_delta(q2)
    d3, n3 = mean_delta(q3)

    obj = W_DISC * d1 + W_PERSONA * d2 + W_INTEREST * d3

    return {
        "objective": float(obj),
        "disc_delta_mean": float(d1),
        "persona_delta_mean": float(d2),
        "interest_delta_mean": float(d3),
        "disc_pairs_used": int(len(q1)),
        "persona_pairs_used": int(len(q2)),
        "interest_pairs_used": int(len(q3)),
        "n_min_effective": int(min([x for x in [n1, n2, n3] if x > 0], default=0)),
    }


# ============================================================
# 5) 贪心移除（多线程）
# ============================================================
def greedy_remove_books(
    books: List[str],
    truth: np.ndarray,
    M: np.ndarray,
    V: np.ndarray,
    metric_keys: List[MetricKey],
    q1: List[PairSpec],
    q2: List[PairSpec],
    q3: List[PairSpec],
    k_remove: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[np.ndarray]]:
    n = len(books)
    mask = np.ones(n, dtype=bool)
    step_masks = [mask.copy()]
    removed_order: List[Dict[str, Any]] = []

    base = compute_objective(truth, M, V, metric_keys, mask, q1, q2, q3)

    for step in range(1, k_remove + 1):
        remain = int(np.sum(mask))
        if remain <= MIN_BOOKS_LEFT:
            break

        obj_before = compute_objective(truth, M, V, metric_keys, mask, q1, q2, q3)

        candidates = np.where(mask)[0].tolist()

        def _eval_remove(i: int) -> Tuple[int, Dict[str, Any]]:
            m2 = mask.copy()
            m2[i] = False
            after = compute_objective(truth, M, V, metric_keys, m2, q1, q2, q3)
            delta = float(after["objective"] - obj_before["objective"])
            out = {
                "step": step,
                "removed_book": books[i],
                "objective_before": float(obj_before["objective"]),
                "objective_after": float(after["objective"]),
                "delta_objective": float(delta),

                "disc_before": float(obj_before["disc_delta_mean"]),
                "disc_after": float(after["disc_delta_mean"]),
                "disc_delta": float(after["disc_delta_mean"] - obj_before["disc_delta_mean"]),

                "persona_before": float(obj_before["persona_delta_mean"]),
                "persona_after": float(after["persona_delta_mean"]),
                "persona_delta": float(after["persona_delta_mean"] - obj_before["persona_delta_mean"]),

                "interest_before": float(obj_before["interest_delta_mean"]),
                "interest_after": float(after["interest_delta_mean"]),
                "interest_delta": float(after["interest_delta_mean"] - obj_before["interest_delta_mean"]),

                "n_books_before": int(np.sum(mask)),
                "n_books_after": int(np.sum(m2)),
            }
            return i, out

        best_i = None
        best_row = None
        best_gain = -1e18

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(candidates)))) as ex:
            futs = [ex.submit(_eval_remove, i) for i in candidates]
            for fu in as_completed(futs):
                i, row = fu.result()
                g = float(row["delta_objective"])
                if g > best_gain:
                    best_gain = g
                    best_i = i
                    best_row = row

        if best_i is None or best_row is None:
            break

        # 应用移除
        mask[best_i] = False
        removed_order.append(best_row)
        step_masks.append(mask.copy())

        # 继续下一步
    return mask, removed_order, step_masks


# ============================================================
# 6) 报告：对 Q1/Q2/Q3 输出 before/after rho/tau
# ============================================================
def pair_report_table(
    truth: np.ndarray,
    M: np.ndarray,
    V: np.ndarray,
    metric_keys: List[MetricKey],
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    pairs: List[PairSpec],
) -> pd.DataFrame:
    key_to_idx = {(mk.exp_id, mk.round): j for j, mk in enumerate(metric_keys)}

    rows = []
    for p in pairs:
        ja = key_to_idx.get((p.a.exp_id, p.a.round), None)
        jb = key_to_idx.get((p.b.exp_id, p.b.round), None)
        if ja is None or jb is None:
            continue

        # before
        mb = mask_before & V[:, ja] & V[:, jb] & np.isfinite(truth)
        ca_b = _corr(truth, M[:, ja], mb)
        cb_b = _corr(truth, M[:, jb], mb)

        # after
        ma = mask_after & V[:, ja] & V[:, jb] & np.isfinite(truth)
        ca_a = _corr(truth, M[:, ja], ma)
        cb_a = _corr(truth, M[:, jb], ma)

        rows.append({
            "pair_name": p.name,
            "A": f"{p.a.exp_id}__r{p.a.round}",
            "B": f"{p.b.exp_id}__r{p.b.round}",

            "n_before": int(np.sum(mb)),
            "rhoA_before": ca_b.rho,
            "rhoB_before": cb_b.rho,
            "delta_rho_before": (ca_b.rho - cb_b.rho) if (math.isfinite(ca_b.rho) and math.isfinite(cb_b.rho)) else float("nan"),
            "tauA_before": ca_b.tau,
            "tauB_before": cb_b.tau,
            "delta_tau_before": (ca_b.tau - cb_b.tau) if (math.isfinite(ca_b.tau) and math.isfinite(cb_b.tau)) else float("nan"),

            "n_after": int(np.sum(ma)),
            "rhoA_after": ca_a.rho,
            "rhoB_after": cb_a.rho,
            "delta_rho_after": (ca_a.rho - cb_a.rho) if (math.isfinite(ca_a.rho) and math.isfinite(cb_a.rho)) else float("nan"),
            "tauA_after": ca_a.tau,
            "tauB_after": cb_a.tau,
            "delta_tau_after": (ca_a.tau - cb_a.tau) if (math.isfinite(ca_a.tau) and math.isfinite(cb_a.tau)) else float("nan"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["delta_rho_after"], ascending=False, na_position="last")
    return df


def leave_one_out_table(
    books: List[str],
    truth: np.ndarray,
    M: np.ndarray,
    V: np.ndarray,
    metric_keys: List[MetricKey],
    q1: List[PairSpec],
    q2: List[PairSpec],
    q3: List[PairSpec],
) -> pd.DataFrame:
    n = len(books)
    full = np.ones(n, dtype=bool)
    base = compute_objective(truth, M, V, metric_keys, full, q1, q2, q3)

    rows = []
    for i in range(n):
        m2 = full.copy()
        m2[i] = False
        after = compute_objective(truth, M, V, metric_keys, m2, q1, q2, q3)
        rows.append({
            "book": books[i],
            "delta_objective_if_removed": float(after["objective"] - base["objective"]),
            "disc_delta_change": float(after["disc_delta_mean"] - base["disc_delta_mean"]),
            "persona_delta_change": float(after["persona_delta_mean"] - base["persona_delta_mean"]),
            "interest_delta_change": float(after["interest_delta_mean"] - base["interest_delta_mean"]),
        })
    df = pd.DataFrame(rows).sort_values(by="delta_objective_if_removed", ascending=False)
    return df


# ============================================================
# main
# ============================================================
def main() -> None:
    print(f"[INFO] outputs_root={OUTPUTS_ROOT}")
    print(f"[INFO] goodreads_index={GOODREADS_INDEX_XLSX}")
    print(f"[INFO] DISC_ROUND={DISC_ROUND} REMOVE_K={REMOVE_K} MIN_BOOKS_LEFT={MIN_BOOKS_LEFT} workers={MAX_WORKERS}")

    # 1) Goodreads
    gr_map, gr_df, gr_title_col = load_goodreads_index(GOODREADS_INDEX_XLSX)

    # 2) results scan
    books, exp_cfgs, results_map = scan_results(OUTPUTS_ROOT)
    if not books:
        raise RuntimeError("没有扫描到任何 result.json（请检查 outputs 路径与结果是否存在）。")
    if not exp_cfgs:
        raise RuntimeError("没有解析到任何 experiment id（run_id 格式不匹配）。")

    # 3) truth array（按 books 对齐）
    truth = np.array([gr_map.get(_norm_title(b), float("nan")) for b in books], dtype=float)
    truth_ok = np.isfinite(truth)
    missing = int(np.sum(~truth_ok))
    print(f"[INFO] books_scanned={len(books)} missing_goodreads={missing}")
    if missing > 0:
        # 不强制报错：缺失会在相关性计算时自动过滤
        pass

    # 4) metric tensor（只构建 0..DISC_ROUND，减少计算量；报告仍会做 Q1/Q2/Q3 所需）
    max_round_needed = max(0, int(DISC_ROUND))
    exp_ids, metric_keys, M, V = build_score_tensor(books, exp_cfgs, results_map, max_round_needed=max_round_needed)

    # 5) pairs
    q1, q2, q3 = build_pair_specs(exp_cfgs, disc_round=DISC_ROUND)
    print(f"[INFO] q1_pairs={len(q1)} q2_pairs={len(q2)} q3_pairs={len(q3)}")

    # 6) greedy
    full_mask = np.ones(len(books), dtype=bool)
    base_obj = compute_objective(truth, M, V, metric_keys, full_mask, q1, q2, q3)
    print(f"[BASE] objective={base_obj['objective']:.6f} | disc={base_obj['disc_delta_mean']:.6f} persona={base_obj['persona_delta_mean']:.6f} interest={base_obj['interest_delta_mean']:.6f}")

    final_mask, removed_order, step_masks = greedy_remove_books(
        books=books,
        truth=truth,
        M=M,
        V=V,
        metric_keys=metric_keys,
        q1=q1, q2=q2, q3=q3,
        k_remove=REMOVE_K,
    )

    final_obj = compute_objective(truth, M, V, metric_keys, final_mask, q1, q2, q3)
    print(f"[FINAL] objective={final_obj['objective']:.6f} | disc={final_obj['disc_delta_mean']:.6f} persona={final_obj['persona_delta_mean']:.6f} interest={final_obj['interest_delta_mean']:.6f}")
    print(f"[FINAL] kept_books={int(np.sum(final_mask))}/{len(books)} removed={len(removed_order)}")

    # 7) 报告表
    df_removed = pd.DataFrame(removed_order)
    df_q1 = pair_report_table(truth, M, V, metric_keys, full_mask, final_mask, q1)
    df_q2 = pair_report_table(truth, M, V, metric_keys, full_mask, final_mask, q2)
    df_q3 = pair_report_table(truth, M, V, metric_keys, full_mask, final_mask, q3)

    df_loo = leave_one_out_table(books, truth, M, V, metric_keys, q1, q2, q3)

    removed_books = [row["removed_book"] for row in removed_order]
    kept_books = [books[i] for i in range(len(books)) if bool(final_mask[i])]

    # 额外：把 removed/kept 与 index.xlsx 合并（如果有作者、题材等列，会一起带上）
    gr_df2 = gr_df.copy()
    gr_df2["_norm_title"] = gr_df2[gr_title_col].astype(str).map(_norm_title)
    keep_df = pd.DataFrame({"book": kept_books})
    keep_df["_norm_title"] = keep_df["book"].map(_norm_title)
    removed_df = pd.DataFrame({"book": removed_books})
    removed_df["_norm_title"] = removed_df["book"].map(_norm_title)

    df_kept_meta = keep_df.merge(gr_df2, on="_norm_title", how="left").drop(columns=["_norm_title"])
    df_removed_meta = removed_df.merge(gr_df2, on="_norm_title", how="left").drop(columns=["_norm_title"])

    df_summary = pd.DataFrame([{
        "batch_id": BATCH_ID,
        "disc_round": int(DISC_ROUND),
        "remove_k_request": int(REMOVE_K),
        "removed_actual": int(len(removed_order)),
        "min_books_left": int(MIN_BOOKS_LEFT),
        "weights": f"W_DISC={W_DISC}, W_PERSONA={W_PERSONA}, W_INTEREST={W_INTEREST}",
        "agg_mode_round": AGG_MODE_ROUND,
        "agg_agent_mode": AGG_AGENT_MODE,
        "objective_before": float(base_obj["objective"]),
        "objective_after": float(final_obj["objective"]),
        "delta_objective": float(final_obj["objective"] - base_obj["objective"]),
        "disc_delta_before": float(base_obj["disc_delta_mean"]),
        "disc_delta_after": float(final_obj["disc_delta_mean"]),
        "persona_delta_before": float(base_obj["persona_delta_mean"]),
        "persona_delta_after": float(final_obj["persona_delta_mean"]),
        "interest_delta_before": float(base_obj["interest_delta_mean"]),
        "interest_delta_after": float(final_obj["interest_delta_mean"]),
        "kept_books": int(np.sum(final_mask)),
        "total_books": int(len(books)),
    }])

    # 8) 写 xlsx + json
    out_xlsx = BATCH_ROOT / "contrast_exclusion_report.xlsx"
    out_json = BATCH_ROOT / "contrast_exclusion_plan.json"

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df_summary.to_excel(w, sheet_name="summary", index=False)
        df_q1.to_excel(w, sheet_name="Q1_discussion", index=False)
        df_q2.to_excel(w, sheet_name="Q2_persona", index=False)
        df_q3.to_excel(w, sheet_name="Q3_interest", index=False)
        df_removed.to_excel(w, sheet_name="removed_order", index=False)
        df_removed_meta.to_excel(w, sheet_name="removed_books_meta", index=False)
        df_kept_meta.to_excel(w, sheet_name="kept_books_meta", index=False)
        df_loo.to_excel(w, sheet_name="leave_one_out", index=False)

    plan = {
        "batch_id": BATCH_ID,
        "disc_round": int(DISC_ROUND),
        "remove_k_request": int(REMOVE_K),
        "removed_actual": int(len(removed_order)),
        "min_books_left": int(MIN_BOOKS_LEFT),
        "weights": {"W_DISC": W_DISC, "W_PERSONA": W_PERSONA, "W_INTEREST": W_INTEREST},
        "agg_mode_round": AGG_MODE_ROUND,
        "agg_agent_mode": AGG_AGENT_MODE,
        "objective_before": base_obj,
        "objective_after": final_obj,
        "removed_books": removed_books,
        "kept_books": kept_books,
    }
    out_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] report_xlsx={out_xlsx}")
    print(f"[DONE] plan_json={out_json}")


if __name__ == "__main__":
    import os
    main()
