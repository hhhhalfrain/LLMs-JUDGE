# -*- coding: utf-8 -*-
"""
batch_run.py  (实验级断点续跑 + 多端点负载均衡 + STAGE2 结果复用)
------------------------------------------------------------
断点续跑策略：
- 若 outputs/<run_id>/_DONE.json 存在：跳过该实验
- 若 outputs/<run_id>/ 存在但没有 _DONE.json：删掉整个目录后重跑

新增：端点槽位池（每个千问端点可同时跑多个“书籍大任务”，数量可配置）
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from queue import Queue

from dotenv import load_dotenv

from src.runner import (
    setup_logger,
    ThreadLocalLLM,
    AdaptiveConcurrencyController,
    evaluate_single_book,
)


# ============================================================
# ✅ 硬编码配置区
# ============================================================
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOOKS_MERGED_JSON = PROJECT_ROOT / "data" / "books" / "merged_books_fixed.json"
PERSONAS_JSON = PROJECT_ROOT / "data" / "personas_sample.json"

# 续跑关键：固定这个目录名，重复运行会在同一批次上跳过/删除重跑
BATCH_ID = "4agents exp1"
BATCH_ROOT = PROJECT_ROOT / "runs" / "batch" / BATCH_ID
OUTPUTS_ROOT = BATCH_ROOT / "outputs"
BASE_EVAL_ROOT = BATCH_ROOT / "base_eval"  # 全局基线评测缓存（跨 experiment 复用）

# 多个千问端点（你需要填成真实的 base_url）
QWEN_BASE_URLS = [
    "http://10.130.71.2:30071/v1",
    "http://10.130.71.2:30921/v1",
    "http://10.130.71.2:30781/v1",
    "http://10.130.71.2:30116/v1",
    "http://10.130.71.2:30810/v1",
]

# ============================================================
# ✅ 每个千问端点可同时跑几个“书籍大任务”（可配置）
# ============================================================

# 方案 A：所有端点统一容量（推荐先用这个）
QWEN_TASKS_PER_ENDPOINT = 4

# 方案 B：逐端点配置（启用则覆盖方案 A）
# 1) 用 list：长度必须等于 QWEN_BASE_URLS
QWEN_TASKS_PER_ENDPOINT_LIST = None
# 示例：
# QWEN_TASKS_PER_ENDPOINT_LIST = [2, 2, 1, 3, 2]

# 2) 或者用 dict：没写的端点回退到 QWEN_TASKS_PER_ENDPOINT
QWEN_TASKS_PER_ENDPOINT_MAP = None
# 示例：
# QWEN_TASKS_PER_ENDPOINT_MAP = {
#     "http://10.130.71.2:30071/v1": 2,
#     "http://10.130.71.2:30921/v1": 3,
# }

# 顶层 worker 总上限（再怎么有 slots 也不会超过这个）
MAX_EXPERIMENT_WORKERS = 999999

# 错峰启动（建议小一点；0 代表不做错峰）
START_STAGGER_SEC = 0

# ✅ 章节合并 batchsize
CHAPTER_BATCH_SIZE = 1   # <= 1 表示不合并（保持原样）

# LLM 配置（千问 32B，DashScope compatible）
LLM_MODEL = "qwen-32b"
API_KEY_ENV = "QWEN_API_KEY"

LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 10000
LLM_TIMEOUT_SEC = 1500

# 请求级重试
RETRY_MAX_ATTEMPTS = 1000
RETRY_BASE_SLEEP_SEC = 1.0
RETRY_MAX_SLEEP_SEC = 30.0
RETRY_JITTER = 0.2
FAIL_FAST = True

# runner 内部：同一本书 agent 并发的上限（自适应控制会在这个范围内浮动）
PER_BOOK_AGENT_WORKERS = 4


# -----------------------------
# ✅ 网格维度（全部可扫）
# -----------------------------
METHODS = ["aggregation", "incremental", "summary_based"]
USE_PERSONA_OPTS = [False,True]
USE_INTEREST_FILTER_OPTS = [False,True]

# ✅ 只关心最高讨论轮数：程序内部会记录 round=0..R 的所有分数
DISCUSSION_ROUNDS_OPTS = [8]          # 你可以写 [0,2,4]，但只会取 max=4
DISCUSSION_WINDOW_OPTS = [8]
N_AGENTS_OPTS = [4]
SCORE_DECIMALS_OPTS = [1]
DISCUSSION_AFFECTS_SCORE_OPTS = [True]

LBOOK, RBOOK = 0, 20  # 书本子区间 [LBOOK:RBOOK]，None 表示全量

# ============================================================
# 工具函数
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sanitize_name(s: str, max_len: int = 180) -> str:
    bad = '<>:"/\\|?*'
    s = (s or "").strip()
    for ch in bad:
        s = s.replace(ch, "_")
    s = s.strip().strip(".")
    return s[:max_len] if len(s) > max_len else s


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def build_endpoint_pool(
    base_urls: List[str],
    default_slots: int,
    slots_list: List[int] | None = None,
    slots_map: Dict[str, int] | None = None,
) -> Tuple[Queue, Dict[str, int]]:
    """
    端点槽位池：
    - pool 里放 base_url “令牌”
    - 每个 base_url 放入 slots 个令牌
    - interleave 放入令牌，减少“开局全堆一个端点”的概率
    """
    if not base_urls:
        raise RuntimeError("QWEN_BASE_URLS 为空，无法调度。")

    # 计算每个端点 slots
    slots_by_url: Dict[str, int] = {}

    if slots_list is not None:
        if len(slots_list) != len(base_urls):
            raise RuntimeError(
                f"QWEN_TASKS_PER_ENDPOINT_LIST 长度({len(slots_list)})必须等于 QWEN_BASE_URLS 长度({len(base_urls)})"
            )
        for u, s in zip(base_urls, slots_list):
            slots_by_url[u] = max(0, int(s))
    elif slots_map is not None:
        for u in base_urls:
            s = slots_map.get(u, default_slots)
            slots_by_url[u] = max(0, int(s))
    else:
        for u in base_urls:
            slots_by_url[u] = max(0, int(default_slots))

    pool: Queue = Queue()

    max_k = 0
    for u in base_urls:
        max_k = max(max_k, int(slots_by_url.get(u, 0)))

    # interleave 填充：u0,u1,u2,u0,u1,u2...
    for i in range(max_k):
        for u in base_urls:
            if i < int(slots_by_url.get(u, 0)):
                pool.put(u)

    if pool.qsize() <= 0:
        raise RuntimeError("所有端点 slots 都是 0，pool 为空；请检查 QWEN_TASKS_PER_ENDPOINT 配置。")

    return pool, slots_by_url


def batch_merge_chapters(book_record: Dict[str, Any], batch_size: int, inplace: bool = True) -> Dict[str, Any]:
    """
    将 book_record["chapter"] 或 ["chapters"] 按 batch_size 合并。
    - inplace=True：直接修改原 book_record
    - 兼容字段名：chapter / chapters
    """
    if not isinstance(book_record, dict):
        return book_record
    if batch_size is None or int(batch_size) <= 1:
        return book_record

    chap_key = None
    if isinstance(book_record.get("chapters", None), list):
        chap_key = "chapters"
    elif isinstance(book_record.get("chapter", None), list):
        chap_key = "chapter"
    else:
        return book_record

    if book_record.get("_chapter_batched", False) and book_record.get("_chapter_batch_size", None) == int(batch_size):
        return book_record

    chapters = book_record.get(chap_key, [])
    if not chapters:
        return book_record

    def chap_key_fn(c: Any, idx: int) -> Tuple[int, int]:
        if isinstance(c, dict) and ("Number" in c):
            return (_safe_int(c.get("Number", idx + 1), idx + 1), idx)
        return (idx + 1, idx)

    chapters_sorted = sorted(
        [(c, i) for i, c in enumerate(chapters)],
        key=lambda x: chap_key_fn(x[0], x[1])
    )
    chapters_sorted = [c for c, _ in chapters_sorted]

    merged: List[Dict[str, Any]] = []
    n = len(chapters_sorted)
    bs = int(batch_size)

    for b_idx, start in enumerate(range(0, n, bs), start=1):
        group = chapters_sorted[start:start + bs]

        nums = []
        for c in group:
            if isinstance(c, dict):
                nums.append(_safe_int(c.get("Number", 0), 0))
        nums = [x for x in nums if x > 0]
        if nums:
            lo, hi = min(nums), max(nums)
            rng_str = f"{lo}-{hi}" if lo != hi else f"{lo}"
        else:
            rng_str = f"{start+1}-{min(start+bs, n)}"

        merged_title = f"Chapters {rng_str}"

        parts: List[str] = []
        for c in group:
            if not isinstance(c, dict):
                continue
            cnum = c.get("Number", "")
            ctext = str(c.get("text", "") or "")
            head = f"[Chapter {cnum}]".strip()
            if head:
                parts.append(head)
            if ctext:
                parts.append(ctext)

        merged_text = "\n\n".join(parts).strip()

        merged.append({
            "Number": int(b_idx),
            "title": merged_title,
            "text": merged_text,
        })

    target = book_record if inplace else dict(book_record)
    target[chap_key] = merged

    if chap_key == "chapter":
        target["chapters"] = merged
    else:
        target["chapter"] = merged

    target["_chapter_batched"] = True
    target["_chapter_batch_size"] = int(batch_size)
    target["_chapters_original_count"] = int(len(chapters))
    target["_chapters_batched_count"] = int(len(merged))
    target["_chap_key_used"] = chap_key
    return target


def make_run_id(
    method: str,
    use_persona: bool,
    use_discussion: bool,
    use_interest: bool,
    discussion_rounds: int,
    discussion_window: int,
    n_agents: int,
    score_decimals: int,
    discussion_affects_score: bool,
    chapter_batch_size: int,
) -> str:
    return sanitize_name(
        f"m={method}"
        f"__persona={int(use_persona)}"
        f"__disc={int(use_discussion)}"
        f"__interest={int(use_interest)}"
        f"__r={discussion_rounds}"
        f"__w={discussion_window}"
        f"__na={n_agents}"
        f"__sd={score_decimals}"
        f"__das={int(discussion_affects_score)}"
        f"__cbs={int(chapter_batch_size)}"
    )


def make_book_run_id(
    book_title: str,
    method: str,
    use_persona: bool,
    use_discussion: bool,
    use_interest: bool,
    discussion_rounds: int,
    discussion_window: int,
    n_agents: int,
    score_decimals: int,
    discussion_affects_score: bool,
    chapter_batch_size: int,
) -> str:
    book_tag = sanitize_name(book_title, max_len=80) or "UNKNOWN_BOOK"
    core = make_run_id(
        method=method,
        use_persona=use_persona,
        use_discussion=use_discussion,
        use_interest=use_interest,
        discussion_rounds=discussion_rounds,
        discussion_window=discussion_window,
        n_agents=n_agents,
        score_decimals=score_decimals,
        discussion_affects_score=discussion_affects_score,
        chapter_batch_size=chapter_batch_size,
    )
    return sanitize_name(f"book={book_tag}__{core}")


def build_cfg(
    output_root: Path,
    base_url: str,
    method: str,
    use_persona: bool,
    use_discussion: bool,
    use_interest: bool,
    discussion_rounds: int,
    discussion_window: int,
    n_agents: int,
    score_decimals: int,
    discussion_affects_score: bool,
    chapter_batch_size: int,
) -> Any:
    """
    构造 runner 需要的 cfg（属性访问，不依赖 yaml/CLI）
    """
    return SimpleNamespace(
        llm=SimpleNamespace(
            base_url=str(base_url),
            api_key_env=API_KEY_ENV,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS,
            timeout_sec=LLM_TIMEOUT_SEC,

            retry_max_attempts=RETRY_MAX_ATTEMPTS,
            retry_base_sleep_sec=RETRY_BASE_SLEEP_SEC,
            retry_max_sleep_sec=RETRY_MAX_SLEEP_SEC,
            retry_jitter=RETRY_JITTER,
            fail_fast=FAIL_FAST,
        ),
        experiment=SimpleNamespace(
            method=method,
            use_persona=use_persona,
            use_discussion=use_discussion,
            use_interest_filter=use_interest,
            discussion_rounds=discussion_rounds,
            discussion_window=discussion_window,
            n_agents=n_agents,
            score_decimals=score_decimals,
            discussion_affects_score=discussion_affects_score,
            chapter_batch_size=int(chapter_batch_size),
        ),
        concurrency=SimpleNamespace(
            max_workers=PER_BOOK_AGENT_WORKERS
        ),
        paths=SimpleNamespace(
            output_root=str(output_root),
            base_eval_root=str(BASE_EVAL_ROOT),
        )
    )


def build_experiments_grid() -> List[Tuple]:
    """
    改动：
    - 移除 USE_DISCUSSION_OPTS
    - 只关心最高 DISCUSSION_ROUNDS_OPTS（rr_max）
    - use_discussion 由 rr_max>0 自动决定
    - rr_max=0 时等价于 USE_DISCUSSION=False
    """
    exps: List[Tuple] = []
    seen = set()

    rr_max = max([int(x) for x in (DISCUSSION_ROUNDS_OPTS or [0])])
    rr_list = [rr_max]

    for m in METHODS:
        for up in USE_PERSONA_OPTS:
            up = bool(up)
            for ui in USE_INTEREST_FILTER_OPTS:
                ui_eff = bool(ui) if up else False

                for rr in rr_list:
                    rr = int(rr)
                    ud = bool(rr > 0)   # ✅ 由 rr 决定是否启用讨论

                    for ww in DISCUSSION_WINDOW_OPTS:
                        ww = int(ww)
                        for na in N_AGENTS_OPTS:
                            na = int(na)
                            for sd in SCORE_DECIMALS_OPTS:
                                sd = int(sd)
                                for das in DISCUSSION_AFFECTS_SCORE_OPTS:
                                    das = bool(das)

                                    key = (m, up, ud, ui_eff, rr, ww, na, sd, das)
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    exps.append((m, up, ud, ui_eff, rr, ww, na, sd, das))
    return exps




# ============================================================
# 核心：实验级“检查/跳过/删掉重跑”
# ============================================================
def prepare_experiment_dir(exp_root: Path) -> Tuple[bool, Path, Path]:
    """
    返回 (should_run, done_path, running_path)
    - should_run=False 表示已经做过（DONE 存在），直接跳过
    - should_run=True 表示要跑：若目录存在但没 DONE，则删掉重建
    """
    done_path = exp_root / "_DONE.json"
    running_path = exp_root / "_RUNNING.json"

    if done_path.exists():
        return False, done_path, running_path

    if exp_root.exists():
        shutil.rmtree(exp_root, ignore_errors=True)

    safe_mkdir(exp_root)
    safe_mkdir(exp_root / "logs")  # 给 runner 写 jsonl/log 用
    return True, done_path, running_path


# ============================================================
# 单本书“大任务”：顺序跑所有实验配置（绑定一个 base_url）
# ============================================================
def run_all_experiments_for_book(
    book_index: int,
    base_url: str,
    books: List[Dict[str, Any]],
    personas_raw: List[Dict[str, Any]],
    experiments: List[Tuple],
    chapter_batch_size: int,
) -> Tuple[int, str, int, int]:
    """
    对单本书：
    - 绑定一个千问端点 base_url
    - 依次跑 experiments 里的每个实验配置
    - 每个 (book, experiment_cfg) 拥有独立的 run_id / 目录 / _DONE.json
    - 若 _DONE.json 已存在：直接跳过
    返回：(book_index, book_title, n_ran, n_skipped)
    """
    book_record = books[book_index]
    meta = (book_record.get("metadata", {}) or {})
    book_title = str(meta.get("title", "UNKNOWN")).strip() or f"BOOK_{book_index+1}"

    print(f"[TASK] BOOK[{book_index+1}/{len(books)}] {book_title} | base_url={base_url} | exps={len(experiments)}")

    n_ran = 0
    n_skipped = 0

    # 预先合并章节（避免每个实验都再合一次；inplace=False，不污染原 books 列表）
    processed_book = batch_merge_chapters(
        book_record,
        batch_size=int(chapter_batch_size),
        inplace=False,
    )

    for exp_local_idx, (m, up, ud, ui, rr, ww, na, sd, das) in enumerate(experiments):
        run_id = make_book_run_id(
            book_title=book_title,
            method=m,
            use_persona=up,
            use_discussion=ud,
            use_interest=ui,
            discussion_rounds=rr,
            discussion_window=ww,
            n_agents=na,
            score_decimals=sd,
            discussion_affects_score=das,
            chapter_batch_size=chapter_batch_size,
        )

        exp_root = OUTPUTS_ROOT / run_id
        should_run, done_path, running_path = prepare_experiment_dir(exp_root)

        if not should_run:
            print(f"[SKIP EXP] book={book_title} | run_id={run_id} (DONE exists)")
            n_skipped += 1
            continue

        # 轻微错峰：只对每本书的第一个实验做延迟；按端点 index，避免 book_index 越大 sleep 越久
        if START_STAGGER_SEC > 0 and exp_local_idx == 0:
            try:
                url_idx = QWEN_BASE_URLS.index(str(base_url))
            except Exception:
                url_idx = 0
            time.sleep(url_idx * float(START_STAGGER_SEC))

        logger = setup_logger(
            str(exp_root / "logs"),
            also_file=True,
            logger_name=f"llm_trace_{run_id}",
        )

        api_key = os.getenv(API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(f"环境变量 {API_KEY_ENV} 未设置。")

        monitor = AdaptiveConcurrencyController(
            min_workers=1,
            max_workers=int(PER_BOOK_AGENT_WORKERS)
        )

        llm = ThreadLocalLLM(
            base_url=str(base_url),
            api_key=api_key,
            model=LLM_MODEL,
            timeout_sec=int(LLM_TIMEOUT_SEC),
            logger=logger,
            retry_max_attempts=int(RETRY_MAX_ATTEMPTS),
            retry_base_sleep_sec=float(RETRY_BASE_SLEEP_SEC),
            retry_max_sleep_sec=float(RETRY_MAX_SLEEP_SEC),
            retry_jitter=float(RETRY_JITTER),
            fail_fast=bool(FAIL_FAST),
            monitor=monitor,
        )

        personas_used = personas_raw[: int(na)]

        atomic_write_json(
            running_path,
            {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "book_index": book_index,
                "book_title": book_title,
                "base_url": str(base_url),
                "grid": {
                    "method": m,
                    "use_persona": up,
                    "use_discussion": ud,
                    "use_interest_filter": ui,
                    "discussion_rounds": rr,
                    "discussion_window": ww,
                    "n_agents": na,
                    "score_decimals": sd,
                    "discussion_affects_score": das,
                    "chapter_batch_size": int(chapter_batch_size),
                },
                "books": 1,
                "agents": len(personas_used),
            },
        )

        logger.info(
            "=== EXP START === run_id=%s | book=%s | base_url=%s | n_agents=%d | rounds=%d | window=%d | score_decimals=%d | das=%s | chapter_batch_size=%d",
            run_id,
            book_title,
            str(base_url),
            len(personas_used),
            rr,
            ww,
            sd,
            str(das),
            int(chapter_batch_size),
        )

        cfg = build_cfg(
            output_root=exp_root,
            base_url=base_url,
            method=m,
            use_persona=up,
            use_discussion=ud,
            use_interest=ui,
            discussion_rounds=rr,
            discussion_window=ww,
            n_agents=na,
            score_decimals=sd,
            discussion_affects_score=das,
            chapter_batch_size=chapter_batch_size,
        )

        evaluate_single_book(cfg, llm, logger, processed_book, personas_used)

        atomic_write_json(
            done_path,
            {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "book_index": book_index,
                "book_title": book_title,
            },
        )
        try:
            if running_path.exists():
                running_path.unlink()
        except Exception:
            pass

        logger.info("=== EXP END === run_id=%s | book=%s", run_id, book_title)
        n_ran += 1

    return book_index, book_title, n_ran, n_skipped


# ============================================================
# 单个实验：顺序跑所有书（每个大实验绑定一个 base_url）
# ============================================================
def run_one_experiment(
    exp_index: int,
    base_url: str,
    books: List[Dict[str, Any]],
    personas_raw: List[Dict[str, Any]],
    method: str,
    use_persona: bool,
    use_discussion: bool,
    use_interest: bool,
    discussion_rounds: int,
    discussion_window: int,
    n_agents: int,
    score_decimals: int,
    discussion_affects_score: bool,
    chapter_batch_size: int,
) -> str:
    run_id = make_run_id(
        method, use_persona, use_discussion, use_interest,
        discussion_rounds, discussion_window, n_agents,
        score_decimals, discussion_affects_score,
        chapter_batch_size=chapter_batch_size,
    )

    exp_root = OUTPUTS_ROOT / run_id

    should_run, done_path, running_path = prepare_experiment_dir(exp_root)
    if not should_run:
        print(f"[SKIP EXP] {run_id} (DONE exists)")
        return run_id

    if START_STAGGER_SEC > 0:
        time.sleep(exp_index * float(START_STAGGER_SEC))

    logger = setup_logger(str(exp_root / "logs"), also_file=True, logger_name=f"llm_trace_{run_id}")

    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise RuntimeError(f"环境变量 {API_KEY_ENV} 未设置。")

    monitor = AdaptiveConcurrencyController(
        min_workers=1,
        max_workers=int(PER_BOOK_AGENT_WORKERS),
        latency_threshold_s=100.0,
        error_threshold=0.2,
    )

    llm = ThreadLocalLLM(
        base_url=str(base_url),
        api_key=api_key,
        model=LLM_MODEL,
        timeout_sec=int(LLM_TIMEOUT_SEC),
        logger=logger,
        retry_max_attempts=int(RETRY_MAX_ATTEMPTS),
        retry_base_sleep_sec=float(RETRY_BASE_SLEEP_SEC),
        retry_max_sleep_sec=float(RETRY_MAX_SLEEP_SEC),
        retry_jitter=float(RETRY_JITTER),
        fail_fast=bool(FAIL_FAST),
        monitor=monitor,
    )

    personas_used = personas_raw[: int(n_agents)]

    atomic_write_json(running_path, {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "base_url": str(base_url),
        "grid": {
            "method": method,
            "use_persona": use_persona,
            "use_discussion": use_discussion,
            "use_interest_filter": use_interest,
            "discussion_rounds": discussion_rounds,
            "discussion_window": discussion_window,
            "n_agents": n_agents,
            "score_decimals": score_decimals,
            "discussion_affects_score": discussion_affects_score,
            "chapter_batch_size": int(chapter_batch_size),
        },
        "books": len(books),
        "agents": len(personas_used),
    })

    logger.info(
        "=== EXP START === run_id=%s | base_url=%s | books=%d | n_agents=%d | rounds=%d | window=%d | score_decimals=%d | das=%s | chapter_batch_size=%d",
        run_id, str(base_url), len(books), len(personas_used), discussion_rounds, discussion_window, score_decimals,
        str(discussion_affects_score), int(chapter_batch_size)
    )

    cfg = build_cfg(
        output_root=exp_root,
        base_url=base_url,
        method=method,
        use_persona=use_persona,
        use_discussion=use_discussion,
        use_interest=use_interest,
        discussion_rounds=discussion_rounds,
        discussion_window=discussion_window,
        n_agents=n_agents,
        score_decimals=score_decimals,
        discussion_affects_score=discussion_affects_score,
        chapter_batch_size=chapter_batch_size,
    )

    for i, book_record in enumerate(books, start=1):
        title = (book_record.get("metadata", {}) or {}).get("title", "UNKNOWN")

        processed_book = batch_merge_chapters(book_record, batch_size=int(chapter_batch_size))

        orig_cnt = processed_book.get("_chapters_original_count", None)
        bat_cnt = processed_book.get("_chapters_batched_count", None)
        if orig_cnt is not None and bat_cnt is not None and int(chapter_batch_size) > 1:
            logger.info("RUN BOOK [%d/%d] %s | chapters %s -> %s (batch_size=%d)",
                        i, len(books), title, orig_cnt, bat_cnt, int(chapter_batch_size))
        else:
            logger.info("RUN BOOK [%d/%d] %s", i, len(books), title)

        evaluate_single_book(cfg, llm, logger, processed_book, personas_used)

    atomic_write_json(done_path, {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
    })
    try:
        if running_path.exists():
            running_path.unlink()
    except Exception:
        pass

    logger.info("=== EXP END === run_id=%s", run_id)
    return run_id


def main() -> None:
    safe_mkdir(BATCH_ROOT)
    safe_mkdir(OUTPUTS_ROOT)
    safe_mkdir(BASE_EVAL_ROOT)

    books = read_json(BOOKS_MERGED_JSON)
    books = books[LBOOK:RBOOK] if (LBOOK is not None and RBOOK is not None) else books
    personas_raw = read_json(PERSONAS_JSON)
    experiments = build_experiments_grid()

    n_books = len(books)
    n_exps = len(experiments)

    print(f"[BATCH] batch_root={BATCH_ROOT}")
    print(f"[BATCH] outputs_root={OUTPUTS_ROOT}")
    print(f"[BATCH] base_eval_root={BASE_EVAL_ROOT}")
    print(f"[BATCH] books={n_books} personas={len(personas_raw)}")
    print(f"[BATCH] experiments_per_book={n_exps}")
    print(f"[BATCH] chapter_batch_size={CHAPTER_BATCH_SIZE}")

    # 端点槽位池：每个端点可同时跑多个“书籍大任务”
    endpoint_pool, slots_by_url = build_endpoint_pool(
        base_urls=QWEN_BASE_URLS,
        default_slots=int(QWEN_TASKS_PER_ENDPOINT),
        slots_list=QWEN_TASKS_PER_ENDPOINT_LIST,
        slots_map=QWEN_TASKS_PER_ENDPOINT_MAP,
    )

    total_slots = int(endpoint_pool.qsize())
    max_workers = min(int(MAX_EXPERIMENT_WORKERS), int(total_slots), int(n_books))

    print(f"[BATCH] qwen_endpoints={len(QWEN_BASE_URLS)}")
    print(f"[BATCH] endpoint_slots={slots_by_url}")
    print(f"[BATCH] total_slots={total_slots} | max_workers={max_workers}")

    def _run_book_with_endpoint_slot(book_idx: int) -> Tuple[int, str, int, int]:
        """
        每个 book 大任务开始前从 pool 拿一个端点令牌；结束后归还。
        保证：每个 base_url 同时最多运行 slots_by_url[base_url] 个大任务。
        """
        base_url = endpoint_pool.get()
        try:
            return run_all_experiments_for_book(
                book_index=book_idx,
                base_url=base_url,
                books=books,
                personas_raw=personas_raw,
                experiments=experiments,
                chapter_batch_size=CHAPTER_BATCH_SIZE,
            )
        finally:
            endpoint_pool.put(base_url)

    # 顶层并发：worker 总数 = min(总槽位数, 书本数, MAX_EXPERIMENT_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_book_with_endpoint_slot, book_idx) for book_idx in range(n_books)]

        done_books = 0
        for fu in as_completed(futs):
            try:
                book_idx, book_title, n_ran, n_skipped = fu.result()
            except Exception as e:
                done_books += 1
                print(f"[{done_books}/{n_books}] BOOK FUTURE FAILED: {type(e).__name__}: {e}")
                continue

            done_books += 1
            print(f"[{done_books}/{n_books}] BOOK DONE {book_title} | exps_ran={n_ran} | exps_skipped={n_skipped}")


    print(f"[BATCH] ALL BOOKS DONE. Raw outputs under: {OUTPUTS_ROOT}")
    print(f"[BATCH] Base eval cache under: {BASE_EVAL_ROOT}")


def test() -> None:
    """
    纯顺序调试版（不并发），方便本地单步调试。
    """
    safe_mkdir(BATCH_ROOT)
    safe_mkdir(OUTPUTS_ROOT)
    safe_mkdir(BASE_EVAL_ROOT)

    books = read_json(BOOKS_MERGED_JSON)
    personas_raw = read_json(PERSONAS_JSON)
    experiments = build_experiments_grid()

    n_books = len(books)
    n_exps = len(experiments)

    print(f"[BATCH-TEST] batch_root={BATCH_ROOT}")
    print(f"[BATCH-TEST] outputs_root={OUTPUTS_ROOT}")
    print(f"[BATCH-TEST] base_eval_root={BASE_EVAL_ROOT}")
    print(f"[BATCH-TEST] books={n_books} personas={len(personas_raw)}")
    print(f"[BATCH-TEST] experiments_per_book={n_exps} (SEQUENTIAL)")
    print(f"[BATCH-TEST] chapter_batch_size={CHAPTER_BATCH_SIZE}")

    base_url = QWEN_BASE_URLS[0] if QWEN_BASE_URLS else "http://localhost:8000/v1"

    done_books = 0
    for book_idx in range(n_books):
        _, book_title, n_ran, n_skipped = run_all_experiments_for_book(
            book_index=book_idx,
            base_url=base_url,
            books=books,
            personas_raw=personas_raw,
            experiments=experiments,
            chapter_batch_size=CHAPTER_BATCH_SIZE,
        )
        done_books += 1
        print(f"[{done_books}/{n_books}] BOOK DONE {book_title} | exps_ran={n_ran} | exps_skipped={n_skipped}")

    print(f"[BATCH-TEST] DONE. Raw outputs under: {OUTPUTS_ROOT}")
    print(f"[BATCH-TEST] Base eval cache under: {BASE_EVAL_ROOT}")


if __name__ == "__main__":
    main()
    # test()
