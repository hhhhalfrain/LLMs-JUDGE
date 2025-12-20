# -*- coding: utf-8 -*-
"""
batch_run_hardcoded.py
------------------------------------------------------------
无 CLI、硬编码配置的批量运行器：
- 读取合并后的 books 列表 JSON（你新格式：[{metadata:{...}, chapter:[...]} , ...]）
- 读取 personas JSON
- 构造参数网格，并发跑“不同实验配置”
- 每本书直接调用 runner.evaluate_single_book(...)（传参，不走目录扫描）
- 不做任何汇总，只保留原始输出（json + logs/jsonl）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from src.runner import setup_logger, ThreadLocalLLM, evaluate_single_book


# ============================================================
# ✅ 硬编码区：你只需要改这里
# ============================================================
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOOKS_MERGED_JSON = PROJECT_ROOT / "data" / "books" / "merged_books.json"   # 你的新输入（list）
PERSONAS_JSON = PROJECT_ROOT / "data" / "personas_sample.json"

BATCH_ROOT = PROJECT_ROOT / "runs" / "batch"
MAX_EXPERIMENT_WORKERS = 3     # 同时跑多少个“实验配置”
START_STAGGER_SEC = 1.0        # 实验启动错峰（减少 429 / 网络抖动）

# LLM 配置（千问 32B，DashScope compatible）
LLM_BASE_URL = "https://poloai.top/v1"
LLM_MODEL = "gpt-4o-mini"
API_KEY_ENV = "QWEN_API_KEY"

LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 3000
LLM_TIMEOUT_SEC = 120

# 重试参数（网络不稳定时很关键）
RETRY_MAX_ATTEMPTS = 6
RETRY_BASE_SLEEP_SEC = 1.0
RETRY_MAX_SLEEP_SEC = 30.0
RETRY_JITTER = 0.2
FAIL_FAST = True

# runner 内部：同一本书里 agent 并发（每本书评测时用）
PER_BOOK_AGENT_WORKERS = 6


# -----------------------------
# ✅ 实验网格维度（全部可扫）
# -----------------------------
METHODS = ["aggregation", "incremental", "summary_based"]
USE_PERSONA_OPTS = [True,]
USE_DISCUSSION_OPTS = [True]
USE_INTEREST_FILTER_OPTS = [True]

# 你要求的 5 个也进入网格：
DISCUSSION_ROUNDS_OPTS = [2]            # 例如 [1, 2, 3]
DISCUSSION_WINDOW_OPTS = [10]           # 例如 [5, 10, 20]
N_AGENTS_OPTS = [8]                     # 例如 [8, 16, 32]
SCORE_DECIMALS_OPTS = [1]               # 例如 [0, 1, 2]
DISCUSSION_AFFECTS_SCORE_OPTS = [True]  # 例如 [True, False]


# ============================================================
# 工具函数
# ============================================================
def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "_")
    return s.strip()[:180]


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
    )


def build_cfg(
    output_root: Path,
    method: str,
    use_persona: bool,
    use_discussion: bool,
    use_interest: bool,
    discussion_rounds: int,
    discussion_window: int,
    n_agents: int,
    score_decimals: int,
    discussion_affects_score: bool,
) -> Any:
    """
    runner.py 用的是：
      cfg.llm.xxx
      cfg.experiment.xxx
      cfg.paths.output_root
      cfg.concurrency.max_workers
    这里用 SimpleNamespace 构造一个“属性访问型配置”，不依赖 yaml / CLI。
    """
    return SimpleNamespace(
        llm=SimpleNamespace(
            base_url=LLM_BASE_URL,
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
        ),
        concurrency=SimpleNamespace(
            max_workers=PER_BOOK_AGENT_WORKERS
        ),
        paths=SimpleNamespace(
            output_root=str(output_root)
        )
    )


def build_experiments_grid() -> List[Tuple]:
    exps: List[Tuple] = []
    for m in METHODS:
        for up in USE_PERSONA_OPTS:
            for ud in USE_DISCUSSION_OPTS:
                for ui in USE_INTEREST_FILTER_OPTS:
                    for rr in DISCUSSION_ROUNDS_OPTS:
                        for ww in DISCUSSION_WINDOW_OPTS:
                            for na in N_AGENTS_OPTS:
                                for sd in SCORE_DECIMALS_OPTS:
                                    for das in DISCUSSION_AFFECTS_SCORE_OPTS:
                                        exps.append((m, up, ud, ui, rr, ww, na, sd, das))
    return exps


# ============================================================
# 单个实验：跑所有书（顺序），原始输出落在自己的 output_root
# ============================================================
def run_one_experiment(
    exp_index: int,
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
    batch_root: Path,
) -> str:
    run_id = make_run_id(
        method, use_persona, use_discussion, use_interest,
        discussion_rounds, discussion_window, n_agents,
        score_decimals, discussion_affects_score
    )

    out_root = batch_root / "outputs" / run_id
    safe_mkdir(out_root)

    # 每个实验一个 logs 目录 + 独立 logger 名（避免并发互相抢 handler）
    log_dir = out_root / "logs"
    logger = setup_logger(str(log_dir), also_file=True, logger_name=f"llm_trace_{run_id}")

    # 错峰启动，减少同时请求造成的抖动/429
    if START_STAGGER_SEC > 0:
        time.sleep(exp_index * START_STAGGER_SEC)

    cfg = build_cfg(
        out_root, method, use_persona, use_discussion, use_interest,
        discussion_rounds, discussion_window, n_agents, score_decimals,
        discussion_affects_score
    )

    api_key = os.getenv(cfg.llm.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"环境变量 {cfg.llm.api_key_env} 未设置。请先设置 DashScope API Key。")

    # 注意：ThreadLocalLLM 的 __init__ 需要与你 runner.py 的实现保持一致
    llm = ThreadLocalLLM(
        base_url=str(cfg.llm.base_url),
        api_key=api_key,
        model=str(cfg.llm.model),
        timeout_sec=int(cfg.llm.timeout_sec),
        logger=logger,
        retry_max_attempts=int(cfg.llm.retry_max_attempts),
        retry_base_sleep_sec=float(cfg.llm.retry_base_sleep_sec),
        retry_max_sleep_sec=float(cfg.llm.retry_max_sleep_sec),
        retry_jitter=float(cfg.llm.retry_jitter),
        fail_fast=bool(cfg.llm.fail_fast),
    )

    # 限制 persona 数量（由网格控制）
    personas_used = personas_raw[: int(cfg.experiment.n_agents)]

    logger.info(
        "=== EXP START === run_id=%s | books=%d | n_agents=%d | rounds=%d | window=%d | score_decimals=%d | das=%s",
        run_id, len(books), len(personas_used), discussion_rounds, discussion_window, score_decimals, str(discussion_affects_score)
    )

    for i, book_record in enumerate(books[:1], start=1): # TODO 临时只跑前 3 本测试，记得删
        logger.info("=== BOOK %d/%d ===", i, len(books))
        evaluate_single_book(cfg, llm, logger, book_record, personas_used)

    logger.info("=== EXP END === run_id=%s", run_id)
    return run_id


# ============================================================
# main（无 CLI）
# ============================================================
def main() -> None:
    ts = now_tag()
    batch_root = BATCH_ROOT / ts
    safe_mkdir(batch_root)

    books = read_json(BOOKS_MERGED_JSON)
    personas_raw = read_json(PERSONAS_JSON)

    experiments = build_experiments_grid()

    print(f"[BATCH] batch_root={batch_root}")
    print(f"[BATCH] books={len(books)} personas={len(personas_raw)}")
    print(f"[BATCH] experiments={len(experiments)} max_workers={MAX_EXPERIMENT_WORKERS}")

    with ThreadPoolExecutor(max_workers=max(1, MAX_EXPERIMENT_WORKERS)) as ex:
        futs = []
        for idx, (m, up, ud, ui, rr, ww, na, sd, das) in enumerate(experiments):
            futs.append(ex.submit(
                run_one_experiment,
                idx, books, personas_raw,
                m, up, ud, ui,
                rr, ww, na, sd, das,
                batch_root
            ))

        done = 0
        for fu in as_completed(futs):
            run_id = fu.result()
            done += 1
            print(f"[{done}/{len(futs)}] OK  run_id={run_id}")

    print(f"[BATCH] DONE. Raw outputs under: {batch_root / 'outputs'}")


if __name__ == "__main__":
    main()
