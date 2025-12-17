# -*- coding: utf-8 -*-
"""
runner.py（精简控制台输出 + 完整对话写日志 + 多线程并发）
----------------------------------------------------------------
控制台输出：
- 每次 LLM 请求仅输出一行摘要：book/agent/method/stage/chapter + stage进度(done/total/rem) + 耗时 + 输入/输出字符数 + parsed_keys
- 同时会输出阶段开始 STAGE START（让你明确知道当前在哪个阶段）

日志输出（runs/logs）：
- 每本书一个 JSONL：{book}_{method}_{ts}.jsonl
  每行包含一次请求的完整 system/user/assistant_raw/assistant_parsed/error 等，便于完整追踪与复盘
- 另外 logger 仍会写一个普通 .log（里面是“摘要行”，不是全文）

注意：
- PROMPT 全部英文（满足你“评测英文小说”要求）
- 多线程下讨论采用“按轮 barrier”：每轮所有 agent 看到相同的 latest N 条，再并发生成，最后按固定顺序合并，保证可复现
"""

from __future__ import annotations

import os
import json
import re
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from openai import OpenAI
import random
import traceback

# =========================
#  I/O 辅助
# =========================
def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def list_book_dirs(books_root: str) -> List[str]:
    out = []
    for name in os.listdir(books_root):
        p = os.path.join(books_root, name)
        if os.path.isdir(p):
            out.append(p)
    return sorted(out)


# =========================
#  JSON 解析（鲁棒提取）
# =========================
def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"无法从输出中提取 JSON：{text[:200]}...")
    return json.loads(m.group(0))


def clamp_score(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, x))


# =========================
#  Logger：控制台摘要 + 文件摘要
# =========================
def setup_logger(log_dir: str, also_file: bool = True) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("llm_trace")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免 PyCharm 反复运行重复加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if also_file:
        path = os.path.join(log_dir, f"run_summary_{time.strftime('%Y%m%d_%H%M%S')}.log")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def _short_uid(uid: str, n: int = 8) -> str:
    return (uid or "")[:n]


# =========================
#  ProgressTracker：显示阶段剩余任务
# =========================
class ProgressTracker:
    """按 stage 统计：total / done，用于控制台显示“该阶段还剩多少任务”"""
    def __init__(self):
        self._lock = threading.Lock()
        self._totals: Dict[str, int] = {}
        self._done: Dict[str, int] = {}

    def set_total(self, stage: str, total: int) -> None:
        with self._lock:
            self._totals[stage] = int(total)
            self._done.setdefault(stage, 0)

    def tick(self, stage: str) -> Tuple[int, int, int]:
        with self._lock:
            self._done[stage] = self._done.get(stage, 0) + 1
            total = self._totals.get(stage, 0)
            done = self._done[stage]
            rem = max(0, total - done)
            return done, total, rem

    def snapshot(self, stage: str) -> Tuple[int, int, int]:
        with self._lock:
            total = self._totals.get(stage, 0)
            done = self._done.get(stage, 0)
            rem = max(0, total - done)
            return done, total, rem


# =========================
#  JSONLTraceWriter：把每次请求完整对话写入日志文件（线程安全）
# =========================
class JSONLTraceWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._f = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._f.write(line + "\n")
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def log_llm_summary_console(
    logger: logging.Logger,
    tracker: Optional[ProgressTracker],
    trace: Dict[str, Any],
    elapsed_s: float,
    user_len: int,
    assistant_len: int,
    parsed_json: Optional[Dict[str, Any]],
    success: bool,
) -> None:
    """控制台仅输出 trace + 摘要 + 阶段进度"""
    book = trace.get("book")
    agent = _short_uid(str(trace.get("agent", "")))
    method = trace.get("method")
    stage = trace.get("stage")
    chapter = trace.get("chapter")

    done = total = rem = None
    if tracker and stage:
        d, t, r = tracker.snapshot(stage)
        done, total, rem = d, t, r

    keys = list(parsed_json.keys())[:8] if isinstance(parsed_json, dict) else []
    ok = "OK" if success else "ERR"

    logger.info(
        "[%s] book=%s | agent=%s | method=%s | stage=%s | ch=%s | "
        "stage_progress=%s/%s (rem %s) | %.2fs | in_chars=%d out_chars=%d | parsed_keys=%s",
        ok, book, agent, method, stage, str(chapter),
        str(done) if done is not None else "-", str(total) if total is not None else "-", str(rem) if rem is not None else "-",
        elapsed_s, user_len, assistant_len, keys
    )


# =========================
#  Thread-local LLM：每线程一个 client + 控制台摘要 + JSONL全文
# =========================
class ThreadLocalLLM:
    """
    每线程懒加载一个 OpenAI client（DashScope compatible-mode）：
    - 控制台打印摘要行（不打印全文）
    - 完整 system/user/assistant 写入 JSONL 文件
    - 内置网络重试：指数退避 + jitter；对 429/5xx/超时/断连等重试
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: int,
        logger: logging.Logger,
        retry_max_attempts: int = 6,
        retry_base_sleep_sec: float = 1.0,
        retry_max_sleep_sec: float = 30.0,
        retry_jitter: float = 0.2,
        retry_status_codes: Optional[List[int]] = None,
        fail_fast: bool = True,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.logger = logger
        self.local = threading.local()

        # 每本书会设置一套上下文（tracker + writer）
        self.tracker: Optional[ProgressTracker] = None
        self.writer: Optional[JSONLTraceWriter] = None

        # retry 配置
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.retry_base_sleep_sec = float(retry_base_sleep_sec)
        self.retry_max_sleep_sec = float(retry_max_sleep_sec)
        self.retry_jitter = float(retry_jitter)
        self.retry_status_codes = retry_status_codes or [408, 409, 425, 429, 500, 502, 503, 504, 522, 524]
        self.fail_fast = bool(fail_fast)

    def set_run_context(self, tracker: ProgressTracker, writer: JSONLTraceWriter) -> None:
        self.tracker = tracker
        self.writer = writer

    def _get_client(self) -> OpenAI:
        if not hasattr(self.local, "client"):
            # 显式传 http_client，规避某些环境下的 proxies 参数不兼容问题
            http = httpx.Client(timeout=httpx.Timeout(self.timeout_sec), trust_env=True)
            self.local.http = http
            self.local.client = OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http)
        return self.local.client

    @staticmethod
    def _get_status_code(err: Exception) -> Optional[int]:
        # openai SDK 通常把状态码放在 err.status_code 或 err.response.status_code
        sc = getattr(err, "status_code", None)
        if isinstance(sc, int):
            return sc
        resp = getattr(err, "response", None)
        sc2 = getattr(resp, "status_code", None)
        if isinstance(sc2, int):
            return sc2
        return None

    def _should_retry(self, err: Exception) -> bool:
        # 1) httpx 网络/超时类
        if isinstance(err, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError, httpx.ReadError, httpx.ConnectError)):
            return True

        # 2) OpenAI SDK/HTTP 状态码类（429/5xx/408 等）
        sc = self._get_status_code(err)
        if sc is not None:
            return sc in set(self.retry_status_codes)

        # 3) 兜底：通过错误消息判断一些常见的临时网络问题
        msg = (str(err) or "").lower()
        transient_keywords = [
            "timed out", "timeout", "connection reset", "connection aborted", "temporary failure",
            "remote protocol", "server disconnected", "read error", "connect error", "tls",
        ]
        return any(k in msg for k in transient_keywords)

    def _sleep_backoff(self, attempt_index: int) -> float:
        # attempt_index 从 0 开始：0,1,2...
        base = self.retry_base_sleep_sec * (2 ** attempt_index)
        base = min(base, self.retry_max_sleep_sec)
        # jitter: +- retry_jitter 比例
        if self.retry_jitter > 0:
            base *= (1.0 + random.uniform(-self.retry_jitter, self.retry_jitter))
        return max(0.0, base)

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        trace: Dict[str, Any],
        retries: int = 2,  # 保留旧参数，但实际以 self.retry_max_attempts 为主
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        返回：(assistant_raw_text, parsed_json_or_none)

        重试策略：
        - 最大尝试次数 = self.retry_max_attempts
        - 对网络/超时/429/5xx 等进行指数退避重试
        - 每次失败也会写 JSONL（含 error + system/user/raw）
        """
        assistant_raw: Optional[str] = None
        parsed: Optional[Dict[str, Any]] = None
        last_err: Optional[Exception] = None

        max_attempts = self.retry_max_attempts

        for attempt in range(max_attempts):
            t0 = time.perf_counter()
            try:
                client = self._get_client()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                assistant_raw = resp.choices[0].message.content
                dt = time.perf_counter() - t0

                try:
                    parsed = extract_json_object(assistant_raw)
                except Exception:
                    parsed = None

                # 成功：tick 进度
                stage = trace.get("stage")
                if self.tracker and stage:
                    self.tracker.tick(stage)

                # 控制台摘要行
                log_llm_summary_console(
                    logger=self.logger,
                    tracker=self.tracker,
                    trace=trace,
                    elapsed_s=dt,
                    user_len=len(user),
                    assistant_len=len(assistant_raw or ""),
                    parsed_json=parsed,
                    success=True,
                )

                # JSONL 记录完整信息
                if self.writer:
                    self.writer.write({
                        "trace": trace,
                        "elapsed_s": round(dt, 6),
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "system": system,
                        "user": user,
                        "assistant_raw": assistant_raw,
                        "assistant_parsed": parsed,
                    })

                return assistant_raw, parsed

            except Exception as e:
                dt = time.perf_counter() - t0
                last_err = e
                is_last = (attempt == max_attempts - 1)

                # 失败也要写 JSONL（完整信息 + 堆栈）
                if self.writer:
                    self.writer.write({
                        "trace": trace,
                        "elapsed_s": round(dt, 6),
                        "attempt": attempt + 1,
                        "max_attempts": max_attempts,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_status_code": self._get_status_code(e),
                        "traceback": traceback.format_exc(),
                        "system": system,
                        "user": user,
                        "assistant_raw": assistant_raw,
                        "assistant_parsed": parsed,
                    })

                # 控制台摘要行（错误）
                log_llm_summary_console(
                    logger=self.logger,
                    tracker=self.tracker,
                    trace=trace,
                    elapsed_s=dt,
                    user_len=len(user),
                    assistant_len=len(assistant_raw or ""),
                    parsed_json=parsed,
                    success=False,
                )

                # 最后一次：结束
                if is_last:
                    # 让 stage 进度别“卡在 rem>0”（即使失败也算这个任务结束）
                    stage = trace.get("stage")
                    if self.tracker and stage:
                        self.tracker.tick(stage)

                    if self.fail_fast:
                        raise RuntimeError(f"LLM 调用失败（已重试 {attempt+1}/{max_attempts} 次）：{e}") from e
                    # fail_fast=False：返回空结果，让上层用默认值继续跑
                    return "", None

                # 继续重试：退避等待
                sleep_s = self._sleep_backoff(attempt)
                time.sleep(sleep_s)

        # 理论上不会到这
        if self.fail_fast and last_err:
            raise RuntimeError(f"LLM 调用失败：{last_err}") from last_err
        return "", None



# =========================
#  PROMPT（全部英文）
# =========================
def sys_json_only() -> str:
    return (
        "You are a strict JSON generator. "
        "Return ONLY a valid JSON object and nothing else. "
        "Do not add markdown fences. Do not add commentary."
    )


def prompt_interest_filter(meta: Dict[str, Any], persona_text: Optional[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]

    user = (
        f"Book metadata:\n"
        f"- Title: {book_name}\n"
        f"- Blurb: {intro}\n\n"
    )
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Task: Decide whether this reader would be interested enough to read this novel.\n"
        "Return JSON with fields:\n"
        "{\n"
        '  "keep": boolean,\n'
        '  "interest_score": number (0-100),\n'
        '  "reason": string\n'
        "}\n"
    )
    return sys_json_only(), user


def prompt_aggregation_chapter(meta: Dict[str, Any], chapter_text: str, prev_plot_summaries: List[str], persona_text: Optional[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    prev = "\n".join([f"- {s}" for s in prev_plot_summaries]) if prev_plot_summaries else "(none)"

    user = f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Context (plot summaries of previous chapters):\n"
        f"{prev}\n\n"
        "Current chapter (English novel text):\n"
        f"{chapter_text}\n\n"
        "Task:\n"
        "1) Write a short plot summary of THIS chapter (2-4 sentences).\n"
        "2) Give a score for THIS chapter on a 1.0 to 5.0 scale (allow 1 decimal).\n"
        "3) Provide a brief comment (2-4 sentences) in English.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "plot_summary": string,\n'
        '  "score": number (1.0-5.0),\n'
        '  "comment": string\n'
        "}\n"
    )
    return sys_json_only(), user


def prompt_incremental_update(meta: Dict[str, Any], chapter_text: str, prev_summary: str, prev_review: str, prev_score: float, persona_text: Optional[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    user = f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Previous running summary (so far):\n"
        f"{prev_summary if prev_summary else '(empty)'}\n\n"
        "Previous review (so far):\n"
        f"{prev_review if prev_review else '(empty)'}\n\n"
        f"Previous overall score (1.0-5.0): {prev_score:.1f}\n\n"
        "Current chapter (English novel text):\n"
        f"{chapter_text}\n\n"
        "Task:\n"
        "- Update the running summary (keep it concise, <= 180 words).\n"
        "- Update the review (<= 180 words).\n"
        "- Update the overall score on a 1.0-5.0 scale (allow 1 decimal).\n\n"
        "Return JSON:\n"
        "{\n"
        '  "summary": string,\n'
        '  "review": string,\n'
        '  "score": number (1.0-5.0)\n'
        "}\n"
    )
    return sys_json_only(), user


def prompt_summary_incremental(meta: Dict[str, Any], chapter_text: str, global_summary: Dict[str, Any], persona_text: Optional[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    user = f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Current global summary JSON (may be empty):\n"
        f"{global_summary}\n\n"
        "New chapter (English novel text):\n"
        f"{chapter_text}\n\n"
        "Task: Update the global summary. Keep it compact and consistent.\n"
        "- plot: <= 220 words total\n"
        "- characters: <= 220 words total\n"
        "- style_excerpts: keep up to 3 short excerpts, each <= 30 words\n\n"
        "Return JSON:\n"
        "{\n"
        '  "plot": string,\n'
        '  "characters": string,\n'
        '  "style_excerpts": [string, string, ...]\n'
        "}\n"
    )
    return sys_json_only(), user


def prompt_summary_final(meta: Dict[str, Any], global_summary: Dict[str, Any], persona_text: Optional[str], discussion_tail: List[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
        f"Global summary JSON:\n{global_summary}\n\n"
        "Optional group discussion (latest messages):\n"
        f"{disc}\n\n"
        "Task:\n"
        "- Write a critique in English (<= 220 words).\n"
        "- Give an overall score from 1.0 to 5.0 (allow 1 decimal).\n\n"
        "Return JSON:\n"
        "{\n"
        '  "critique": string,\n'
        '  "score": number (1.0-5.0)\n'
        "}\n"
    )
    if persona_text:
        user += "\nNote: The critique should be consistent with the persona's tastes."
    return sys_json_only(), user


def prompt_discussion_message(meta: Dict[str, Any], persona_text: Optional[str], agent_stance: str, discussion_tail: List[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
        f"Your current stance:\n{agent_stance}\n\n"
        "Latest group discussion messages:\n"
        f"{disc}\n\n"
        "Task: Write ONE short message to the group (1-3 sentences), in English.\n"
        "Return JSON:\n"
        "{\n"
        '  "message": string\n'
        "}\n"
    )
    if persona_text:
        user += "\nStay in-character with the persona."
    return sys_json_only(), user


def prompt_finalize_after_discussion(meta: Dict[str, Any], persona_text: Optional[str], pre_score: float, discussion_tail: List[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
        f"Your pre-discussion overall score (1.0-5.0): {pre_score:.1f}\n\n"
        "Latest group discussion messages:\n"
        f"{disc}\n\n"
        "Task:\n"
        "- Decide your final overall score (1.0-5.0, allow 1 decimal).\n"
        "- Provide a final short review (<= 180 words) in English.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "final_score": number (1.0-5.0),\n'
        '  "final_review": string\n'
        "}\n"
    )
    if persona_text:
        user += "\nStay consistent with the persona."
    return sys_json_only(), user


# =========================
#  评测方法（合并在 runner.py 内）
# =========================
def run_aggregation_agent(
    llm: ThreadLocalLLM,
    cfg: Any,
    meta: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    agent_uuid: str,
    persona_text: Optional[str],
    score_decimals: int,
) -> Tuple[List[Dict[str, Any]], float, str]:
    prev_summaries: List[str] = []
    evals: List[Dict[str, Any]] = []
    scores: List[float] = []
    last_comment = ""

    for ch in chapters:
        chapter_idx = int(ch["Number"])
        chapter_text = str(ch["text"])

        system, user = prompt_aggregation_chapter(meta, chapter_text, prev_summaries, persona_text)
        _, parsed = llm.chat_json(
            system=system,
            user=user,
            temperature=float(cfg.llm.temperature),
            top_p=float(cfg.llm.top_p),
            max_tokens=int(cfg.llm.max_tokens),
            trace={"book": meta["book_name"], "agent": agent_uuid, "method": "aggregation", "stage": "aggregation_chapter", "chapter": chapter_idx},
        )
        obj = parsed or {}

        plot_summary = str(obj.get("plot_summary", "")).strip()
        comment = str(obj.get("comment", "")).strip()
        last_comment = comment

        try:
            score = float(obj.get("score", 3.0))
        except Exception:
            score = 3.0
        score = clamp_score(score)
        score = round(score, score_decimals)

        prev_summaries.append(plot_summary)
        scores.append(score)
        evals.append({"chapter_index": chapter_idx, "score": score, "plot_summary": plot_summary, "comment": comment})

    pre_score = round(sum(scores) / max(1, len(scores)), score_decimals)
    stance = f"My current impression is around {pre_score:.1f}. Latest thought: {last_comment}" if last_comment else f"My current impression is around {pre_score:.1f}."
    return evals, pre_score, stance


def run_incremental_agent(
    llm: ThreadLocalLLM,
    cfg: Any,
    meta: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    agent_uuid: str,
    persona_text: Optional[str],
    score_decimals: int,
) -> Tuple[List[Dict[str, Any]], float, str, Dict[str, Any]]:
    """
    Incremental-updated：
    - 每章更新 summary/review/score
    - pre_score：所有 step score 的平均（全文平均分）
    - last_state：最后一章后的 summary/review/score（用于无讨论时 final_review）
    """

    prev_summary = ""
    prev_review = ""
    prev_score = 3.0

    steps: List[Dict[str, Any]] = []
    scores: List[float] = []

    for ch in chapters:
        chapter_idx = int(ch["Number"])
        chapter_text = str(ch["text"])

        system, user = prompt_incremental_update(
            meta, chapter_text, prev_summary, prev_review, float(prev_score), persona_text
        )
        _, parsed = llm.chat_json(
            system=system,
            user=user,
            temperature=float(cfg.llm.temperature),
            top_p=float(cfg.llm.top_p),
            max_tokens=int(cfg.llm.max_tokens),
            trace={
                "book": meta["book_name"],
                "agent": agent_uuid,
                "method": "incremental",
                "stage": "incremental_update",
                "chapter": chapter_idx,
            },
        )
        obj = parsed or {}

        # ✅ 关键：字段缺失/空字符串时，保留旧状态，不要把 prev_* 清空
        new_summary = obj.get("summary", None)
        if new_summary is not None and str(new_summary).strip():
            prev_summary = str(new_summary).strip()

        new_review = obj.get("review", None)
        if new_review is not None and str(new_review).strip():
            prev_review = str(new_review).strip()

        try:
            score = float(obj.get("score", prev_score))
        except Exception:
            score = prev_score
        score = round(clamp_score(score), score_decimals)
        prev_score = score

        scores.append(score)
        steps.append(
            {
                "chapter_index": chapter_idx,
                "score": score,
                "summary": prev_summary,
                "review": prev_review,
            }
        )

    pre_score = round(sum(scores) / max(1, len(scores)), score_decimals)
    stance = (
        f"My current impression is around {pre_score:.1f}. Latest review: {prev_review}"
        if prev_review else f"My current impression is around {pre_score:.1f}."
    )

    last_state = {"summary": prev_summary, "review": prev_review, "last_score": prev_score}
    return steps, pre_score, stance, last_state



def build_summary_agent(
    llm: ThreadLocalLLM,
    cfg: Any,
    meta: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    agent_uuid: str,
    persona_text: Optional[str],
) -> Dict[str, Any]:
    global_summary: Dict[str, Any] = {"plot": "", "characters": "", "style_excerpts": []}

    for ch in chapters:
        chapter_idx = int(ch["Number"])
        chapter_text = str(ch["text"])

        system, user = prompt_summary_incremental(meta, chapter_text, global_summary, persona_text)
        _, parsed = llm.chat_json(
            system=system,
            user=user,
            temperature=float(cfg.llm.temperature),
            top_p=float(cfg.llm.top_p),
            max_tokens=int(cfg.llm.max_tokens),
            trace={"book": meta["book_name"], "agent": agent_uuid, "method": "summary_based", "stage": "summary_incremental", "chapter": chapter_idx},
        )
        obj = parsed or {}
        global_summary = {
            "plot": str(obj.get("plot", "")).strip(),
            "characters": str(obj.get("characters", "")).strip(),
            "style_excerpts": obj.get("style_excerpts", []) or [],
        }

    return global_summary


# =========================
#  cfg / env 工具
# =========================
def _get_api_key_from_env(cfg: Any) -> str:
    env_name = getattr(cfg.llm, "api_key_env", "DASHSCOPE_API_KEY")
    key = os.getenv(env_name, "")
    if not key:
        raise RuntimeError(f"环境变量 {env_name} 未设置。请在 .env 或系统环境变量中设置它。")
    return key


def _persona_text(persona_raw: Dict[str, Any]) -> str:
    return str(persona_raw.get("persona", "")).strip()


def _get_int(cfg: Any, path: str, default: int) -> int:
    """
    从 cfg 中安全取整型值，例如 path="experiment.n_agents"
    """
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    try:
        return int(cur)
    except Exception:
        return default


def _get_bool(cfg: Any, path: str, default: bool) -> bool:
    cur = cfg
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
    return bool(cur)


def _safe_cfg_dump(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        try:
            return cfg.model_dump()
        except Exception:
            pass
    # 退化：只取关键字段
    out: Dict[str, Any] = {}
    try:
        out["llm"] = {
            "base_url": getattr(cfg.llm, "base_url", None),
            "api_key_env": getattr(cfg.llm, "api_key_env", None),
            "model": getattr(cfg.llm, "model", None),
            "temperature": getattr(cfg.llm, "temperature", None),
            "top_p": getattr(cfg.llm, "top_p", None),
            "max_tokens": getattr(cfg.llm, "max_tokens", None),
            "timeout_sec": getattr(cfg.llm, "timeout_sec", None),
        }
        out["paths"] = {
            "personas_json": getattr(cfg.paths, "personas_json", None),
            "books_root": getattr(cfg.paths, "books_root", None),
            "output_root": getattr(cfg.paths, "output_root", None),
        }
        out["experiment"] = {
            "method": getattr(cfg.experiment, "method", None),
            "use_persona": getattr(cfg.experiment, "use_persona", None),
            "use_discussion": getattr(cfg.experiment, "use_discussion", None),
            "use_interest_filter": getattr(cfg.experiment, "use_interest_filter", None),
            "discussion_rounds": getattr(cfg.experiment, "discussion_rounds", None),
            "discussion_window": getattr(cfg.experiment, "discussion_window", None),
            "n_agents": getattr(cfg.experiment, "n_agents", None),
            "score_decimals": getattr(cfg.experiment, "score_decimals", None),
            "discussion_affects_score": getattr(cfg.experiment, "discussion_affects_score", None),
        }
    except Exception:
        pass
    return out


# =========================
#  主流程：run_all_books / run_one_book
# =========================
def run_all_books(cfg: Any, _unused_llm: Any = None) -> None:
    """
    兼容 main.py 原来的 run_all_books(cfg, llm) 调用：
    runner 内部自己创建 thread-local llm（确保多线程安全 + 全量 JSONL 日志）
    """
    output_root = str(cfg.paths.output_root)
    os.makedirs(output_root, exist_ok=True)

    log_dir = os.path.join(output_root, "logs")
    logger = setup_logger(log_dir, also_file=True)

    api_key = _get_api_key_from_env(cfg)
    llm = ThreadLocalLLM(
        base_url=str(cfg.llm.base_url),
        api_key=api_key,
        model=str(cfg.llm.model),
        timeout_sec=int(cfg.llm.timeout_sec),
        logger=logger,
    )

    personas_raw = read_json(str(cfg.paths.personas_json))
    n_agents = _get_int(cfg, "experiment.n_agents", 8)
    personas_raw = personas_raw[:n_agents]

    books_root = str(cfg.paths.books_root)
    book_dirs = list_book_dirs(books_root)

    logger.info("=== RUN START === books=%d | n_agents=%d | method=%s", len(book_dirs), n_agents, str(cfg.experiment.method))

    for book_dir in book_dirs:
        meta = read_json(os.path.join(book_dir, "book_metadata.json"))
        chapters = read_json(os.path.join(book_dir, "chapters.json"))
        run_one_book(cfg, llm, logger, meta, chapters, personas_raw)

    logger.info("=== RUN END ===")


def run_one_book(
    cfg: Any,
    llm: ThreadLocalLLM,
    logger: logging.Logger,
    meta: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    personas_raw: List[Dict[str, Any]],
) -> None:
    method = str(cfg.experiment.method)
    use_persona = _get_bool(cfg, "experiment.use_persona", True)
    use_discussion = _get_bool(cfg, "experiment.use_discussion", False)
    use_interest_filter = _get_bool(cfg, "experiment.use_interest_filter", True)
    discussion_affects_score = _get_bool(cfg, "experiment.discussion_affects_score", True)

    # 并发：默认 workers=min(n_agents, 8)，也可在 config 里加 concurrency.max_workers
    n_agents = len(personas_raw)
    max_workers = _get_int(cfg, "concurrency.max_workers", min(8, n_agents)) if hasattr(cfg, "concurrency") else min(8, n_agents)
    max_workers = max(1, min(max_workers, 32))  # 再给一个硬上限，避免误配

    rounds = _get_int(cfg, "experiment.discussion_rounds", 2)
    window = _get_int(cfg, "experiment.discussion_window", 10)
    score_decimals = _get_int(cfg, "experiment.score_decimals", 1)

    book_name = str(meta.get("book_name", "UNKNOWN"))
    logger.info(">>> BOOK START: %s | method=%s | agents=%d | workers=%d", book_name, method, n_agents, max_workers)

    # 每本书一个 tracker + JSONL（完整对话日志）
    tracker = ProgressTracker()
    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(str(cfg.paths.output_root), "logs", f"{book_name}_{method}_{ts}.jsonl")
    writer = JSONLTraceWriter(jsonl_path)
    llm.set_run_context(tracker, writer)
    logger.info("FULL TRACE(JSONL): %s", jsonl_path)

    # -------------------------
    # STAGE 1: 兴趣筛选（并发）
    # -------------------------
    decisions: Dict[str, Dict[str, Any]] = {}
    if use_interest_filter:
        tracker.set_total("interest_filter", n_agents)
        logger.info("STAGE START: interest_filter | total_tasks=%d", n_agents)

        def _interest_worker(persona_raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            uid = str(persona_raw.get("uuid"))
            persona_text = _persona_text(persona_raw) if use_persona else None
            system, user = prompt_interest_filter(meta, persona_text)
            _, parsed = llm.chat_json(
                system=system,
                user=user,
                temperature=float(cfg.llm.temperature),
                top_p=float(cfg.llm.top_p),
                max_tokens=int(cfg.llm.max_tokens),
                trace={"book": book_name, "agent": uid, "method": method, "stage": "interest_filter", "chapter": None},
            )
            obj = parsed or {}
            obj.setdefault("keep", True)
            obj.setdefault("interest_score", 50.0)
            obj.setdefault("reason", "")
            return uid, obj

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_interest_worker, p) for p in personas_raw]
            for fu in as_completed(futures):
                uid, obj = fu.result()
                decisions[uid] = obj
    else:
        for p in personas_raw:
            uid = str(p.get("uuid"))
            decisions[uid] = {"keep": True, "interest_score": 50.0, "reason": "interest_filter_disabled"}

    kept_personas = [p for p in personas_raw if bool(decisions.get(str(p.get("uuid")), {}).get("keep", True))]
    filtered_pre = len(personas_raw) - len(kept_personas)

    logger.info("STAGE DONE: interest_filter | kept=%d/%d | filtered_pre_read=%d", len(kept_personas), n_agents, filtered_pre)

    # -------------------------
    # STAGE 2: 基线评测（并发，且每章会产生多次 LLM 调用 → tracker 按总调用数计）
    # -------------------------
    base_payload: Dict[str, Any] = {}
    pre_scores: Dict[str, float] = {}
    stances: Dict[str, str] = {}

    n_kept = len(kept_personas)
    n_chapters = len(chapters)

    if n_kept > 0:
        if method == "aggregation":
            tracker.set_total("aggregation_chapter", n_kept * n_chapters)
            logger.info("STAGE START: aggregation_chapter | total_tasks=%d", n_kept * n_chapters)
        elif method == "incremental":
            tracker.set_total("incremental_update", n_kept * n_chapters)
            logger.info("STAGE START: incremental_update | total_tasks=%d", n_kept * n_chapters)
        elif method == "summary_based":
            tracker.set_total("summary_incremental", n_kept * n_chapters)
            logger.info("STAGE START: summary_incremental | total_tasks=%d", n_kept * n_chapters)
        else:
            writer.close()
            raise ValueError(f"Unknown method: {method}")

        def _eval_worker(persona_raw: Dict[str, Any]) -> Tuple[str, str, Any, float, str]:
            uid = str(persona_raw.get("uuid"))
            persona_text = _persona_text(persona_raw) if use_persona else None

            if method == "aggregation":
                evals, pre_score, stance = run_aggregation_agent(
                    llm=llm, cfg=cfg, meta=meta, chapters=chapters,
                    agent_uuid=uid, persona_text=persona_text, score_decimals=score_decimals
                )
                return uid, "aggregation", evals, float(pre_score), stance

            if method == "incremental":
                steps, pre_score, stance, last_state = run_incremental_agent(
                    llm=llm, cfg=cfg, meta=meta, chapters=chapters,
                    agent_uuid=uid, persona_text=persona_text,
                    score_decimals=score_decimals
                )
                return uid, "incremental", {"steps": steps, "last_state": last_state}, float(pre_score), stance

            if method == "summary_based":
                gs = build_summary_agent(
                    llm=llm, cfg=cfg, meta=meta, chapters=chapters,
                    agent_uuid=uid, persona_text=persona_text
                )
                # pre_score 先占位，最终评分在 STAGE 4 的 summary_final
                stance = "I have formed a coherent view based on the full summary of the novel."
                return uid, "summary_based", gs, 3.0, stance

            raise ValueError(method)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_eval_worker, p) for p in kept_personas]
            for fu in as_completed(futures):
                uid, kind, payload, pre_score, stance = fu.result()
                base_payload[uid] = {"kind": kind, "payload": payload}
                pre_scores[uid] = round(clamp_score(float(pre_score)), score_decimals)
                stances[uid] = stance

        logger.info("STAGE DONE: base_evaluation | kept_agents=%d", n_kept)

    # -------------------------
    # STAGE 3: 讨论（按轮 barrier，每轮一个 stage，保证“最新 n 条”语义一致）
    # -------------------------
    global_messages: List[str] = []
    per_agent_discussion: Dict[str, List[Dict[str, Any]]] = {str(p.get("uuid")): [] for p in kept_personas}

    if use_discussion and n_kept > 0:
        for r in range(1, rounds + 1):
            stage_name = f"discussion_round_{r}"
            tracker.set_total(stage_name, n_kept)
            logger.info("STAGE START: %s | total_tasks=%d | visible_window=%d", stage_name, n_kept, window)

            tail = global_messages[-window:]  # 本轮所有 agent 看到同一份 tail（barrier）

            def _disc_worker(persona_raw: Dict[str, Any]) -> Tuple[str, str]:
                uid = str(persona_raw.get("uuid"))
                persona_text = _persona_text(persona_raw) if use_persona else None
                stance = stances.get(uid, "My impression is forming.")
                system, user = prompt_discussion_message(meta, persona_text, stance, tail)
                _, parsed = llm.chat_json(
                    system=system,
                    user=user,
                    temperature=float(cfg.llm.temperature),
                    top_p=float(cfg.llm.top_p),
                    max_tokens=int(cfg.llm.max_tokens),
                    trace={"book": book_name, "agent": uid, "method": method, "stage": stage_name, "chapter": None},
                )
                obj = parsed or {}
                msg = str(obj.get("message", "")).strip() or "(no message)"
                return uid, msg

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_disc_worker, p) for p in kept_personas]
                round_msgs = [fu.result() for fu in as_completed(futures)]

            # 固定顺序合并，保证可复现
            for uid, msg in sorted(round_msgs, key=lambda x: x[0]):
                global_messages.append(msg)
                per_agent_discussion[uid].append({"round": r, "message": msg})

            logger.info("STAGE DONE: %s | total_messages=%d", stage_name, len(global_messages))

    # -------------------------
    # STAGE 4: 最终评分（并发）
    # - summary_based：无论是否讨论，都用 summary_final（输入 global_summary + 最新讨论n条）
    # - 其他方法：若讨论开启，用 finalize_after_discussion；否则不额外调用 LLM（直接用 pre_scores）
    # -------------------------
    post_scores: Dict[str, float] = {}
    final_reviews: Dict[str, str] = {}

    disc_tail = global_messages[-window:]

    if n_kept > 0:
        if method == "summary_based":
            tracker.set_total("summary_final", n_kept)
            logger.info("STAGE START: summary_final | total_tasks=%d | visible_window=%d", n_kept, window)

            def _summary_final_worker(persona_raw: Dict[str, Any]) -> Tuple[str, float, str]:
                uid = str(persona_raw.get("uuid"))
                persona_text = _persona_text(persona_raw) if use_persona else None
                gs = base_payload[uid]["payload"]

                system, user = prompt_summary_final(meta, gs, persona_text, discussion_tail=disc_tail if use_discussion else [])
                _, parsed = llm.chat_json(
                    system=system,
                    user=user,
                    temperature=float(cfg.llm.temperature),
                    top_p=float(cfg.llm.top_p),
                    max_tokens=int(cfg.llm.max_tokens),
                    trace={"book": book_name, "agent": uid, "method": method, "stage": "summary_final", "chapter": None},
                )
                obj = parsed or {}
                critique = str(obj.get("critique", "")).strip()
                try:
                    score = float(obj.get("score", 3.0))
                except Exception:
                    score = 3.0
                score = round(clamp_score(score), score_decimals)
                return uid, score, critique

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_summary_final_worker, p) for p in kept_personas]
                for fu in as_completed(futures):
                    uid, sc, critique = fu.result()
                    pre_scores[uid] = sc
                    final_reviews[uid] = critique

            logger.info("STAGE DONE: summary_final")

        else:
            if use_discussion:
                tracker.set_total("finalize_after_discussion", n_kept)
                logger.info("STAGE START: finalize_after_discussion | total_tasks=%d | visible_window=%d", n_kept, window)

                def _finalize_worker(persona_raw: Dict[str, Any]) -> Tuple[str, float, str]:
                    uid = str(persona_raw.get("uuid"))
                    persona_text = _persona_text(persona_raw) if use_persona else None
                    pre = float(pre_scores.get(uid, 3.0))

                    system, user = prompt_finalize_after_discussion(meta, persona_text, pre, disc_tail)
                    _, parsed = llm.chat_json(
                        system=system,
                        user=user,
                        temperature=float(cfg.llm.temperature),
                        top_p=float(cfg.llm.top_p),
                        max_tokens=int(cfg.llm.max_tokens),
                        trace={"book": book_name, "agent": uid, "method": method, "stage": "finalize_after_discussion", "chapter": None},
                    )
                    obj = parsed or {}
                    try:
                        fs = float(obj.get("final_score", pre))
                    except Exception:
                        fs = pre
                    fs = round(clamp_score(fs), score_decimals)
                    review = str(obj.get("final_review", "")).strip()
                    return uid, fs, review

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [ex.submit(_finalize_worker, p) for p in kept_personas]
                    for fu in as_completed(futures):
                        uid, fs, review = fu.result()
                        post_scores[uid] = fs
                        final_reviews[uid] = review

                logger.info("STAGE DONE: finalize_after_discussion")

    # -------------------------
    # 组装输出
    # -------------------------
    agents_out: List[Dict[str, Any]] = []
    for p in personas_raw:
        uid = str(p.get("uuid"))
        keep = bool(decisions.get(uid, {}).get("keep", True))

        agent_obj: Dict[str, Any] = {
            "agent_uuid": uid,
            "kept": keep,
            "filtered_reason": None if keep else str(decisions.get(uid, {}).get("reason", "not_interested")),
            "interest": {
                "keep": keep,
                "interest_score": float(decisions.get(uid, {}).get("interest_score", 50.0)),
                "reason": str(decisions.get(uid, {}).get("reason", "")),
            } if use_interest_filter else None,
            "discussion": per_agent_discussion.get(uid, []) if (use_discussion and keep) else [],
            "pre_discussion_score": None,
            "post_discussion_score": None,
            "final_review": None,
            "chapter_evals": None,
            "incremental_steps": None,
            "global_summary": None,
        }

        if not keep:
            agents_out.append(agent_obj)
            continue

        kind = base_payload[uid]["kind"]
        payload = base_payload[uid]["payload"]
        if kind == "aggregation":
            agent_obj["chapter_evals"] = payload
        elif kind == "incremental":
            agent_obj["incremental_steps"] = payload["steps"]
            if (not use_discussion) and (method == "incremental"):
                agent_obj["final_review"] = str(payload["last_state"].get("review", "")).strip()
        elif kind == "summary_based":
            agent_obj["global_summary"] = payload

        agent_obj["pre_discussion_score"] = float(pre_scores.get(uid, None)) if pre_scores.get(uid, None) is not None else None

        if use_discussion and method != "summary_based":
            agent_obj["post_discussion_score"] = float(post_scores.get(uid, None)) if post_scores.get(uid, None) is not None else None
            agent_obj["final_review"] = final_reviews.get(uid, "")
        else:
            # summary_based：final_reviews 是 critique；aggregation/incremental 无讨论时 final_review 可能为空
            agent_obj["final_review"] = final_reviews.get(uid, "")

        agents_out.append(agent_obj)

    kept_scores: List[float] = []
    for a in agents_out:
        if not a["kept"]:
            continue
        if use_discussion and discussion_affects_score and method != "summary_based":
            if a["post_discussion_score"] is not None:
                kept_scores.append(float(a["post_discussion_score"]))
        else:
            if a["pre_discussion_score"] is not None:
                kept_scores.append(float(a["pre_discussion_score"]))

    book_score = round(sum(kept_scores) / max(1, len(kept_scores)), score_decimals) if kept_scores else None

    out = {
        "book_name": book_name,
        "metadata": {"book_name": meta.get("book_name", ""), "intro": meta.get("intro", "")},
        "config": _safe_cfg_dump(cfg),
        "agents": agents_out,
        "aggregate": {
            "book_score": book_score,
            "n_agents_total": len(personas_raw),
            "n_kept": len(kept_personas),
            "filtered_pre_read": filtered_pre,
        },
    }

    out_path = os.path.join(str(cfg.paths.output_root), f"{book_name}_{method}_{ts}.json")
    write_json(out_path, out)

    # 关闭 JSONL writer（防止文件句柄悬挂）
    writer.close()

    logger.info("<<< BOOK END: %s | book_score=%s | out=%s", book_name, str(book_score), out_path)
