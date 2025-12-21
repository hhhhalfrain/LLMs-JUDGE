# -*- coding: utf-8 -*-
# runner.py

from __future__ import annotations

import os
import json
import re
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .trace_logger import setup_logger

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


def _short_uid(uid: str, n: int = 8) -> str:
    return (uid or "")[:n]


def _truncate(s: str, n: int = 260) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[:n] + " ...")


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
    error_msg: Optional[str] = None,
    error_type: Optional[str] = None,
    error_status_code: Optional[int] = None,
) -> None:
    """控制台仅输出 trace + 摘要 + 阶段进度；若 ERR 则追加错误信息"""
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

    err_part = ""
    if not success:
        pieces = []
        if error_status_code is not None:
            pieces.append(f"status={error_status_code}")
        if error_type:
            pieces.append(str(error_type))
        if error_msg:
            pieces.append(_truncate(error_msg))
        if pieces:
            err_part = " | err=" + " | ".join(pieces)

    logger.info(
        "[%s] book=%s | agent=%s | method=%s | stage=%s | ch=%s | "
        "stage_progress=%s/%s (rem %s) | %.2fs | in_chars=%d out_chars=%d | parsed_keys=%s%s",
        ok,
        book,
        agent,
        method,
        stage,
        str(chapter),
        str(done) if done is not None else "-",
        str(total) if total is not None else "-",
        str(rem) if rem is not None else "-",
        elapsed_s,
        user_len,
        assistant_len,
        keys,
        err_part,
    )


from collections import deque
import threading
from typing import Tuple


class AdaptiveConcurrencyController:
    """
    滑动窗口自适应并发控制器（最近 N 次）：

    维护最近 N=window_size 次调用的 (success, elapsed_s)。
    - 若 avg_latency > high_latency_s 或 error_rate >= error_rate_high：
        workers = workers // 2  (不低于 min_workers)
    - 若 avg_latency < low_latency_s 且窗口内 0 错误：
        workers += 1  (不超过 max_workers)

    注意：
    - 这里的并发是“建议值”，你当前代码是在每个 stage 开始前调用 suggest_workers()，
      所以并发调整会在下一次创建 ThreadPoolExecutor 时生效（不会在 stage 内即时缩容）。
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        window_size: int = 10,          # ✅ N = 10
        high_latency_s: float = 100.0,  # ✅ avg > 100 -> /2
        low_latency_s: float = 60.0,    # ✅ avg < 60 & no error -> +1
        error_rate_high: float = 0.1,   # ✅ err_rate >= 0.1 -> /2
    ):
        self.min_workers = max(1, int(min_workers))
        self.max_workers = max(self.min_workers, int(max_workers))

        self.window_size = max(1, int(window_size))
        self.high_latency_s = float(high_latency_s)
        self.low_latency_s = float(low_latency_s)
        self.error_rate_high = float(error_rate_high)

        self._lock = threading.Lock()
        self._recent = deque(maxlen=self.window_size)  # items: (success: bool, elapsed_s: float)

        # “粘性”并发：不是每次都从 default 开始算，而是基于上一次 workers 调整
        self._current_workers = None  # type: int | None

    def reset(self) -> None:
        """清空窗口，并重置 current_workers。"""
        with self._lock:
            self._recent.clear()
            self._current_workers = None

    def register_call(self, success: bool, elapsed_s: float) -> None:
        """记录一次调用结果（线程安全）。"""
        with self._lock:
            self._recent.append((bool(success), float(elapsed_s)))

    def window_stats(self) -> Tuple[int, float, float, int]:
        """
        返回 (n, avg_latency, error_rate, error_count) 基于最近窗口。
        """
        with self._lock:
            n = len(self._recent)
            if n == 0:
                return 0, 0.0, 0.0, 0
            total_lat = 0.0
            err = 0
            for ok, dt in self._recent:
                total_lat += float(dt)
                if not ok:
                    err += 1
            avg = total_lat / n
            err_rate = err / n
            return n, avg, err_rate, err

    def suggest_workers(self, default_workers: int) -> int:
        """
        给出建议并发数（线程安全）。
        - default_workers：你期望的默认并发（会被 min/max 裁剪）
        - 返回：根据最近 N 次窗口规则更新后的 workers
        """
        default_workers = int(default_workers)
        if default_workers <= 0:
            default_workers = self.min_workers

        with self._lock:
            # 初始化 current_workers：第一次用 default（并裁剪）
            if self._current_workers is None:
                self._current_workers = max(self.min_workers, min(default_workers, self.max_workers))
            else:
                # 允许外部 default 变化（例如 agent 数变少），把 current 也裁剪一下
                self._current_workers = max(self.min_workers, min(self._current_workers, self.max_workers))

            n = len(self._recent)
            if n == 0:
                return self._current_workers

            total_lat = 0.0
            err = 0
            for ok, dt in self._recent:
                total_lat += float(dt)
                if not ok:
                    err += 1

            avg_lat = total_lat / n
            err_rate = err / n

            # ✅ 规则 1：avg > 100 或 err_rate >= 0.1 -> 并发 / 2
            if (avg_lat > self.high_latency_s) or (err_rate >= self.error_rate_high):
                halved = self._current_workers // 2
                if halved <= 0:
                    halved = 1
                self._current_workers = max(self.min_workers, halved)

            # ✅ 规则 2：avg < 60 且 0 错误 -> 并发 + 1
            elif (avg_lat < self.low_latency_s) and (err == 0):
                self._current_workers = min(self.max_workers, self._current_workers + 1)

            # 最终再裁剪一次
            self._current_workers = max(self.min_workers, min(self._current_workers, self.max_workers))
            return self._current_workers



# =========================
#  Thread-local LLM：每线程一个 client + 控制台摘要 + JSONL全文 + 重试 + 自适应并发统计
# =========================
class ThreadLocalLLM:
    """
    每线程懒加载一个 OpenAI client（DashScope compatible-mode）：
    - 控制台打印摘要行（不打印全文）
    - 完整 system/user/assistant 写入 JSONL 文件
    - 内置网络重试：指数退避 + jitter；对 429/5xx/超时/断连等重试
    - 可选挂载 AdaptiveConcurrencyController，用于阶段内动态调节并发
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
        monitor: Optional[AdaptiveConcurrencyController] = None,
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

        # 自适应并发监控
        self.monitor = monitor

    def set_run_context(self, tracker: ProgressTracker, writer: JSONLTraceWriter) -> None:
        self.tracker = tracker
        self.writer = writer

    def _get_client(self) -> OpenAI:
        if not hasattr(self.local, "client"):
            http = httpx.Client(timeout=httpx.Timeout(self.timeout_sec), trust_env=True)
            self.local.http = http
            self.local.client = OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http)
        return self.local.client

    @staticmethod
    def _get_status_code(err: Exception) -> Optional[int]:
        sc = getattr(err, "status_code", None)
        if isinstance(sc, int):
            return sc
        resp = getattr(err, "response", None)
        sc2 = getattr(resp, "status_code", None)
        if isinstance(sc2, int):
            return sc2
        return None

    def _should_retry(self, err: Exception) -> bool:
        if isinstance(
            err,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.TransportError,
                httpx.ReadError,
                httpx.ConnectError,
            ),
        ):
            return True

        sc = self._get_status_code(err)
        if sc is not None:
            return sc in set(self.retry_status_codes)

        msg = (str(err) or "").lower()
        transient_keywords = [
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "temporary failure",
            "remote protocol",
            "server disconnected",
            "read error",
            "connect error",
            "tls",
        ]
        return any(k in msg for k in transient_keywords)

    def _sleep_backoff(self, attempt_index: int) -> float:
        base = self.retry_base_sleep_sec * (2**attempt_index)
        base = min(base, self.retry_max_sleep_sec)
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

                # 自适应监控：记录一次成功调用
                if self.monitor is not None:
                    self.monitor.register_call(True, dt)

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
                    self.writer.write(
                        {
                            "trace": trace,
                            "elapsed_s": round(dt, 6),
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "system": system,
                            "user": user,
                            "assistant_raw": assistant_raw,
                            "assistant_parsed": parsed,
                        }
                    )

                return assistant_raw, parsed

            except Exception as e:
                dt = time.perf_counter() - t0
                last_err = e
                is_last = attempt == max_attempts - 1

                # 自适应监控：记录一次失败调用
                if self.monitor is not None:
                    self.monitor.register_call(False, dt)

                # 失败也要写 JSONL（完整信息 + 堆栈）
                if self.writer:
                    self.writer.write(
                        {
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
                        }
                    )

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
                    error_msg=str(e),
                    error_type=type(e).__name__,
                    error_status_code=self._get_status_code(e),
                )

                if is_last:
                    stage = trace.get("stage")
                    if self.tracker and stage:
                        self.tracker.tick(stage)

                    if self.fail_fast:
                        raise RuntimeError(f"LLM 调用失败（已重试 {attempt + 1}/{max_attempts} 次）：{e}") from e
                    return "", None

                sleep_s = self._sleep_backoff(attempt)
                time.sleep(sleep_s)

        if self.fail_fast and last_err:
            raise RuntimeError(f"LLM 调用失败：{last_err}") from last_err
        return "", None

    # 为了兼容旧代码，还挂一个 logger 属性
    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        self._logger = value


# =========================
#  PROMPT（全部英文）
# =========================
def sys_json_only() -> str:
    return (
        "You are a strict JSON generator. "
        "Return ONLY a valid JSON object and nothing else. "
        "No markdown. No explanations. No extra text."
    )


def _persona_block(persona_text: Optional[str]) -> str:
    if not persona_text:
        return ""
    return "".join(
        [
            "READER PERSONA (this is YOU — stay in-character):\n",
            f"{persona_text}\n\n",
            "Persona grounding (MANDATORY):\n",
            "- Treat the persona as your identity + taste filter.\n",
            "- Speak with this persona's voice (not generic/academic).\n\n",
        ]
    )


def _discussion_block(discussion_tail: List[str]) -> str:
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"
    return "".join(
        [
            "GROUP CHAT LOG (latest messages):\n",
            f"{disc}\n\n",
            "Discussion rules (MANDATORY):\n",
            "- Read the chat log and react to specific points (paraphrase).\n",
            "- Keep the chat clean: respectful, no personal attacks, no slurs, no spam.\n\n",
        ]
    )


def prompt_interest_filter(meta: Dict[str, Any], persona_text: Optional[str]) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()

    parts = [
        "Book metadata:\n",
        f"- Title: {book_name}\n",
        (f"- Author: {author}\n" if author else ""),
        f"- Blurb: {intro}\n\n",
        _persona_block(persona_text),
        "Task: Would YOU (as this persona) start reading this novel?\n",
        "Return JSON:\n",
        "{\n",
        '  "keep": boolean,\n',
        '  "interest_score": number (0-100),\n',
        '  "reason": string\n',
        "}\n",
        "Note: The reason should clearly connect to persona preferences.\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_aggregation_chapter(
    meta: Dict[str, Any],
    chapter_text: str,
    prev_plot_summaries: List[str],
    persona_text: Optional[str],
) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()
    prev = "\n".join([f"- {s}" for s in prev_plot_summaries]) if prev_plot_summaries else "(none)"

    parts = [
        "Novel metadata:\n",
        f"Title: {book_name}\n",
        (f"Author: {author}\n" if author else ""),
        f"Blurb: {intro}\n\n",
        _persona_block(persona_text),
        "Context (previous plot summaries):\n",
        f"{prev}\n\n",
        "Current chapter (English novel text):\n",
        f"{chapter_text}\n\n",
        "Task:\n",
        "- Plot summary of THIS chapter (2-4 sentences).\n",
        "- Score THIS chapter (1.0-5.0, 1 decimal).\n",
        "- Short comment in THIS persona's voice (2-4 sentences).\n\n",
        "Return JSON:\n",
        "{\n",
        '  "plot_summary": string,\n',
        '  "score": number (1.0-5.0),\n',
        '  "comment": string\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_incremental_update(
    meta: Dict[str, Any],
    chapter_text: str,
    prev_summary: str,
    prev_review: str,
    prev_score: float,
    persona_text: Optional[str],
) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()

    parts = [
        "Novel metadata:\n",
        f"Title: {book_name}\n",
        (f"Author: {author}\n" if author else ""),
        f"Blurb: {intro}\n\n",
        _persona_block(persona_text),
        "You update your opinion chapter-by-chapter.\n\n",
        "Previous running summary:\n",
        f"{prev_summary if prev_summary else '(empty)'}\n\n",
        "Previous review:\n",
        f"{prev_review if prev_review else '(empty)'}\n\n",
        f"Previous overall score (1.0-5.0): {prev_score:.1f}\n\n",
        "Current chapter text:\n",
        f"{chapter_text}\n\n",
        "Task:\n",
        "- Update summary (<= 180 words).\n",
        "- Update review (<= 180 words) in persona voice.\n",
        "- Update score (1.0-5.0, 1 decimal).\n\n",
        "Return JSON:\n",
        "{\n",
        '  "summary": string,\n',
        '  "review": string,\n',
        '  "score": number (1.0-5.0)\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_summary_incremental(
    meta: Dict[str, Any],
    chapter_text: str,
    global_summary: Dict[str, Any],
    persona_text: Optional[str],
) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()

    parts = [
        "Novel metadata:\n",
        f"Title: {book_name}\n",
        (f"Author: {author}\n" if author else ""),
        f"Blurb: {intro}\n\n",
        _persona_block(persona_text),
        "Current global summary JSON:\n",
        f"{global_summary}\n\n",
        "New chapter text:\n",
        f"{chapter_text}\n\n",
        "Task: Update global summary, compact and consistent.\n\n",
        "Return JSON:\n",
        "{\n",
        '  "plot": string,\n',
        '  "characters": string,\n',
        '  "style_excerpts": [string, string, ...]\n',
        "}\n",
        "Persona note: choose style_excerpts this persona would quote.\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_summary_final(
    meta: Dict[str, Any],
    global_summary: Dict[str, Any],
    persona_text: Optional[str],
    username: str,
    current_score: float,
    discussion_tail: List[str],
) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()

    parts = [
        "Novel metadata:\n",
        f"Title: {book_name}\n",
        (f"Author: {author}\n" if author else ""),
        f"Blurb: {intro}\n\n",
        _persona_block(persona_text),
        f"Your chat username: {username}\n",
        f"Your current score (after discussion rounds): {current_score:.1f}\n\n",
        "Global summary JSON:\n",
        f"{global_summary}\n\n",
        _discussion_block(discussion_tail),
        "Task:\n",
        "- Write a critique in English (<= 220 words) in THIS persona's voice.\n",
        "- Give an overall score (1.0-5.0, 1 decimal).\n\n",
        "Return JSON:\n",
        "{\n",
        '  "critique": string,\n',
        '  "score": number (1.0-5.0)\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_discussion_message(
    meta: Dict[str, Any],
    persona_text: Optional[str],
    username: str,
    agent_stance: str,
    current_score: float,
    discussion_tail: List[str],
) -> Tuple[str, str]:
    """
    讨论轮输出（对模型强约束，但不做代码强制校验）：
    - 本轮建议更新评分（尽量不要保持不变）
    - 建议单轮调整幅度在 [-0.3, +0.3]（允许 +0.1 / -0.2 等）
    - 若讨论明显一边倒：尽可能提出一个合理的不同意见/反例/担忧点（强烈建议，不强制）
    - JSON 不再输出 delta，只输出当前 score
    """
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"

    parts = [
        "Novel metadata:\n",
        f"Title: {book_name}\n",
        (f"Author: {author}\n" if author else ""),
        f"Blurb: {intro}\n\n",
        _persona_block(persona_text),
        f"Your chat username: {username}\n",
        f"Your current score (before this round): {current_score:.1f}\n\n",
        "Your current stance (your view so far):\n",
        f"{agent_stance}\n\n",
        "Latest group chat messages:\n",
        f"{disc}\n\n",
        _counterpoint_block(),
        "Scoring update guidance:\n",
        "- You SHOULD update your score this round (try not to keep it unchanged).\n",
        "- Recommended per-round change range: from -1.0 to +1.0.\n",
        "- Keep score within [1.0, 5.0].\n\n",
        "Task:\n",
        "- Write ONE short chat message (1-3 sentences), clean and respectful.\n",
        f"- The message MUST start with your name like: [{username}] ...\n",
        "- Mention your updated score in the message.\n",
        "- React to at least one specific point from the chat log.\n",
        "- If the chat is one-sided, try to add one thoughtful counterpoint.\n\n",
        "Return JSON:\n",
        "{\n",
        '  "message": string,\n',
        '  "score": number\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)



def prompt_finalize_after_discussion(
    meta: Dict[str, Any],
    persona_text: Optional[str],
    username: str,
    pre_score: float,
    discussion_tail: List[str],
) -> Tuple[str, str]:
    book_name = meta["book_name"]
    intro = meta["intro"]
    author = str(meta.get("author", "")).strip()

    parts = [
        "Novel metadata:\n",
        f"Title: {book_name}\n",
        (f"Author: {author}\n" if author else ""),
        f"Blurb: {intro}\n\n",
        _persona_block(persona_text),
        f"Your chat username: {username}\n",
        f"Your current score (after discussion rounds): {pre_score:.1f}\n\n",
        _discussion_block(discussion_tail),
        "Task:\n",
        "- Decide your final overall score (1.0-5.0, 1 decimal).\n",
        "- Write a final short review (<= 180 words) in English, persona voice, referencing discussion points.\n\n",
        "Return JSON:\n",
        "{\n",
        '  "final_score": number (1.0-5.0),\n',
        '  "final_review": string\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)



# =========================
#  评测方法（单 agent 基线评测）
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
            trace={
                "book": meta["book_name"],
                "agent": agent_uuid,
                "method": "aggregation",
                "stage": "aggregation_chapter",
                "chapter": chapter_idx,
            },
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
        evals.append(
            {
                "chapter_index": chapter_idx,
                "score": score,
                "plot_summary": plot_summary,
                "comment": comment,
            }
        )

    pre_score = round(sum(scores) / max(1, len(scores)), score_decimals)
    stance = (
        f"My current impression is around {pre_score:.1f}. Latest thought: {last_comment}"
        if last_comment
        else f"My current impression is around {pre_score:.1f}."
    )
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

        system, user = prompt_incremental_update(meta, chapter_text, prev_summary, prev_review, float(prev_score), persona_text)
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
        if prev_review
        else f"My current impression is around {pre_score:.1f}."
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
            trace={
                "book": meta["book_name"],
                "agent": agent_uuid,
                "method": "summary_based",
                "stage": "summary_incremental",
                "chapter": chapter_idx,
            },
        )
        obj = parsed or {}
        global_summary = {
            "plot": str(obj.get("plot", "")).strip(),
            "characters": str(obj.get("characters", "")).strip(),
            "style_excerpts": obj.get("style_excerpts", []) or [],
        }

    return global_summary

def _counterpoint_block() -> str:
    return "".join(
        [
            "Counterpoint rule (STRONGLY RECOMMENDED):\n",
            "- If the group chat is one-sided (everyone agrees in the same direction, no real pushback),\n",
            "  try to contribute at least ONE thoughtful counterpoint or alternative interpretation.\n",
            "- Do NOT be contrarian just to be contrarian. Keep it plausible, text-grounded, and respectful.\n",
            "- You may disagree partially (e.g., agree on pacing but question character motivation).\n\n",
        ]
    )



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
            "personas_json": getattr(getattr(cfg, "paths", object()), "personas_json", None),
            "books_root": getattr(getattr(cfg, "paths", object()), "books_root", None),
            "output_root": getattr(getattr(cfg, "paths", object()), "output_root", None),
            "base_eval_root": getattr(getattr(cfg, "paths", object()), "base_eval_root", None),
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
            "chapter_batch_size": getattr(cfg.experiment, "chapter_batch_size", None),
        }
    except Exception:
        pass
    return out


# =========================
#  字符串工具：文件名清洗
# =========================
def sanitize_name(s: str, max_len: int = 180) -> str:
    bad = '<>:"/\\|?*'
    s = (s or "").strip()
    for ch in bad:
        s = s.replace(ch, "_")
    s = s.strip().strip(".")
    return s[:max_len] if len(s) > max_len else s


# =========================
#  书本结构适配
# =========================
def _normalize_book_record(book_record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    把你现在的输入格式：
    {
      "chapter": [{"id":1,"title":"Chapter 1","text":"..."}, ...],
      "metadata": {"title": "...", "author":"...", "intro":"...", ...}
    }
    适配成 runner 内部使用的：
    meta: {"book_name": ..., "intro": ..., "author": ...}
    chapters: [{"Number": 1, "title": "...", "text": "..."}, ...]
    """
    md = book_record.get("metadata", {}) or {}
    chs = book_record.get("chapter", []) or []

    title = str(md.get("title", "UNKNOWN")).strip()
    intro = str(md.get("intro", "")).strip()
    author = str(md.get("author", "")).strip()

    meta = {
        "book_name": title,
        "intro": intro,
        "author": author,
    }

    chapters: List[Dict[str, Any]] = []
    for c in sorted(chs, key=lambda x: int(x.get("id", x.get("Number", 0)) or 0)):
        num = c.get("Number", None)
        if num is None or int(num) == 0:
            num = c.get("id", 0)
        chapters.append(
            {
                "Number": int(num or 0),
                "title": str(c.get("title", "")).strip(),
                "text": str(c.get("text", "")),
            }
        )

    return meta, chapters


# =========================
#  BaseEval 缓存辅助
# =========================
def _get_base_eval_root(cfg: Any) -> Optional[str]:
    paths = getattr(cfg, "paths", None)
    if paths is None:
        return None
    root = getattr(paths, "base_eval_root", None)
    return str(root) if root else None


def _base_eval_cache_path(cfg: Any, method: str, book_name: str, persona_key: str) -> Optional[str]:
    root = _get_base_eval_root(cfg)
    if not root:
        return None
    fname = sanitize_name(f"{book_name}__persona={persona_key}.json")
    return os.path.join(root, method, fname)


def _load_base_eval_from_disk(cfg: Any, method: str, book_name: str, persona_key: str) -> Optional[Dict[str, Any]]:
    path = _base_eval_cache_path(cfg, method, book_name, persona_key)
    if not path or not os.path.exists(path):
        return None
    try:
        obj = read_json(path)
        be = obj.get("base_eval")
        if isinstance(be, dict):
            return be
    except Exception:
        pass
    return None


def _save_base_eval_to_disk(
    cfg: Any,
    method: str,
    book_name: str,
    persona_key: str,
    agent_uuid: str,
    base_eval: Dict[str, Any],
) -> None:
    path = _base_eval_cache_path(cfg, method, book_name, persona_key)
    if not path:
        return
    obj = {
        "book_name": book_name,
        "method": method,
        "persona_key": persona_key,
        "agent_uuid_sample": agent_uuid,
        "config": _safe_cfg_dump(cfg),
        "base_eval": base_eval,
    }
    write_json(path, obj)


def _ensure_base_eval_for_persona(
    llm: ThreadLocalLLM,
    cfg: Any,
    meta: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    method: str,
    persona_text: Optional[str],
    persona_key: str,
    agent_uuid: str,
    score_decimals: int,
) -> Dict[str, Any]:
    """
    确保某本书 / 某方法 / 某 persona_key 的基线评测存在：
    - 若缓存存在：直接加载
    - 若不存在：调用对应 run_* 函数评测，然后写入缓存
    返回结构：
    {
      "kind": "aggregation" | "incremental" | "summary_based",
      "payload": ...,
      "pre_discussion_score": float,
      "stance": str
    }
    """
    book_name = str(meta.get("book_name", "UNKNOWN"))
    cached = _load_base_eval_from_disk(cfg, method, book_name, persona_key)
    if isinstance(cached, dict) and "kind" in cached and "payload" in cached:
        return cached

    # 需要真正跑一遍
    if method == "aggregation":
        evals, pre_score, stance = run_aggregation_agent(
            llm=llm,
            cfg=cfg,
            meta=meta,
            chapters=chapters,
            agent_uuid=agent_uuid,
            persona_text=persona_text,
            score_decimals=score_decimals,
        )
        base_eval = {
            "kind": "aggregation",
            "payload": evals,
            "pre_discussion_score": float(pre_score),
            "stance": stance,
        }
    elif method == "incremental":
        steps, pre_score, stance, last_state = run_incremental_agent(
            llm=llm,
            cfg=cfg,
            meta=meta,
            chapters=chapters,
            agent_uuid=agent_uuid,
            persona_text=persona_text,
            score_decimals=score_decimals,
        )
        base_eval = {
            "kind": "incremental",
            "payload": {"steps": steps, "last_state": last_state},
            "pre_discussion_score": float(pre_score),
            "stance": stance,
        }
    elif method == "summary_based":
        gs = build_summary_agent(
            llm=llm,
            cfg=cfg,
            meta=meta,
            chapters=chapters,
            agent_uuid=agent_uuid,
            persona_text=persona_text,
        )
        stance = "I have formed a coherent view based on the full summary of the novel."
        base_eval = {
            "kind": "summary_based",
            "payload": gs,
            "pre_discussion_score": 3.0,
            "stance": stance,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    _save_base_eval_to_disk(cfg, method, book_name, persona_key, agent_uuid, base_eval)
    return base_eval


# =========================
#  外部入口：评测“单本书”
# =========================
def evaluate_single_book(
    cfg: Any,
    llm: ThreadLocalLLM,
    logger: logging.Logger,
    book_record: Dict[str, Any],
    personas_raw: List[Dict[str, Any]],
) -> None:
    meta, chapters = _normalize_book_record(book_record)
    run_one_book(cfg, llm, logger, meta, chapters, personas_raw)


# =========================
#  主流程：单本书
# =========================
def run_one_book(
    cfg: Any,
    llm: ThreadLocalLLM,
    logger: logging.Logger,
    meta: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    personas_raw: List[Dict[str, Any]],
) -> None:
    """
    新版流程（你的需求）：
    - STEP A（全局基线 / 原 STAGE2）：对所有 persona 或 nopersona 做基线评测（aggregation / incremental / summary_based）
      并写入 base_eval_root 缓存。
    - STEP B（兴趣筛选 / 原 STAGE1）：根据 metadata + persona 做 interest_filter，决定哪些 persona 进入讨论。
    - STEP C（讨论 / 原 STAGE3）：只对 kept personas 进行多轮讨论。
    - STEP D（讨论后最终打分 / 原 STAGE4）：summary_based 使用 summary_final，其余方法可选使用 finalize_after_discussion。
    - 输出：只写原始数据（per-agent 基线 + interest + 讨论 + final_score），不再在这里计算 book_level 的 aggregate score。
    """

    # >>> 你的修改要求：max_workers > 4 强制固定为 4
    HARD_MAX_WORKERS = 4

    method = str(cfg.experiment.method)
    use_persona = _get_bool(cfg, "experiment.use_persona", True)
    use_discussion = _get_bool(cfg, "experiment.use_discussion", False)
    use_interest_filter = _get_bool(cfg, "experiment.use_interest_filter", True)

    n_agents = len(personas_raw)
    max_workers_cfg = _get_int(cfg, "concurrency.max_workers", min(8, n_agents))
    if max_workers_cfg > HARD_MAX_WORKERS:
        max_workers_cfg = HARD_MAX_WORKERS

    rounds = _get_int(cfg, "experiment.discussion_rounds", 2)
    window = _get_int(cfg, "experiment.discussion_window", 10)
    score_decimals = _get_int(cfg, "experiment.score_decimals", 1)

    book_name = str(meta.get("book_name", "UNKNOWN"))
    logger.info(
        ">>> BOOK START: %s | method=%s | agents=%d | max_workers_cfg=%d (hard_cap=%d)",
        book_name,
        method,
        n_agents,
        max_workers_cfg,
        HARD_MAX_WORKERS,
    )

    # 每本书一个 tracker + JSONL（完整对话日志）
    # 每本书一个 tracker + JSONL（完整对话日志）
    tracker = ProgressTracker()
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # ✅ 固定命名：避免 Windows 路径过长 / 特殊字符
    jsonl_path = os.path.join(str(cfg.paths.output_root), "logs", "trace.jsonl")
    writer = JSONLTraceWriter(jsonl_path)
    llm.set_run_context(tracker, writer)
    logger.info("FULL TRACE(JSONL): %s", jsonl_path)

    try:
        # -----------------------------------------------------
        # 下面保持你原来的 STEP A/B/C/D 全部逻辑不变
        # -----------------------------------------------------

        # -----------------------------------------------------
        # STEP A: 基线评测（对应原 STAGE2，但对所有 persona / nopersona，一次性跑完）
        # -----------------------------------------------------
        base_payload: Dict[str, Dict[str, Any]] = {}
        pre_scores: Dict[str, float] = {}
        stances: Dict[str, str] = {}

        if use_persona:
            personas_for_base = personas_raw
        else:
            # ✅ persona 关闭：不借用任何真实 persona 作为“保底面具”
            # 只跑一次“空 persona”基线，然后复制给所有 agent
            personas_for_base = [{"uuid": "nopersona"}] if personas_raw else []

        n_eval_agents = len(personas_for_base)
        n_chapters = len(chapters)

        if n_eval_agents > 0 and n_chapters > 0:
            monitor = getattr(llm, "monitor", None)
            default_workers = max(1, min(max_workers_cfg, n_eval_agents))
            if isinstance(monitor, AdaptiveConcurrencyController):
                workers = monitor.suggest_workers(default_workers)
            else:
                workers = default_workers
            workers = max(1, min(workers, n_eval_agents, HARD_MAX_WORKERS))

            logger.info(
                "STAGE START: base_eval | method=%s | eval_agents=%d | chapters=%d | workers=%d",
                method,
                n_eval_agents,
                n_chapters,
                workers,
            )

            def persona_cache_key(p: Dict[str, Any]) -> str:
                if use_persona:
                    return str(p.get("uuid", "unknown"))
                return "nopersona"

            # 仅统计“需要实际调用 LLM 的 persona 数量”用于 tracker
            to_eval: List[Dict[str, Any]] = []
            base_root = _get_base_eval_root(cfg)
            if base_root:
                for p in personas_for_base:
                    pk = persona_cache_key(p)
                    path = _base_eval_cache_path(cfg, method, book_name, pk)
                    if not (path and os.path.exists(path)):
                        to_eval.append(p)
            else:
                to_eval = list(personas_for_base)

            # 用真实 trace stage 名来设 total，保证 snapshot/tick 能命中
            method_stage_map = {
                "aggregation": "aggregation_chapter",
                "incremental": "incremental_update",
                "summary_based": "summary_incremental",
            }
            base_call_stage = method_stage_map.get(method, "base_eval")

            if to_eval:
                tracker.set_total(base_call_stage, len(to_eval) * n_chapters)

            def _base_worker(persona_raw: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
                uid = str(persona_raw.get("uuid"))
                pk = persona_cache_key(persona_raw)
                persona_txt = _persona_text(persona_raw) if use_persona else None
                rec = _ensure_base_eval_for_persona(
                    llm=llm,
                    cfg=cfg,
                    meta=meta,
                    chapters=chapters,
                    method=method,
                    persona_text=persona_txt,
                    persona_key=pk,
                    agent_uuid=uid,
                    score_decimals=score_decimals,
                )
                return uid, pk, rec

            base_result_by_key: Dict[str, Dict[str, Any]] = {}

            if n_eval_agents == 1:
                uid, pk, rec = _base_worker(personas_for_base[0])
                base_result_by_key[pk] = rec
                base_payload[uid] = {"kind": rec["kind"], "payload": rec["payload"]}
                pre_scores[uid] = float(rec["pre_discussion_score"])
                stances[uid] = rec["stance"]
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_base_worker, p) for p in personas_for_base]
                    for fu in as_completed(futures):
                        uid, pk, rec = fu.result()
                        base_result_by_key[pk] = rec
                        base_payload[uid] = {"kind": rec["kind"], "payload": rec["payload"]}
                        pre_scores[uid] = float(rec["pre_discussion_score"])
                        stances[uid] = rec["stance"]

            logger.info("STAGE DONE: base_eval | method=%s", method)

            # persona 关闭时：把 nopersona 的结果复制到所有 agent 上（你说的“乘以智能体数”）
            if not use_persona and base_result_by_key:
                shared = base_result_by_key.get("nopersona")
                if shared:
                    for p in personas_raw:
                        uid = str(p.get("uuid"))
                        base_payload[uid] = {"kind": shared["kind"], "payload": shared["payload"]}
                        pre_scores[uid] = float(shared["pre_discussion_score"])
                        stances[uid] = shared["stance"]

        # -----------------------------------------------------
        # STEP B: 兴趣筛选（对应原 STAGE1）
        # -----------------------------------------------------
        decisions: Dict[str, Dict[str, Any]] = {}
        if use_interest_filter:
            tracker.set_total("interest_filter", n_agents)
            logger.info("STAGE START: interest_filter | total_tasks=%d", n_agents)

            monitor = getattr(llm, "monitor", None)
            default_workers = max(1, min(max_workers_cfg, n_agents))
            if isinstance(monitor, AdaptiveConcurrencyController):
                workers = monitor.suggest_workers(default_workers)
            else:
                workers = default_workers
            workers = max(1, min(workers, n_agents, HARD_MAX_WORKERS))

            def _interest_worker(persona_raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
                uid = str(persona_raw.get("uuid"))
                persona_txt = _persona_text(persona_raw) if use_persona else None
                system, user = prompt_interest_filter(meta, persona_txt)
                _, parsed = llm.chat_json(
                    system=system,
                    user=user,
                    temperature=float(cfg.llm.temperature),
                    top_p=float(cfg.llm.top_p),
                    max_tokens=int(cfg.llm.max_tokens),
                    trace={
                        "book": book_name,
                        "agent": uid,
                        "method": method,
                        "stage": "interest_filter",
                        "chapter": None,
                    },
                )
                obj = parsed or {}
                obj.setdefault("keep", True)
                obj.setdefault("interest_score", 50.0)
                obj.setdefault("reason", "")
                return uid, obj

            with ThreadPoolExecutor(max_workers=workers) as ex:
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
        logger.info(
            "STAGE DONE: interest_filter | kept=%d/%d | filtered_pre_read=%d",
            len(kept_personas),
            n_agents,
            filtered_pre,
        )

        # -----------------------------------------------------
        # STEP C: 讨论（原 STAGE3）
        # 目标：
        # - 干净聊天室（尊重、简洁、带用户名）
        # - 每轮讨论都要改分（delta!=0），幅度建议[-0.3, +0.3]（允许0.1/-0.2等）
        # - 不做严格校验/纠偏：尽量相信模型给的 delta/new_score（只做分数范围clamp）
        # -----------------------------------------------------
        global_messages: List[str] = []
        per_agent_discussion: Dict[str, List[Dict[str, Any]]] = {str(p.get("uuid")): [] for p in kept_personas}

        n_kept = len(kept_personas)

        disc_decimals = max(1, int(score_decimals))  # 讨论分数建议至少1位小数
        discussion_scores: Dict[str, float] = {
            str(p.get("uuid")): round(float(pre_scores.get(str(p.get("uuid")), 3.0)), disc_decimals)
            for p in kept_personas
        }

        def _username_of(persona_raw: Dict[str, Any], uid: str) -> str:
            u = str(persona_raw.get("username", "") or "").strip()
            return u if u else f"user_{_short_uid(uid)}"

        def _clean_one_line(s: str, n: int = 240) -> str:
            s = str(s or "").replace("\r", " ").replace("\n", " ").strip()
            s = re.sub(r"\s+", " ", s)
            return s if len(s) <= n else (s[:n] + " ...")

        def _strip_leading_bracket_name(msg: str) -> str:
            m = re.match(r"^\s*\[[^\]]+\]\s*[:\-]?\s*(.*)$", msg.strip())
            return (m.group(1).strip() if m else msg.strip())

        if use_discussion and n_kept > 0:
            for r in range(1, rounds + 1):
                stage_name = f"discussion_round_{r}"
                tracker.set_total(stage_name, n_kept)

                monitor = getattr(llm, "monitor", None)
                default_workers = max(1, min(max_workers_cfg, n_kept))
                if isinstance(monitor, AdaptiveConcurrencyController):
                    workers = monitor.suggest_workers(default_workers)
                else:
                    workers = default_workers
                workers = max(1, min(workers, n_kept, HARD_MAX_WORKERS))

                logger.info(
                    "STAGE START: %s | total_tasks=%d | visible_window=%d | workers=%d",
                    stage_name,
                    n_kept,
                    window,
                    workers,
                )

                tail = global_messages[-window:]

                def _disc_worker(persona_raw: Dict[str, Any]) -> Tuple[str, str, float, float, str, str]:
                    uid = str(persona_raw.get("uuid"))
                    username = _username_of(persona_raw, uid)
                    persona_txt = _persona_text(persona_raw) if use_persona else None

                    prev_sc = round(float(discussion_scores.get(uid, pre_scores.get(uid, 3.0))), disc_decimals)
                    stance = stances.get(uid, f"My current score is {prev_sc:.1f}.")

                    system, user = prompt_discussion_message(
                        meta=meta,
                        persona_text=persona_txt,
                        username=username,
                        agent_stance=stance,
                        current_score=prev_sc,
                        discussion_tail=tail,
                    )
                    _, parsed = llm.chat_json(
                        system=system,
                        user=user,
                        temperature=float(cfg.llm.temperature),
                        top_p=float(cfg.llm.top_p),
                        max_tokens=int(cfg.llm.max_tokens),
                        trace={
                            "book": book_name,
                            "agent": uid,
                            "method": method,
                            "stage": stage_name,
                            "chapter": None,
                        },
                    )
                    obj = parsed or {}

                    raw_msg = str(obj.get("message", "")).strip() or "(no message)"
                    body = _clean_one_line(_strip_leading_bracket_name(raw_msg))

                    # 新协议：优先读 obj["score"]；兼容旧协议（new_score / delta）
                    score_val: Optional[float] = None
                    try:
                        if obj.get("score", None) is not None:
                            score_val = float(obj.get("score"))
                    except Exception:
                        score_val = None

                    # 兼容旧字段（如果模型还在吐 new_score）
                    if score_val is None:
                        try:
                            if obj.get("new_score", None) is not None:
                                score_val = float(obj.get("new_score"))
                        except Exception:
                            score_val = None

                    # 最后兜底兼容：如果只给了 delta（旧格式），允许用它更新
                    if score_val is None:
                        try:
                            if obj.get("delta", None) is not None:
                                score_val = float(prev_sc) + float(obj.get("delta"))
                        except Exception:
                            score_val = None

                    if score_val is not None:
                        new_sc = round(clamp_score(score_val), disc_decimals)
                    else:
                        new_sc = prev_sc  # 不做强制纠偏/强制改分

                    delta_used = round(new_sc - prev_sc, disc_decimals)  # 仅用于内部记录/分析（不再喂给模型）

                    # 聊天行：不再展示 Δ，避免“跟风调分”信号
                    chat_line = f"[{username}] (score {new_sc:.1f}): {body}"
                    chat_line = _clean_one_line(chat_line, n=320)

                    return uid, username, float(delta_used), float(new_sc), chat_line, body

                round_rows: List[Tuple[str, str, float, float, str, str]] = []
                if workers == 1:
                    for persona_raw in kept_personas:
                        round_rows.append(_disc_worker(persona_raw))
                else:
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futures = [ex.submit(_disc_worker, p) for p in kept_personas]
                        for fu in as_completed(futures):
                            round_rows.append(fu.result())

                # 固定顺序合并（按 uid），避免并发导致顺序漂移
                for uid, username, delta_used, new_sc, chat_line, body in sorted(round_rows, key=lambda x: x[0]):
                    global_messages.append(chat_line)
                    per_agent_discussion[uid].append(
                        {
                            "round": r,
                            "username": username,
                            "score": float(new_sc),
                            "message": body,
                        }
                    )
                    discussion_scores[uid] = float(new_sc)
                    stances[uid] = f"My current score is {new_sc:.1f}. {body}"

                logger.info("STAGE DONE: %s | total_messages=%d", stage_name, len(global_messages))

        # -----------------------------------------------------
        # STEP D: 最终打分（原 STAGE4）
        # baseline pre_score = discussion_scores（讨论轮已经滚动更新）
        # -----------------------------------------------------
        post_scores: Dict[str, float] = {}
        final_reviews: Dict[str, str] = {}

        disc_tail = global_messages[-window:]

        if n_kept > 0:
            if method == "summary_based":
                stage_name = "summary_final"
                tracker.set_total(stage_name, n_kept)

                monitor = getattr(llm, "monitor", None)
                default_workers = max(1, min(max_workers_cfg, n_kept))
                if isinstance(monitor, AdaptiveConcurrencyController):
                    workers = monitor.suggest_workers(default_workers)
                else:
                    workers = default_workers
                workers = max(1, min(workers, n_kept, HARD_MAX_WORKERS))

                logger.info(
                    "STAGE START: summary_final | total_tasks=%d | visible_window=%d | workers=%d",
                    n_kept,
                    window,
                    workers,
                )

                def _summary_final_worker(persona_raw: Dict[str, Any]) -> Tuple[str, float, str]:
                    uid = str(persona_raw.get("uuid"))
                    username = _username_of(persona_raw, uid)
                    persona_txt = _persona_text(persona_raw) if use_persona else None
                    gs = base_payload[uid]["payload"]

                    baseline_sc = round(float(discussion_scores.get(uid, pre_scores.get(uid, 3.0))), disc_decimals)

                    system, user = prompt_summary_final(
                        meta=meta,
                        global_summary=gs,
                        persona_text=persona_txt,
                        username=username,
                        current_score=baseline_sc,
                        discussion_tail=disc_tail if use_discussion else [],
                    )
                    _, parsed = llm.chat_json(
                        system=system,
                        user=user,
                        temperature=float(cfg.llm.temperature),
                        top_p=float(cfg.llm.top_p),
                        max_tokens=int(cfg.llm.max_tokens),
                        trace={
                            "book": book_name,
                            "agent": uid,
                            "method": method,
                            "stage": "summary_final",
                            "chapter": None,
                        },
                    )
                    obj = parsed or {}
                    critique = str(obj.get("critique", "")).strip()

                    try:
                        score = float(obj.get("score", baseline_sc))
                    except Exception:
                        score = baseline_sc
                    score = round(clamp_score(score), score_decimals)

                    return uid, score, critique

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_summary_final_worker, p) for p in kept_personas]
                    for fu in as_completed(futures):
                        uid, sc, critique = fu.result()
                        post_scores[uid] = sc
                        final_reviews[uid] = critique

                logger.info("STAGE DONE: summary_final")

            else:
                if use_discussion:
                    stage_name = "finalize_after_discussion"
                    tracker.set_total(stage_name, n_kept)

                    monitor = getattr(llm, "monitor", None)
                    default_workers = max(1, min(max_workers_cfg, n_kept))
                    if isinstance(monitor, AdaptiveConcurrencyController):
                        workers = monitor.suggest_workers(default_workers)
                    else:
                        workers = default_workers
                    workers = max(1, min(workers, n_kept, HARD_MAX_WORKERS))

                    logger.info(
                        "STAGE START: finalize_after_discussion | total_tasks=%d | visible_window=%d | workers=%d",
                        n_kept,
                        window,
                        workers,
                    )

                    def _finalize_worker(persona_raw: Dict[str, Any]) -> Tuple[str, float, str]:
                        uid = str(persona_raw.get("uuid"))
                        username = _username_of(persona_raw, uid)
                        persona_txt = _persona_text(persona_raw) if use_persona else None

                        baseline_sc = round(float(discussion_scores.get(uid, pre_scores.get(uid, 3.0))), disc_decimals)

                        system, user = prompt_finalize_after_discussion(
                            meta=meta,
                            persona_text=persona_txt,
                            username=username,
                            pre_score=baseline_sc,
                            discussion_tail=disc_tail,
                        )
                        _, parsed = llm.chat_json(
                            system=system,
                            user=user,
                            temperature=float(cfg.llm.temperature),
                            top_p=float(cfg.llm.top_p),
                            max_tokens=int(cfg.llm.max_tokens),
                            trace={
                                "book": book_name,
                                "agent": uid,
                                "method": method,
                                "stage": "finalize_after_discussion",
                                "chapter": None,
                            },
                        )
                        obj = parsed or {}

                        try:
                            fs = float(obj.get("final_score", baseline_sc))
                        except Exception:
                            fs = baseline_sc
                        fs = round(clamp_score(fs), score_decimals)

                        review = str(obj.get("final_review", "")).strip()
                        return uid, fs, review

                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        futures = [ex.submit(_finalize_worker, p) for p in kept_personas]
                        for fu in as_completed(futures):
                            uid, fs, review = fu.result()
                            post_scores[uid] = fs
                            final_reviews[uid] = review

                    logger.info("STAGE DONE: finalize_after_discussion")

        # -----------------------------------------------------
        # 输出 struct：只保留原始数据，不在这里做“book_score”之类的评测聚合
        # -----------------------------------------------------
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
                }
                if use_interest_filter
                else None,
                "discussion": per_agent_discussion.get(uid, []) if (use_discussion and keep) else [],
                "pre_discussion_score": float(pre_scores.get(uid, None)) if uid in pre_scores else None,
                "post_discussion_score": float(post_scores.get(uid, None)) if uid in post_scores else None,
                "final_review": final_reviews.get(uid, None),
                "chapter_evals": None,
                "incremental_steps": None,
                "global_summary": None,
            }

            # 挂上基线评测的 payload（所有 persona 都有，而不仅是 kept）
            if uid in base_payload:
                kind = base_payload[uid]["kind"]
                payload = base_payload[uid]["payload"]

                if kind == "aggregation":
                    agent_obj["chapter_evals"] = payload
                elif kind == "incremental":
                    agent_obj["incremental_steps"] = payload.get("steps", [])
                    # 若没有讨论，则可以把最后的 review 当作 final_review 的初始值
                    if (not use_discussion) and method == "incremental" and agent_obj.get("final_review") is None:
                        agent_obj["final_review"] = str(payload.get("last_state", {}).get("review", "")).strip()
                elif kind == "summary_based":
                    agent_obj["global_summary"] = payload

            agents_out.append(agent_obj)

        out = {
            "book_name": book_name,
            "metadata": {
                "book_name": meta.get("book_name", ""),
                "intro": meta.get("intro", ""),
                "author": meta.get("author", ""),
            },
            "config": _safe_cfg_dump(cfg),
            "agents": agents_out,
            "aggregate": {
                # 不在这里计算 book_score，把聚合评测交给后续脚本
                "n_agents_total": len(personas_raw),
                "n_kept": len(kept_personas),
                "filtered_pre_read": filtered_pre,
            },
        }

        # （可选但推荐）把 run_ts 写进结果，方便你后处理知道是哪次跑的
        out["run_ts"] = run_ts

        # ✅ 固定命名：后处理直接读 outputs/<run_id>/result.json
        out_path = os.path.join(str(cfg.paths.output_root), "result.json")
        write_json(out_path, out)

        logger.info("<<< BOOK END: %s | out=%s", book_name, out_path)

    finally:
        try:
            writer.close()
        except Exception:
            pass

