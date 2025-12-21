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
        ok, book, agent, method, stage, str(chapter),
        str(done) if done is not None else "-", str(total) if total is not None else "-", str(rem) if rem is not None else "-",
        elapsed_s, user_len, assistant_len, keys, err_part
    )


# =========================
#  自适应并发控制器
# =========================
class AdaptiveConcurrencyController:
    """
    非精细型的自适应并发：
    - 记录全局调用次数 / 错误次数 / 总耗时
    - 若平均耗时 >= latency_threshold 或 错误率 >= error_threshold，则把并发砍半
    - 若平均耗时较低且错误率很低，则缓慢提升并发
    """
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        latency_threshold_s: float = 100.0,
        error_threshold: float = 0.2,
    ):
        self.min_workers = max(1, int(min_workers))
        self.max_workers = max(self.min_workers, int(max_workers))
        self.latency_threshold_s = float(latency_threshold_s)
        self.error_threshold = float(error_threshold)

        self._lock = threading.Lock()
        self._total_calls = 0
        self._error_calls = 0
        self._total_latency = 0.0

    def register_call(self, success: bool, elapsed_s: float) -> None:
        with self._lock:
            self._total_calls += 1
            if not success:
                self._error_calls += 1
            self._total_latency += float(elapsed_s)

    def stats(self) -> Tuple[int, float, float]:
        with self._lock:
            if self._total_calls == 0:
                return 0, 0.0, 0.0
            err_rate = self._error_calls / self._total_calls
            avg_lat = self._total_latency / self._total_calls
            return self._total_calls, err_rate, avg_lat

    def suggest_workers(self, default_workers: int) -> int:
        total, err_rate, avg_lat = self.stats()
        default_workers = max(self.min_workers, min(int(default_workers), self.max_workers))

        if total == 0:
            return default_workers

        workers = default_workers

        # 高延迟或高错误率 → 缩减并发
        if avg_lat >= self.latency_threshold_s or err_rate >= self.error_threshold:
            workers = max(self.min_workers, workers // 2 or 1)
        # 延迟健康且错误率低 → 温和提升一点并发
        elif avg_lat <= self.latency_threshold_s * 0.6 and err_rate <= self.error_threshold * 0.5:
            if workers < self.max_workers:
                workers += 1

        return max(self.min_workers, min(workers, self.max_workers))


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
        if isinstance(err, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError, httpx.ReadError, httpx.ConnectError)):
            return True

        sc = self._get_status_code(err)
        if sc is not None:
            return sc in set(self.retry_status_codes)

        msg = (str(err) or "").lower()
        transient_keywords = [
            "timed out", "timeout", "connection reset", "connection aborted", "temporary failure",
            "remote protocol", "server disconnected", "read error", "connect error", "tls",
        ]
        return any(k in msg for k in transient_keywords)

    def _sleep_backoff(self, attempt_index: int) -> float:
        base = self.retry_base_sleep_sec * (2 ** attempt_index)
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

                # 自适应监控：记录一次失败调用
                if self.monitor is not None:
                    self.monitor.register_call(False, dt)

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
                    error_msg=str(e),
                    error_type=type(e).__name__,
                    error_status_code=self._get_status_code(e),
                )

                if is_last:
                    stage = trace.get("stage")
                    if self.tracker and stage:
                        self.tracker.tick(stage)

                    if self.fail_fast:
                        raise RuntimeError(f"LLM 调用失败（已重试 {attempt+1}/{max_attempts} 次）：{e}") from e
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
    return "".join([
        "READER PERSONA (this is YOU — stay in-character):\n",
        f"{persona_text}\n\n",
        "Persona grounding rules (MANDATORY):\n",
        "- Treat the persona as your identity, values, and taste filter.\n",
        "- Your judgments MUST reflect what this persona would notice, enjoy, dislike, or be sensitive to.\n",
        "- Do NOT become a generic critic. Avoid neutral, academic voice unless the persona would do that.\n",
        "- If uncertain, make the smallest plausible inference consistent with the persona.\n\n",
    ])


def _discussion_block(discussion_tail: List[str]) -> str:
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"
    return "".join([
        "GROUP DISCUSSION (latest messages):\n",
        f"{disc}\n\n",
        "Discussion integration rules (MANDATORY):\n",
        "- You MUST carefully read and integrate the discussion.\n",
        "- Extract 2-4 concrete takeaways (arguments, observations, criticisms, or praise).\n",
        "- Your score MUST be reconsidered after discussion:\n",
        "  * Either change the score, OR clearly justify why you keep the same score despite the discussion.\n",
        "- Avoid herd behavior: if the discussion seems one-sided, you MUST introduce at least one thoughtful counterpoint\n",
        "  (minority view, missing nuance, alternative interpretation, or a caveat).\n",
        "- When you mention discussion, refer to specific points (paraphrase). Do not hand-wave.\n\n",
    ])


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
        "Task: Decide whether YOU (as this persona) would be interested enough to start reading this novel.\n",
        "Be strict: if it does not fit your persona's tastes, values, emotional bandwidth, or genre tolerance, reject it.\n\n",
        "Return JSON with fields:\n",
        "{\n",
        '  "keep": boolean,\n',
        '  "interest_score": number (0-100),\n',
        '  "reason": string\n',
        "}\n",
        "Constraints:\n",
        "- The reason MUST explicitly connect to persona traits/preferences (not generic).\n",
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
        "Context (plot summaries of previous chapters):\n",
        f"{prev}\n\n",
        "Current chapter (English novel text):\n",
        f"{chapter_text}\n\n",
        "Task (MANDATORY, persona-driven):\n",
        "1) Write a short plot summary of THIS chapter (2-4 sentences).\n",
        "2) Give a score for THIS chapter on a 1.0 to 5.0 scale (allow 1 decimal).\n",
        "3) Provide a brief comment (2-4 sentences) emphasizing what THIS persona cares about.\n\n",
        "Return JSON:\n",
        "{\n",
        '  "plot_summary": string,\n',
        '  "score": number (1.0-5.0),\n',
        '  "comment": string\n',
        "}\n",
        "Constraints:\n",
        "- The comment must be persona-specific (avoid generic critique).\n",
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
        "You are reading chapter-by-chapter and updating your opinion as you go.\n\n",
        "Previous running summary (so far):\n",
        f"{prev_summary if prev_summary else '(empty)'}\n\n",
        "Previous review (so far):\n",
        f"{prev_review if prev_review else '(empty)'}\n\n",
        f"Previous overall score (1.0-5.0): {prev_score:.1f}\n\n",
        "Current chapter (English novel text):\n",
        f"{chapter_text}\n\n",
        "Task (MANDATORY, persona-driven):\n",
        "- Update the running summary (concise, <= 180 words).\n",
        "- Update the review (<= 180 words). The review must sound like THIS persona's evolving feelings.\n",
        "- Update the overall score on a 1.0-5.0 scale (allow 1 decimal).\n",
        "- If your score changes, reflect the reason inside the review.\n\n",
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
        "Current global summary JSON (may be empty):\n",
        f"{global_summary}\n\n",
        "New chapter (English novel text):\n",
        f"{chapter_text}\n\n",
        "Task: Update the global summary. Keep it compact and consistent.\n",
        "- plot: <= 220 words total\n",
        "- characters: <= 220 words total\n",
        "- style_excerpts: keep up to 3 short excerpts, each <= 30 words\n\n",
        "Persona constraint (MANDATORY):\n",
        "- When choosing style_excerpts, pick lines THIS persona would actually notice or quote.\n\n",
        "Return JSON:\n",
        "{\n",
        '  "plot": string,\n',
        '  "characters": string,\n',
        '  "style_excerpts": [string, string, ...]\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_summary_final(
    meta: Dict[str, Any],
    global_summary: Dict[str, Any],
    persona_text: Optional[str],
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
        "Global summary JSON:\n",
        f"{global_summary}\n\n",
        _discussion_block(discussion_tail),
        "Task (MANDATORY, persona-driven and discussion-aware):\n",
        "- Write a critique in English (<= 220 words).\n",
        "- Give an overall score from 1.0 to 5.0 (allow 1 decimal).\n\n",
        "Return JSON (must include these fields; you may include extra fields):\n",
        "{\n",
        '  "critique": string,\n',
        '  "score": number (1.0-5.0)\n',
        "}\n",
        "Strong requirements:\n",
        "- The critique must sound like THIS persona (not generic).\n",
        "- Integrate 2-4 discussion takeaways inside the critique.\n",
        "- If the discussion is one-sided, include at least one thoughtful counterpoint.\n",
        "- Your score must be reconsidered after discussion: change it OR justify resistance in the critique.\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_discussion_message(
    meta: Dict[str, Any],
    persona_text: Optional[str],
    agent_stance: str,
    discussion_tail: List[str],
) -> Tuple[str, str]:
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
        "Your current stance (your own view so far):\n",
        f"{agent_stance}\n\n",
        "Latest group discussion messages:\n",
        f"{disc}\n\n",
        "Task: Write ONE short message to the group (1-3 sentences), in English.\n\n",
        "Discussion behavior rules (MANDATORY):\n",
        "- Do NOT just echo the crowd.\n",
        "- If the discussion looks one-sided, you MUST add a counterpoint or missing nuance.\n",
        "- Even if you agree, add a caveat, limitation, or alternative angle.\n",
        "- Your message must reflect THIS persona's priorities.\n\n",
        "Return JSON:\n",
        "{\n",
        '  "message": string\n',
        "}\n",
    ]
    return sys_json_only(), "".join(parts)


def prompt_finalize_after_discussion(
    meta: Dict[str, Any],
    persona_text: Optional[str],
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
        f"Your pre-discussion overall score (1.0-5.0): {pre_score:.1f}\n\n",
        _discussion_block(discussion_tail),
        "Task (MANDATORY):\n",
        "- Decide your final overall score (1.0-5.0, allow 1 decimal).\n",
        "- Provide a final short review (<= 180 words) in English.\n\n",
        "Scoring rules (MANDATORY):\n",
        "- You MUST reconsider the score after discussion: either change it, OR explicitly justify why you keep it.\n",
        "- If discussion is one-sided, include at least one thoughtful counterpoint in your final review.\n\n",
        "Return JSON (must include these fields; you may include extra fields):\n",
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


def _base_eval_cache_path(
    cfg: Any,
    method: str,
    book_name: str,
    persona_key: str,
) -> Optional[str]:
    root = _get_base_eval_root(cfg)
    if not root:
        return None
    fname = sanitize_name(f"{book_name}__persona={persona_key}.json")
    return os.path.join(root, method, fname)


def _load_base_eval_from_disk(
    cfg: Any,
    method: str,
    book_name: str,
    persona_key: str,
) -> Optional[Dict[str, Any]]:
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

    method = str(cfg.experiment.method)
    use_persona = _get_bool(cfg, "experiment.use_persona", True)
    use_discussion = _get_bool(cfg, "experiment.use_discussion", False)
    use_interest_filter = _get_bool(cfg, "experiment.use_interest_filter", True)

    n_agents = len(personas_raw)
    max_workers_cfg = _get_int(cfg, "concurrency.max_workers", min(8, n_agents))
    rounds = _get_int(cfg, "experiment.discussion_rounds", 2)
    window = _get_int(cfg, "experiment.discussion_window", 10)
    score_decimals = _get_int(cfg, "experiment.score_decimals", 1)

    book_name = str(meta.get("book_name", "UNKNOWN"))
    logger.info(
        ">>> BOOK START: %s | method=%s | agents=%d | max_workers_cfg=%d",
        book_name, method, n_agents, max_workers_cfg
    )

    # 每本书一个 tracker + JSONL（完整对话日志）
    tracker = ProgressTracker()
    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(str(cfg.paths.output_root), "logs", f"{book_name}_{method}_{ts}.jsonl")
    writer = JSONLTraceWriter(jsonl_path)
    llm.set_run_context(tracker, writer)
    logger.info("FULL TRACE(JSONL): %s", jsonl_path)

    # -----------------------------------------------------
    # STEP A: 基线评测（对应原 STAGE2，但对所有 persona / nopersona，一次性跑完）
    # -----------------------------------------------------
    base_payload: Dict[str, Dict[str, Any]] = {}
    pre_scores: Dict[str, float] = {}
    stances: Dict[str, str] = {}

    if use_persona:
        personas_for_base = personas_raw
    else:
        personas_for_base = personas_raw[:1] if personas_raw else []

    n_eval_agents = len(personas_for_base)
    n_chapters = len(chapters)

    if n_eval_agents > 0 and n_chapters > 0:
        monitor = getattr(llm, "monitor", None)
        default_workers = max(1, min(max_workers_cfg, n_eval_agents))
        if isinstance(monitor, AdaptiveConcurrencyController):
            workers = monitor.suggest_workers(default_workers)
        else:
            workers = default_workers
        workers = max(1, min(workers, n_eval_agents))

        logger.info(
            "STAGE START: base_eval | method=%s | eval_agents=%d | chapters=%d | workers=%d",
            method, n_eval_agents, n_chapters, workers
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
        workers = max(1, min(workers, n_agents))

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
            decisions[uid] = {
                "keep": True,
                "interest_score": 50.0,
                "reason": "interest_filter_disabled",
            }

    kept_personas = [p for p in personas_raw if bool(decisions.get(str(p.get("uuid")), {}).get("keep", True))]
    filtered_pre = len(personas_raw) - len(kept_personas)
    logger.info(
        "STAGE DONE: interest_filter | kept=%d/%d | filtered_pre_read=%d",
        len(kept_personas), n_agents, filtered_pre
    )

    # -----------------------------------------------------
    # STEP C: 讨论（原 STAGE3）——已禁用多线程，顺序执行
    # -----------------------------------------------------
    global_messages: List[str] = []
    per_agent_discussion: Dict[str, List[Dict[str, Any]]] = {
        str(p.get("uuid")): [] for p in kept_personas
    }

    n_kept = len(kept_personas)

    if use_discussion and n_kept > 0:
        for r in range(1, rounds + 1):
            stage_name = f"discussion_round_{r}"
            tracker.set_total(stage_name, n_kept)

            # 讨论阶段不再使用多线程，强制顺序执行，避免端点在高压下“炸锅”
            logger.info(
                "STAGE START: %s | total_tasks=%d | visible_window=%d | workers=%d",
                stage_name, n_kept, window, 1
            )

            tail = global_messages[-window:]

            def _disc_worker(persona_raw: Dict[str, Any]) -> Tuple[str, str]:
                uid = str(persona_raw.get("uuid"))
                persona_txt = _persona_text(persona_raw) if use_persona else None
                stance = stances.get(uid, "My impression is forming.")
                system, user = prompt_discussion_message(meta, persona_txt, stance, tail)
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
                msg = str(obj.get("message", "")).strip() or "(no message)"
                return uid, msg

            # 🔴 这里改成顺序调用，按 persona 顺序一个个来
            round_msgs: List[Tuple[str, str]] = []
            for persona_raw in kept_personas:
                uid, msg = _disc_worker(persona_raw)
                round_msgs.append((uid, msg))

            # 固定顺序合并，保证可复现
            for uid, msg in sorted(round_msgs, key=lambda x: x[0]):
                global_messages.append(msg)
                per_agent_discussion[uid].append({"round": r, "message": msg})

            logger.info("STAGE DONE: %s | total_messages=%d", stage_name, len(global_messages))


    # -----------------------------------------------------
    # STEP D: 最终打分（原 STAGE4）
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
            workers = max(1, min(workers, n_kept))

            logger.info(
                "STAGE START: summary_final | total_tasks=%d | visible_window=%d | workers=%d",
                n_kept, window, workers
            )

            def _summary_final_worker(persona_raw: Dict[str, Any]) -> Tuple[str, float, str]:
                uid = str(persona_raw.get("uuid"))
                persona_txt = _persona_text(persona_raw) if use_persona else None
                gs = base_payload[uid]["payload"]

                system, user = prompt_summary_final(
                    meta,
                    gs,
                    persona_txt,
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
                    score = float(obj.get("score", 3.0))
                except Exception:
                    score = 3.0
                score = round(clamp_score(score), score_decimals)
                return uid, score, critique

            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_summary_final_worker, p) for p in kept_personas]
                for fu in as_completed(futures):
                    uid, sc, critique = fu.result()
                    pre_scores[uid] = sc  # 对 summary_based 来说，最终得分就是这里的 score
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
                workers = max(1, min(workers, n_kept))

                logger.info(
                    "STAGE START: finalize_after_discussion | total_tasks=%d | visible_window=%d | workers=%d",
                    n_kept, window, workers
                )

                def _finalize_worker(persona_raw: Dict[str, Any]) -> Tuple[str, float, str]:
                    uid = str(persona_raw.get("uuid"))
                    persona_txt = _persona_text(persona_raw) if use_persona else None
                    pre = float(pre_scores.get(uid, 3.0))

                    system, user = prompt_finalize_after_discussion(meta, persona_txt, pre, disc_tail)
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
                        fs = float(obj.get("final_score", pre))
                    except Exception:
                        fs = pre
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
            } if use_interest_filter else None,
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

    out_path = os.path.join(str(cfg.paths.output_root), f"{book_name}_{method}_{ts}.json")
    write_json(out_path, out)

    writer.close()
    logger.info("<<< BOOK END: %s | out=%s", book_name, out_path)
