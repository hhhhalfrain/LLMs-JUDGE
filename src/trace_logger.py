from __future__ import annotations
import os
import json
import time
import logging
from typing import Any, Dict, Optional


def setup_logger(log_dir: str, also_file: bool = True, logger_name: str = "llm_trace") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 避免重复加 handler
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


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def log_llm_call(
    logger: logging.Logger,
    trace: Dict[str, Any],
    system: str,
    user: str,
    response_text: Optional[str],
    elapsed_s: float,
    parsed_json: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """
    打印完整“对话内容”：system/user/assistant 输出 + trace 元信息
    注意：用户要求“完整记录”，这里不会截断文本（会很长）
    """
    head = {
        "book": trace.get("book"),
        "agent": trace.get("agent"),
        "method": trace.get("method"),
        "stage": trace.get("stage"),
        "chapter": trace.get("chapter"),
        "elapsed_s": round(elapsed_s, 3),
    }
    logger.info("=" * 90)
    logger.info("TRACE:\n%s", _pretty(head))
    if error:
        logger.info("ERROR:\n%s", error)

    logger.info("-" * 90)
    logger.info("SYSTEM:\n%s", system)
    logger.info("-" * 90)
    logger.info("USER:\n%s", user)
    logger.info("-" * 90)
    logger.info("ASSISTANT(raw):\n%s", response_text if response_text is not None else "(none)")

    if parsed_json is not None:
        logger.info("-" * 90)
        logger.info("ASSISTANT(parsed JSON):\n%s", _pretty(parsed_json))

    logger.info("=" * 90)
