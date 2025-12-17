from __future__ import annotations
from openai import OpenAI
from typing import Dict, Any, Optional
import time
import httpx
import logging

from .json_utils import extract_json_object
from .trace_logger import log_llm_call


class LLMClient:
    """
    OpenAI 兼容接口封装：
    - 每次请求可输出完整 trace 日志
    - 支持多线程：建议每个线程单独实例化一个 LLMClient（见下文 LLMClientPool）
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: int = 120,
        logger: Optional[logging.Logger] = None,
        verbose_llm_io: bool = True,
        print_parsed_json: bool = True,
    ):
        self._http = httpx.Client(timeout=httpx.Timeout(timeout_sec), trust_env=True)
        self.client = OpenAI(base_url=base_url, api_key=api_key, http_client=self._http)
        self.model = model

        self.logger = logger
        self.verbose_llm_io = verbose_llm_io
        self.print_parsed_json = print_parsed_json

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        top_p: float = 0.8,
        max_tokens: int = 800,
        retries: int = 2,
        response_format: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> str:
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            resp_text = None
            parsed = None
            try:
                kwargs = {}
                if response_format is not None:
                    kwargs["response_format"] = response_format

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                resp_text = resp.choices[0].message.content
                dt = time.perf_counter() - t0

                if self.print_parsed_json:
                    try:
                        parsed = extract_json_object(resp_text)
                    except Exception:
                        parsed = None

                if self.logger and self.verbose_llm_io:
                    log_llm_call(
                        logger=self.logger,
                        trace=trace or {},
                        system=system,
                        user=user,
                        response_text=resp_text,
                        elapsed_s=dt,
                        parsed_json=parsed,
                        error=None,
                    )
                return resp_text

            except Exception as e:
                dt = time.perf_counter() - t0
                last_err = e
                if self.logger and self.verbose_llm_io:
                    log_llm_call(
                        logger=self.logger,
                        trace=trace or {},
                        system=system,
                        user=user,
                        response_text=resp_text,
                        elapsed_s=dt,
                        parsed_json=parsed,
                        error=str(e),
                    )
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"LLM 调用失败：{last_err}")
