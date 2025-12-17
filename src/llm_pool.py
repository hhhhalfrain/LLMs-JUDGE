from __future__ import annotations
import threading
import logging
from .llm_client import LLMClient


class LLMClientPool:
    """
    线程本地池：每个线程懒加载一个 LLMClient，避免共享底层连接导致并发问题。
    """
    def __init__(self, base_url: str, api_key: str, model: str, timeout_sec: int,
                 logger: logging.Logger, verbose_llm_io: bool, print_parsed_json: bool):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.logger = logger
        self.verbose_llm_io = verbose_llm_io
        self.print_parsed_json = print_parsed_json
        self.local = threading.local()

    def get(self) -> LLMClient:
        if not hasattr(self.local, "client"):
            self.local.client = LLMClient(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                timeout_sec=self.timeout_sec,
                logger=self.logger,
                verbose_llm_io=self.verbose_llm_io,
                print_parsed_json=self.print_parsed_json,
            )
        return self.local.client
