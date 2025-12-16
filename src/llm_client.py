from __future__ import annotations
from openai import OpenAI
from typing import Dict, Any, Optional
import os
import time


class LLMClient:
    """
    对 OpenAI 兼容接口的轻封装：
    - 统一传入 system + user
    - 统一超时、重试
    - 返回 message.content (字符串)
    """
    def __init__(self, base_url: str, api_key: str, model: str, timeout_sec: int = 120):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.timeout_sec = timeout_sec

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        top_p: float = 0.8,
        max_tokens: int = 800,
        retries: int = 2,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        让模型输出 JSON 字符串。
        注意：有些平台支持 response_format=json_object，有些不支持。
        为了更稳：即便你传了 response_format，也仍然用 system 强约束“只输出 JSON”。
        """
        last_err = None
        for attempt in range(retries + 1):
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
                    timeout=self.timeout_sec,
                    **kwargs,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                # 简单退避，避免频繁触发限流
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"LLM 调用失败：{last_err}")
