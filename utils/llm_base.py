from __future__ import annotations
import os, json
from typing import Any, Dict, Optional
from openai import OpenAI
from pathlib import Path
import json
from datetime import datetime

LOG_DIR = Path("log")
LOG_DIR.mkdir(parents=True, exist_ok=True)
def log_request_response(request: dict, response: dict):
    """
    保存日志到 log/{timestamp}.json
    timestamp 格式: YYYYMMDDhhmmss (例如 20251029123511)
    直接使用 json.dump(..., default=str) 以简单明了地处理不可序列化对象
    """
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    filepath = LOG_DIR / f"{ts}.json"
    payload = {
        "timestamp": ts,
        "request": request,
        "response": response,
    }
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

class LLMBase:
    def __init__(self):
        from dotenv import load_dotenv
        env_path = os.environ.get("ENV_PATH", ".env")
        load_dotenv(env_path)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
        if not self.OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY (or API_KEY) in .env")
        self.STRONG_TEXT_MODEL = os.getenv("STRONG_TEXT_MODEL", "gpt-5")
        self.WEAK_TEXT_MODEL = os.getenv("WEAK_TEXT_MODEL", "gpt-5-mini")
        self.client = OpenAI(api_key=self.OPENAI_API_KEY,
                             base_url=self.OPENAI_BASE_URL or None)

    def call_structured_json(self, *, model: str, system_prompt: str, user_prompt: str,
                              json_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.7) -> Any:
        request_payload = {
            "component": "LLMBase.call_structured_json",
            "model": model,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

        try:
            if json_schema:
                resp = self.client.responses.create(
                    model=model,
                    temperature=temperature,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text={"format": {"type": "json_schema", "name": "structured_output",
                                     "strict": False, "schema": json_schema}},
                )
                output = json.loads(resp.output_text)
            else:
                resp = self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                output = resp.choices[0].message.content
        except Exception as e:
            log_request_response(request_payload, {"error": str(e), "output_json": None})
            raise

        log_request_response(request_payload, {
            "raw": str(resp),
            "output": output,
        })
        return output
