from __future__ import annotations
import json
import re
from typing import Any, Dict


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    尝试从模型输出中提取第一个 JSON 对象。
    说明：即使你要求“只输出 JSON”，模型偶尔也会夹带解释文字。
    """
    text = text.strip()

    # 直接尝试整体解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 从文本中抓取 {...} 的最外层
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"无法从输出中提取 JSON：{text[:200]}...")
    candidate = m.group(0)
    return json.loads(candidate)


def clamp_score(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, x))
