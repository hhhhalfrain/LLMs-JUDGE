from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from ..schemas import Chapter
from ..json_utils import extract_json_object, clamp_score
from ..prompts import summary_incremental_prompt, summary_final_eval_prompt


def run_summary_based_for_agent(
    llm,
    meta: Dict,
    chapters: List[Chapter],
    persona_text: Optional[str],
    llm_cfg: Dict,
    score_decimals: int,
    discussion_tail: List[str],
) -> Tuple[Dict, float, str]:
    """
    Summary-based：
    - 每章更新 global_summary（plot/characters/style_excerpts）
    - 最后用 global_summary 一次性评分
    """
    global_summary: Dict = {"plot": "", "characters": "", "style_excerpts": []}

    for ch in chapters:
        p = summary_incremental_prompt(
            chapter_text=ch.text,
            global_summary=global_summary,
            meta=meta,
            persona_text=persona_text,
        )
        text = llm.chat_json(
            system=p["system"],
            user=p["user"],
            temperature=llm_cfg["temperature"],
            top_p=llm_cfg["top_p"],
            max_tokens=llm_cfg["max_tokens"],
            response_format=None,
        )
        obj = extract_json_object(text)
        global_summary = {
            "plot": str(obj.get("plot", "")).strip(),
            "characters": str(obj.get("characters", "")).strip(),
            "style_excerpts": obj.get("style_excerpts", []) or [],
        }

    p2 = summary_final_eval_prompt(
        meta=meta,
        global_summary=global_summary,
        persona_text=persona_text,
        discussion_tail=discussion_tail,
    )
    text2 = llm.chat_json(
        system=p2["system"],
        user=p2["user"],
        temperature=llm_cfg["temperature"],
        top_p=llm_cfg["top_p"],
        max_tokens=llm_cfg["max_tokens"],
        response_format=None,
    )
    obj2 = extract_json_object(text2)
    score = float(obj2.get("score", 3.0))
    score = clamp_score(score)
    score = round(score, score_decimals)
    critique = str(obj2.get("critique", "")).strip()
    return global_summary, score, critique
