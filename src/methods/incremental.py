from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from ..schemas import Chapter, IncrementalStep
from ..json_utils import extract_json_object, clamp_score
from ..prompts import incremental_update_prompt


def run_incremental_for_agent(
    llm,
    meta: Dict,
    chapters: List[Chapter],
    persona_text: Optional[str],
    llm_cfg: Dict,
    score_decimals: int,
) -> Tuple[List[IncrementalStep], float]:
    """
    Incremental-updated：
    - 每章更新 summary + review + score
    - pre_score：所有章节后 score 的平均（全文平均分）
    """
    prev_summary = ""
    prev_review = ""
    prev_score = 3.0

    steps: List[IncrementalStep] = []
    scores: List[float] = []

    for ch in chapters:
        p = incremental_update_prompt(
            chapter_text=ch.text,
            prev_summary=prev_summary,
            prev_review=prev_review,
            prev_score=float(prev_score),
            persona_text=persona_text,
            meta=meta,
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

        prev_summary = str(obj.get("summary", "")).strip()
        prev_review = str(obj.get("review", "")).strip()
        score = float(obj.get("score", prev_score))
        score = clamp_score(score)
        score = round(score, score_decimals)
        prev_score = score

        scores.append(score)
        steps.append(
            IncrementalStep(
                chapter_index=int(ch.Number),
                score=score,
                summary=prev_summary,
                review=prev_review,
            )
        )

    pre_score = sum(scores) / max(1, len(scores))
    pre_score = round(pre_score, score_decimals)
    return steps, pre_score
