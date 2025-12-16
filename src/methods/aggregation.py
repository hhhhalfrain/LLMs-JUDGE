from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from ..schemas import Chapter, ChapterEval
from ..json_utils import extract_json_object, clamp_score
from ..prompts import aggregation_chapter_prompt


def run_aggregation_for_agent(
    llm,
    meta: Dict,
    chapters: List[Chapter],
    persona_text: Optional[str],
    llm_cfg: Dict,
    score_decimals: int,
) -> Tuple[List[ChapterEval], float]:
    """
    Aggregation-based：
    - 每章：输入 metadata + 当前章 + 之前章节 plot summaries
    - 输出：plot_summary + chapter_score + comment
    - pre_score：章节分的平均
    """
    prev_plot_summaries: List[str] = []
    evals: List[ChapterEval] = []
    scores: List[float] = []

    for ch in chapters:
        p = aggregation_chapter_prompt(
            meta=meta,
            chapter_text=ch.text,
            prev_plot_summaries=prev_plot_summaries,
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

        plot_summary = str(obj.get("plot_summary", "")).strip()
        comment = str(obj.get("comment", "")).strip()
        score = float(obj.get("score", 3.0))
        score = clamp_score(score)
        score = round(score, score_decimals)

        prev_plot_summaries.append(plot_summary)
        scores.append(score)
        evals.append(
            ChapterEval(
                chapter_index=int(ch.Number),
                score=score,
                plot_summary=plot_summary,
                comment=comment,
            )
        )

    pre_score = sum(scores) / max(1, len(scores))
    pre_score = round(pre_score, score_decimals)
    return evals, pre_score
