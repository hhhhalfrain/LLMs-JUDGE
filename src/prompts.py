from __future__ import annotations
from typing import Dict, List, Optional


def _sys_json_only() -> str:
    # 注意：为了提高结构化输出稳定性，system 里强调只输出 JSON
    return (
        "You are a strict JSON generator. "
        "Return ONLY a valid JSON object and nothing else. "
        "Do not add markdown fences. Do not add commentary."
    )


def interest_filter_prompt(meta: Dict, persona_text: Optional[str]) -> Dict:
    """
    读前兴趣筛选：只看 book_name + intro。
    输出：keep, interest_score(0-100), reason
    """
    book_name = meta["book_name"]
    intro = meta["intro"]

    user = (
        f"Book metadata:\n"
        f"- Title: {book_name}\n"
        f"- Blurb: {intro}\n\n"
    )

    if persona_text:
        user += (
            f"Reader persona:\n{persona_text}\n\n"
            "Task: Decide whether this reader would be interested enough to read this novel.\n"
        )
    else:
        user += (
            "Task: Decide whether a generic reader would be interested enough to read this novel.\n"
        )

    user += (
        "Return JSON with fields:\n"
        "{\n"
        '  "keep": boolean,\n'
        '  "interest_score": number (0-100),\n'
        '  "reason": string\n'
        "}\n"
    )

    return {
        "system": _sys_json_only(),
        "user": user,
    }


def aggregation_chapter_prompt(meta: Dict, chapter_text: str, prev_plot_summaries: List[str], persona_text: Optional[str]) -> Dict:
    """
    Aggregation-based：每章单独评一次，同时产出该章 plot summary（短）。
    """
    book_name = meta["book_name"]
    intro = meta["intro"]
    prev = "\n".join([f"- {s}" for s in prev_plot_summaries]) if prev_plot_summaries else "(none)"

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
    )
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Context (plot summaries of previous chapters):\n"
        f"{prev}\n\n"
        "Current chapter (English novel text):\n"
        f"{chapter_text}\n\n"
        "Task:\n"
        "1) Write a short plot summary of THIS chapter (2-4 sentences).\n"
        "2) Give a score for THIS chapter on a 1.0 to 5.0 scale (allow 1 decimal).\n"
        "3) Provide a brief comment (2-4 sentences) in English.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "plot_summary": string,\n'
        '  "score": number (1.0-5.0),\n'
        '  "comment": string\n'
        "}\n"
    )
    return {"system": _sys_json_only(), "user": user}


def incremental_update_prompt(chapter_text: str, prev_summary: str, prev_review: str, prev_score: float, persona_text: Optional[str], meta: Dict) -> Dict:
    """
    Incremental-updated：越读越改观，每章更新 summary + review + score。
    """
    book_name = meta["book_name"]
    intro = meta["intro"]

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
    )
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Previous running summary (so far):\n"
        f"{prev_summary if prev_summary else '(empty)'}\n\n"
        "Previous review (so far):\n"
        f"{prev_review if prev_review else '(empty)'}\n\n"
        f"Previous overall score (1.0-5.0): {prev_score:.1f}\n\n"
        "Current chapter (English novel text):\n"
        f"{chapter_text}\n\n"
        "Task:\n"
        "- Update the running summary (keep it concise, <= 180 words).\n"
        "- Update the review (<= 180 words).\n"
        "- Update the overall score on a 1.0-5.0 scale (allow 1 decimal).\n\n"
        "Return JSON:\n"
        "{\n"
        '  "summary": string,\n'
        '  "review": string,\n'
        '  "score": number (1.0-5.0)\n'
        "}\n"
    )
    return {"system": _sys_json_only(), "user": user}


def summary_incremental_prompt(chapter_text: str, global_summary: Dict, meta: Dict, persona_text: Optional[str]) -> Dict:
    """
    Summary-based：增量式更新全书 summary（plot + characters + style_excerpts）。
    """
    book_name = meta["book_name"]
    intro = meta["intro"]

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
    )
    if persona_text:
        user += f"Reader persona:\n{persona_text}\n\n"

    user += (
        "Current global summary JSON (may be empty):\n"
        f"{global_summary}\n\n"
        "New chapter (English novel text):\n"
        f"{chapter_text}\n\n"
        "Task: Update the global summary. Keep it compact and consistent.\n"
        "- plot: <= 220 words total\n"
        "- characters: <= 220 words total\n"
        "- style_excerpts: keep up to 3 short excerpts, each <= 30 words\n\n"
        "Return JSON:\n"
        "{\n"
        '  "plot": string,\n'
        '  "characters": string,\n'
        '  "style_excerpts": [string, string, ...]\n'
        "}\n"
    )
    return {"system": _sys_json_only(), "user": user}


def summary_final_eval_prompt(meta: Dict, global_summary: Dict, persona_text: Optional[str], discussion_tail: List[str]) -> Dict:
    """
    Summary-based：最终一次性写 critique + 打分。
    """
    book_name = meta["book_name"]
    intro = meta["intro"]
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
        f"Global summary JSON:\n{global_summary}\n\n"
        "Optional group discussion (latest messages):\n"
        f"{disc}\n\n"
        "Task:\n"
        "- Write a critique in English (<= 220 words).\n"
        "- Give an overall score from 1.0 to 5.0 (allow 1 decimal).\n\n"
        "Return JSON:\n"
        "{\n"
        '  "critique": string,\n'
        '  "score": number (1.0-5.0)\n'
        "}\n"
    )
    if persona_text:
        user += "\nNote: The critique should be consistent with the persona's tastes."
    return {"system": _sys_json_only(), "user": user}


def discussion_message_prompt(meta: Dict, persona_text: Optional[str], agent_stance: str, discussion_tail: List[str]) -> Dict:
    """
    讨论发言：每个 agent 看到最新 n 条消息 + 自己立场。
    """
    book_name = meta["book_name"]
    intro = meta["intro"]
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
        f"Your current stance:\n{agent_stance}\n\n"
        "Latest group discussion messages:\n"
        f"{disc}\n\n"
        "Task: Write ONE short message to the group (1-3 sentences), in English.\n"
        "Return JSON:\n"
        "{\n"
        '  "message": string\n'
        "}\n"
    )
    if persona_text:
        user += "\nStay in-character with the persona."
    return {"system": _sys_json_only(), "user": user}


def finalize_after_discussion_prompt(pre_score: float, meta: Dict, persona_text: Optional[str], discussion_tail: List[str]) -> Dict:
    """
    讨论后最终定分：让讨论影响最终分（可控开关）。
    """
    disc = "\n".join([f"- {m}" for m in discussion_tail]) if discussion_tail else "(none)"
    book_name = meta["book_name"]
    intro = meta["intro"]

    user = (
        f"Novel metadata:\nTitle: {book_name}\nBlurb: {intro}\n\n"
        f"Your pre-discussion overall score (1.0-5.0): {pre_score:.1f}\n\n"
        "Latest group discussion messages:\n"
        f"{disc}\n\n"
        "Task:\n"
        "- Decide your final overall score (1.0-5.0, allow 1 decimal).\n"
        "- Provide a final short review (<= 180 words) in English.\n\n"
        "Return JSON:\n"
        "{\n"
        '  "final_score": number (1.0-5.0),\n'
        '  "final_review": string\n'
        "}\n"
    )
    if persona_text:
        user += "\nStay consistent with the persona."
    return {"system": _sys_json_only(), "user": user}
