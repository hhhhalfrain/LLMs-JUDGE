from __future__ import annotations
from typing import Dict, List, Optional
import os
import time
from tqdm import tqdm

from .schemas import Persona, BookMeta, Chapter, AgentResult, RunOutput
from .io_utils import read_json, write_json, list_book_dirs
from .json_utils import extract_json_object, clamp_score
from .prompts import interest_filter_prompt, finalize_after_discussion_prompt
from .discussion import run_discussion

from .methods.aggregation import run_aggregation_for_agent
from .methods.incremental import run_incremental_for_agent
from .methods.summary_based import run_summary_based_for_agent


def _persona_text(persona: Persona) -> str:
    # 只把“persona 描述”喂进去通常最有效，避免塞过多字段浪费 token
    return persona.persona


def _agent_stance_from_result(res: AgentResult) -> str:
    """
    给讨论模块用的“立场摘要”。这里用最简单的策略：
    - aggregation：拿最后一章 comment
    - incremental：拿最后一章 review
    - summary_based：拿 final_review 或 critique
    """
    if res.chapter_evals:
        last = res.chapter_evals[-1]
        return f"My current impression: score trend seems around {res.pre_discussion_score:.1f}. Latest thought: {last.comment}"
    if res.incremental_steps:
        last = res.incremental_steps[-1]
        return f"My current impression: overall around {res.pre_discussion_score:.1f}. Latest review: {last.review}"
    if res.final_review:
        return f"My current impression: {res.final_review}"
    return "My current impression is forming."


def run_all_books(cfg, llm):
    book_dirs = list_book_dirs(cfg.paths.books_root)
    personas_raw = read_json(cfg.paths.personas_json)
    personas = [Persona.model_validate(x) for x in personas_raw][: cfg.experiment.n_agents]

    os.makedirs(cfg.paths.output_root, exist_ok=True)

    for book_dir in book_dirs:
        meta_path = os.path.join(book_dir, "book_metadata.json")
        ch_path = os.path.join(book_dir, "chapters.json")

        meta = BookMeta.model_validate(read_json(meta_path))
        chapters_raw = read_json(ch_path)
        chapters = [Chapter.model_validate(x) for x in chapters_raw]

        run_one_book(cfg, llm, meta, chapters, personas)


def run_one_book(cfg, llm, meta: BookMeta, chapters: List[Chapter], personas: List[Persona]):
    """
    单本书完整流程：
    1) 读前兴趣筛选（可关）
    2) 三种方法之一跑每个 agent
    3) 评分前讨论（可关）
    4) 讨论后 finalize（可选影响最终分）
    5) 聚合输出
    """
    method = cfg.experiment.method
    out_agents: List[AgentResult] = []

    # 全局讨论记录（所有 agent 共享）
    global_discussion: List[str] = []

    # 用于统计
    filtered_pre = 0
    kept = 0

    # 先走兴趣筛选，决定 keep
    decisions: Dict[str, Dict] = {}
    for persona in personas:
        if not cfg.experiment.use_interest_filter:
            decisions[persona.uuid] = {"keep": True, "interest_score": 50.0, "reason": "interest_filter_disabled"}
            continue

        persona_text = _persona_text(persona) if cfg.experiment.use_persona else None
        p = interest_filter_prompt(meta=meta.model_dump(), persona_text=persona_text)
        text = llm.chat_json(
            system=p["system"],
            user=p["user"],
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            max_tokens=cfg.llm.max_tokens,
            response_format=None,
        )
        obj = extract_json_object(text)
        keep = bool(obj.get("keep", True))
        decisions[persona.uuid] = obj

        if not keep:
            filtered_pre += 1

    # 再跑每个 agent 的评测
    for persona in tqdm(personas, desc=f"Book={meta.book_name} method={method}"):
        d = decisions.get(persona.uuid, {"keep": True})
        keep = bool(d.get("keep", True))
        persona_text = _persona_text(persona) if cfg.experiment.use_persona else None

        res = AgentResult(agent_uuid=persona.uuid, kept=keep, interest=None)

        # 记录兴趣筛选结果
        if cfg.experiment.use_interest_filter:
            res.interest = {
                "keep": keep,
                "interest_score": float(d.get("interest_score", 50.0)),
                "reason": str(d.get("reason", "")),
            }
            if not keep:
                res.filtered_reason = str(d.get("reason", "not_interested"))

        if not keep:
            out_agents.append(res)
            continue

        kept += 1

        # 1) 跑基线方法，得到 pre_discussion_score
        if method == "aggregation":
            evals, pre_score = run_aggregation_for_agent(
                llm=llm,
                meta=meta.model_dump(),
                chapters=chapters,
                persona_text=persona_text,
                llm_cfg=cfg.llm.model_dump(),
                score_decimals=cfg.experiment.score_decimals,
            )
            res.chapter_evals = evals
            res.pre_discussion_score = pre_score

        elif method == "incremental":
            steps, pre_score = run_incremental_for_agent(
                llm=llm,
                meta=meta.model_dump(),
                chapters=chapters,
                persona_text=persona_text,
                llm_cfg=cfg.llm.model_dump(),
                score_decimals=cfg.experiment.score_decimals,
            )
            res.incremental_steps = steps
            res.pre_discussion_score = pre_score

        elif method == "summary_based":
            # summary_based 的讨论在“最终评分前”才有意义，因此先不评分：
            # 先构建 global_summary，稍后结合讨论尾巴一起 final_eval
            # 这里暂存 summary，pre_score 先留空
            global_summary, final_score, critique = run_summary_based_for_agent(
                llm=llm,
                meta=meta.model_dump(),
                chapters=chapters,
                persona_text=persona_text,
                llm_cfg=cfg.llm.model_dump(),
                score_decimals=cfg.experiment.score_decimals,
                discussion_tail=[],  # 讨论稍后再注入
            )
            res.global_summary = global_summary
            res.pre_discussion_score = final_score
            res.final_review = critique

        else:
            raise ValueError(f"未知 method: {method}")

        # 2) 评分前讨论（可选）
        if cfg.experiment.use_discussion:
            stance = _agent_stance_from_result(res)
            agent_msgs = run_discussion(
                llm=llm,
                meta=meta.model_dump(),
                agent_uuid=persona.uuid,
                persona_text=persona_text,
                agent_stance=stance,
                global_messages=global_discussion,
                rounds=cfg.experiment.discussion_rounds,
                window=cfg.experiment.discussion_window,
                llm_cfg=cfg.llm.model_dump(),
            )
            res.discussion.extend(agent_msgs)

            # 3) finalize：讨论后再定分（让“讨论是否开启”成为有效变量）
            tail = global_discussion[-cfg.experiment.discussion_window:]
            p = finalize_after_discussion_prompt(
                pre_score=float(res.pre_discussion_score or 3.0),
                meta=meta.model_dump(),
                persona_text=persona_text,
                discussion_tail=tail,
            )
            text = llm.chat_json(
                system=p["system"],
                user=p["user"],
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                max_tokens=cfg.llm.max_tokens,
                response_format=None,
            )
            obj = extract_json_object(text)
            final_score = float(obj.get("final_score", res.pre_discussion_score or 3.0))
            final_score = clamp_score(final_score)
            final_score = round(final_score, cfg.experiment.score_decimals)

            res.post_discussion_score = final_score
            res.final_review = str(obj.get("final_review", "")).strip()

        out_agents.append(res)

    # 聚合整本书结果
    kept_scores = []
    for a in out_agents:
        if not a.kept:
            continue
        if cfg.experiment.use_discussion and cfg.experiment.discussion_affects_score and a.post_discussion_score is not None:
            kept_scores.append(a.post_discussion_score)
        else:
            if a.pre_discussion_score is not None:
                kept_scores.append(a.pre_discussion_score)

    book_score = (sum(kept_scores) / max(1, len(kept_scores))) if kept_scores else None
    if book_score is not None:
        book_score = round(book_score, cfg.experiment.score_decimals)

    aggregate = {
        "book_score": book_score,
        "n_agents_total": len(personas),
        "n_kept": kept,
        "filtered_pre_read": filtered_pre,
    }

    run_out = RunOutput(
        book_name=meta.book_name,
        metadata=meta,
        config=cfg.model_dump(),
        agents=out_agents,
        aggregate=aggregate,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.paths.output_root, f"{meta.book_name}_{cfg.experiment.method}_{ts}.json")
    write_json(out_path, run_out.model_dump())
