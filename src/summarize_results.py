# -*- coding: utf-8 -*-
"""
汇总 batch 结果到一个 xlsx 工作簿（包含 per_agent / per_book 两个工作表）

使用方式：
1. 把本文件放到项目里，比如：scripts/summarize_results_to_xlsx.py
2. 修改 BATCH_ID 使其指向你要分析的那一批实验
3. python summarize_results_to_xlsx.py
"""

import json
from pathlib import Path

import pandas as pd


# =========================
#  配置区（按需修改）
# =========================
# 和 batch_run.py 里的一样
BATCH_ID = "4agents t1"

# PROJECT_ROOT 认为是：本脚本所在目录的上一级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_ROOT = PROJECT_ROOT / "runs" / "batch" / BATCH_ID
OUTPUTS_ROOT = BATCH_ROOT / "outputs"

# 输出的 Excel 文件路径
OUT_XLSX = BATCH_ROOT / f"{BATCH_ID}_summary.xlsx"


# =========================
#  工具函数
# =========================
def iter_result_json_files(outputs_root: Path):
    """
    遍历 outputs_root 下所有实验目录，找到真正的结果 JSON 文件：
    - 目录结构类似：
        outputs/
          book=Fourth Wing ...__m=aggregation__...
            _DONE.json
            _RUNNING.json (可能不存在)
            logs/
            Fourth Wing (The Empyrean, #1)_aggregation_20251221_173000.json   <- 我们要的
    """
    for exp_dir in sorted(outputs_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        # 只看这一层的 .json 文件，忽略 _DONE/_RUNNING 等控制文件
        for f in exp_dir.glob("*.json"):
            if f.name.startswith("_"):
                continue
            yield f


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================
#  主逻辑
# =========================
def main():
    if not OUTPUTS_ROOT.exists():
        raise RuntimeError(f"outputs 目录不存在：{OUTPUTS_ROOT}")

    rows = []

    print(f"[INFO] 扫描目录: {OUTPUTS_ROOT}")

    for json_path in iter_result_json_files(OUTPUTS_ROOT):
        data = load_json(json_path)

        book_name = data.get("book_name") or data.get("metadata", {}).get("book_name") or "UNKNOWN"
        meta = data.get("metadata", {}) or {}
        author = meta.get("author", "")
        intro = meta.get("intro", "")

        cfg = data.get("config", {}) or {}
        exp_cfg = cfg.get("experiment", {}) or {}

        method = exp_cfg.get("method")
        use_persona = exp_cfg.get("use_persona")
        use_discussion = exp_cfg.get("use_discussion")
        use_interest_filter = exp_cfg.get("use_interest_filter")
        discussion_rounds = exp_cfg.get("discussion_rounds")
        discussion_window = exp_cfg.get("discussion_window")
        n_agents_cfg = exp_cfg.get("n_agents")
        score_decimals = exp_cfg.get("score_decimals")
        discussion_affects_score = exp_cfg.get("discussion_affects_score")
        chapter_batch_size = exp_cfg.get("chapter_batch_size")

        aggregate = data.get("aggregate", {}) or {}
        n_agents_total = aggregate.get("n_agents_total")
        n_kept = aggregate.get("n_kept")
        filtered_pre_read = aggregate.get("filtered_pre_read")

        run_dir_name = json_path.parent.name

        agents = data.get("agents", []) or []
        if not agents:
            # 理论上不应该发生，但防御一下
            print(f"[WARN] 文件中没有 agents: {json_path}")
            continue

        for agent in agents:
            agent_uuid = agent.get("agent_uuid")
            kept = agent.get("kept")

            interest = agent.get("interest")
            interest_keep = None
            interest_score = None
            interest_reason = None
            if isinstance(interest, dict):
                interest_keep = interest.get("keep")
                interest_score = interest.get("interest_score")
                interest_reason = interest.get("reason")

            pre_discussion_score = agent.get("pre_discussion_score")
            post_discussion_score = agent.get("post_discussion_score")

            # 对于 summary_based：post_discussion_score 可能是 None，
            # 这时把 pre_discussion_score 当作 final_score。
            if post_discussion_score is not None:
                final_score = post_discussion_score
            else:
                final_score = pre_discussion_score

            row = {
                # 结构信息
                "batch_id": BATCH_ID,
                "run_dir": run_dir_name,
                "json_file": json_path.name,

                # 书本信息
                "book_name": book_name,
                "author": author,
                "intro": intro,

                # 实验配置
                "method": method,
                "use_persona": use_persona,
                "use_discussion": use_discussion,
                "use_interest_filter": use_interest_filter,
                "discussion_rounds": discussion_rounds,
                "discussion_window": discussion_window,
                "n_agents_cfg": n_agents_cfg,
                "score_decimals": score_decimals,
                "discussion_affects_score": discussion_affects_score,
                "chapter_batch_size": chapter_batch_size,

                # 聚合统计（整体层面，同一 json 文件内所有 agent 共享）
                "n_agents_total": n_agents_total,
                "n_kept": n_kept,
                "filtered_pre_read": filtered_pre_read,

                # agent 信息
                "agent_uuid": agent_uuid,
                "agent_kept": kept,
                "interest_keep": interest_keep,
                "interest_score": interest_score,
                "interest_reason": interest_reason,
                "pre_discussion_score": pre_discussion_score,
                "post_discussion_score": post_discussion_score,
                "final_score": final_score,
            }

            rows.append(row)

    if not rows:
        raise RuntimeError("没有收集到任何数据行，检查 outputs 目录是否为空。")

    df = pd.DataFrame(rows)

    # ========= per_book 聚合一张表（方便看整体表现） =========
    book_rows = []
    group_keys = ["run_dir", "book_name", "method"]

    for (run_dir, book_name, method), g in df.groupby(group_keys):
        # 用 group 里的第一行拿公共配置即可
        first = g.iloc[0]

        book_rows.append(
            {
                "batch_id": first["batch_id"],
                "run_dir": run_dir,
                "book_name": book_name,
                "method": method,
                "use_persona": first["use_persona"],
                "use_discussion": first["use_discussion"],
                "use_interest_filter": first["use_interest_filter"],
                "discussion_rounds": first["discussion_rounds"],
                "discussion_window": first["discussion_window"],
                "n_agents_cfg": first["n_agents_cfg"],
                "score_decimals": first["score_decimals"],
                "discussion_affects_score": first["discussion_affects_score"],
                "chapter_batch_size": first["chapter_batch_size"],
                "n_agents_total": int(g["n_agents_total"].max()),
                "n_kept": int(g["n_kept"].max()),
                "mean_pre_discussion_score": g["pre_discussion_score"].mean(),
                "mean_post_discussion_score": g["post_discussion_score"].mean(),
                "mean_final_score": g["final_score"].mean(),
                "mean_interest_score": g["interest_score"].mean(),
            }
        )

    df_book = pd.DataFrame(book_rows)

    # ========= 写入 Excel =========
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="per_agent", index=False)
        df_book.to_excel(writer, sheet_name="per_book", index=False)

    print(f"[DONE] 共写入 {len(df)} 行（per_agent），"
          f"{len(df_book)} 行（per_book） -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
