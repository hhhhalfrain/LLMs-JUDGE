import asyncio
import json
import os
import random
import sqlite3
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.prompts import TextPrompt
import oasis
from oasis import (
    ActionType,
    LLMAction,
    ManualAction,
    SocialAgent,
    AgentGraph,
    UserInfo,
    Platform,
)
from oasis.clock import Clock

# ===== 路径 & 常量配置（作为默认值，可以被 config 覆盖） =====

BOOK_PATH = "data/books/Things Have Gotten Worse Since We Last Spoke and Other -- Eric LaRocca_3000_chapter_summaries.json"
BOOK_NAME = "Things Have Gotten Worse Since We Last Spoke and Other -- Eric LaRocca"
PERSONA_PATH = "./data/reddit/personas_sample_100.json"  # 可以换成 1M personas 文件

RESULT_ROOT = Path("./result")
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("./data/db")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# 连载窗口大小，例如 5 表示每个时间节点看到 [t, t+4] 的章节
WINDOW_SIZE = 5

# 本次实验创建多少个智能体（全局总数，不再做章节×persona 笛卡尔积）
NUM_EXPERIMENT_AGENTS = 30

# 是否只用前 N 章来做实验；为 None 表示使用全部章节
MAX_CHAPTERS: int | None = None

# 随机种子，保证可复现
RANDOM_SEED = 42


# ===== 数据加载 & prompt 构造 =====

def load_book_chapters(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        chapters = json.load(f)
    return chapters


def load_personas(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        personas = json.load(f)
    return personas


def make_user_info_template() -> TextPrompt:
    return TextPrompt(
        """
# OBJECTIVE
You are a Reddit user who is reading the horror book "{book_name}" as a serialized story.
You react to ONE specific chapter, described in your profile, and participate in an ongoing subreddit discussion about this chapter.

# SELF-DESCRIPTION
personality: {persona}
sex: {sex}
age: {age}
education_level: {education_level}
occupation: {occupation}

# DISCUSSION BEHAVIOR
- IMPORTANT: If you find that positive posts or comments dominate, you tend to refute them, and vice versa
- When you CREATE_POST or CREATE_COMMENT, you MUST clearly explain WHY you like or dislike specific aspects of this chapter.
- Always give concrete reasons: specific events, character moments, pacing, atmosphere, writing style, emotional impact, etc.
- Pay attention to coherence, relevance and logic:
  - coherence: does the story flow naturally without random jumps?
  - relevance: is the text clearly connected to the title/premise and previous chapters?
  - logic: do character motivations and events make sense, without breaking common sense for no reason?
- When you strongly feel that coherence, relevance or logic are bad, reflect this in your tone and in your use of downvotes and critical comments.
- Do NOT be vague. Avoid generic statements like "it's good" or "I don't like it" without explanation.
- Write in a casual Reddit voice that is consistent with your SELF-DESCRIPTION and personal tastes.

# INPUT
book_name: {book_name}
chapter_number: {chapter_number}
story_so_far_summary: {story_so_far_summary}
chapter_text: {chapter_text}

"""
    )


def make_chapter_rating_prompt(
    book_name: str,
    story_so_far_summary: str,
    chapter_text: str,
) -> str:
    return f"""
You are the same Reddit user described in your profile (with specific age, occupation, personality and tastes).
You have already read the previous chapters summary (story_so_far_summary) and the current chapter text (chapter_text)
stored in your profile, and you have been following the subreddit discussion about this chapter.

Now you need to give a careful evaluation of THIS chapter of the book:

[Book Title]
{book_name}

Your score should be based on TWO groups of factors:

[1] TEXT QUALITY (objective-ish)
Carefully evaluate the quality of this chapter itself, regardless of your personal taste.
Use the story_so_far_summary and chapter_text in your profile as the source of content.

- COHERENCE:
  - Do sentences, paragraphs and scenes flow naturally?
  - Are there abrupt jumps, contradictions, or random insertions that break the reading experience?

- RELEVANCE:
  - Is this chapter clearly connected to the book title, premise and the previous chapters described in story_so_far_summary?
  - Does it stay on topic, or does it drift into content that feels unrelated or filler?

- LOGIC AND NARRATIVE TENSION:
  - Do character motivations, actions and events make sense?
  - Are causal relations between events reasonable?
  - Does the chapter maintain or build narrative tension instead of killing it for no good reason?
  - Does it avoid obvious violations of common sense or the story's own rules?

If coherence, relevance OR logic are seriously broken
(for example, nonsense ordering, off-topic sections, impossible events with no explanation,
or very flat and pointless storytelling),
you MUST give a LOW score (1 or 2), even if some sentences are well written.

[2] PERSONAL INTEREST ALIGNMENT (subjective)
Now adjust your judgement according to your own persona and reading preferences:

- Based on your persona (personality, age, occupation, etc.), decide how much this chapter fits what YOU usually enjoy
  (e.g. genre, themes, darkness level, pacing, emotional tone, intellectual depth).
- If this chapter strongly matches your interests and you would genuinely recommend it to people like you,
  you may slightly INCREASE the score (for example +2, staying within [1,5]).
- If this chapter clearly does NOT match your tastes, bores you, or feels unpleasant in a way you personally dislike,
  you may slightly DECREASE the score (for example -2, staying within [1,5]).

[SCORING SCALE]
Use the FULL 1–5 scale. 

First, internally decide a base score from 1 to 5 based mainly on TEXT QUALITY
(using the chapter and summary in your profile).
Then adjust it slightly up or down (at most 2 point) according to PERSONAL INTEREST,
and clamp the final score to [1, 5].

[YOUR TASK]
Based on your personal reading preferences AND the Reddit discussion so far,
please rate the current chapter of this book and briefly explain your reasoning
using the criteria above.

You only need to do two things:

1. Give this chapter a FINAL integer score from 1 to 5  
   (1 = very bad, 5 = excellent), after combining TEXT QUALITY and PERSONAL INTEREST as described above.

2. Provide 1–3 sentences of natural-language explanation for your score.

Strictly output your answer in the following JSON format, with no extra text, no commentary, and no backticks:

{{
  "score": INTEGER_SCORE,
  "reason": "One to three sentences explaining your score, explicitly referencing the criteria above."
}}
"""


# ===== trace.info 解析：针对你给的真实格式 =====

def parse_interview_score(info: Dict[str, Any]) -> Tuple[int | None, str | None]:
    """
    解析 trace.info 里的评分结果。

    当前 info 的结构类似：
    {
        "prompt": "...",
        "response": "{ \"score\": 4, \"reason\": \"...\" }",
        "interview_id": "..."
    }
    """
    if not isinstance(info, dict):
        return None, None

    raw = info.get("response")
    if not isinstance(raw, str):
        return None, None

    raw = raw.strip()
    if not raw:
        return None, None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 如果后续有脏输出，可以在这里再做截取 {...} 的 fallback
        return None, None

    score_raw = data.get("score")
    reason = data.get("reason")

    try:
        score = int(score_raw)
    except Exception:
        score = None

    if reason is not None:
        reason = str(reason)

    return score, reason


# ===== 单次实验：根据 config 运行一轮仿真 =====

async def run_one_experiment(config: Dict[str, Any]) -> None:
    """
    运行一次完整实验。config 是一个 dict，支持的字段示例：

    {
      "task_name": "exp_window5_agents30",
      "book_path": "data/books/xxx.json",
      "book_name": "Things Have Gotten Worse ...",
      "persona_path": "data/reddit/personas_sample_100.json",
      "num_agents": 30,
      "window_size": 5,
      "max_chapters": null,
      "random_seed": 42
    }
    """
    # 1. 读取配置 & 准备结果目录
    task_name = config.get("task_name", "default_experiment")
    result_dir = RESULT_ROOT / task_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 把本次 config 也存一份，便于之后复现
    with open(result_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    book_path = config.get("book_path", BOOK_PATH)
    book_name = config.get("book_name", BOOK_NAME)
    persona_path = config.get("persona_path", PERSONA_PATH)
    num_experiment_agents = config.get("num_agents", NUM_EXPERIMENT_AGENTS)
    window_size = config.get("window_size", WINDOW_SIZE)
    max_chapters = config.get("max_chapters", MAX_CHAPTERS)
    random_seed = config.get("random_seed", RANDOM_SEED)

    random.seed(random_seed)

    # 2. 底层模型
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
    )

    # 3. agent 可用动作全集（具体权限后面按角色裁剪）
    all_actions = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
    ]

    # 4. 加载章节 & persona
    book_data = load_book_chapters(book_path)
    if max_chapters is not None:
        book_data = book_data[:max_chapters]

    personas = load_personas(persona_path)

    num_chapters = len(book_data)
    if num_chapters == 0:
        raise RuntimeError("No chapters loaded from book summaries.")

    if len(personas) == 0:
        raise RuntimeError("No personas loaded.")

    # 5. 实验规模：本次只创建 num_experiment_agents 个智能体
    num_agents = min(num_experiment_agents, len(personas))
    selected_personas = random.sample(personas, num_agents)

    # 为智能体分配“动作角色”：10% 只发帖、30% 只评论、60% 只点赞/点踩
    num_posters = max(1, int(num_agents * 0.1))           # 至少 1 个发帖的
    num_commenters = max(1, int(num_agents * 0.3))        # 至少 1 个评论的
    num_voters = max(0, num_agents - num_posters - num_commenters)

    roles: List[str] = (
        ["poster"] * num_posters
        + ["commenter"] * num_commenters
        + ["voter"] * num_voters
    )

    # 防止四舍五入溢出，多余的截掉
    roles = roles[:num_agents]
    random.shuffle(roles)

    print(
        f"[SYSTEM][{task_name}] Using {num_chapters} chapters, "
        f"{len(personas)} personas loaded, "
        f"{num_agents} agents selected for this experiment."
    )

    # 6. 为这 num_agents 个智能体分配章节：
    #    - 如果 agents <= chapters：让每个 agent 绑定一个不同章节（分散）
    #    - 如果 agents > chapters：对章节索引做循环填充再 shuffle
    if num_agents <= num_chapters:
        chapter_indices = random.sample(range(num_chapters), k=num_agents)
    else:
        base = list(range(num_chapters))
        times = num_agents // num_chapters
        rem = num_agents % num_chapters
        chapter_indices = base * times + random.sample(base, rem)
        random.shuffle(chapter_indices)

    # 7. 创建单一 AgentGraph 和时间调度表
    agent_graph = AgentGraph()
    user_info_template = make_user_info_template()

    agent_by_id: Dict[int, SocialAgent] = {}

    # 每个时间节点的“发言名单”，时间节点从 1 到 num_chapters（连载进度）
    schedule: Dict[int, List[SocialAgent]] = {
        t: [] for t in range(1, num_chapters + 1)
    }

    for agent_id, (persona, chapter_idx) in enumerate(
        zip(selected_personas, chapter_indices)
    ):
        chapter = book_data[chapter_idx]
        chapter_number = int(chapter["Number"])

        # 一次性 profile：persona + 被分配到的章节 + 角色
        role = roles[agent_id]
        profile = dict(persona)
        profile.update(
            {
                "book_name": book_name,
                "chapter_number": chapter_number,
                "story_so_far_summary": chapter["story_so_far_summary"],
                "chapter_text": chapter["chapter_text"],
                "role": role,  # 方便之后分析：poster / commenter / voter
            }
        )

        userinfo = UserInfo(
            name=profile.get("name", f"User_{agent_id}"),
            user_name=profile.get("username", f"user_{agent_id}"),
            profile=profile,
        )

        # 根据角色限制允许的动作集合
        if role == "poster":
            allowed_actions = [
                ActionType.CREATE_POST,
            ]
        elif role == "commenter":
            allowed_actions = [
                ActionType.CREATE_COMMENT,
            ]
        else:  # "voter"
            allowed_actions = [
                ActionType.LIKE_POST,
                ActionType.DISLIKE_POST,
                ActionType.LIKE_COMMENT,
                ActionType.DISLIKE_COMMENT,
            ]

        sa = SocialAgent(
            agent_id=agent_id,
            user_info=userinfo,
            user_info_template=user_info_template,
            agent_graph=agent_graph,
            model=openai_model,
            available_actions=allowed_actions,
            interview_record=True,
        )
        agent_graph.add_agent(sa)
        agent_by_id[agent_id] = sa

        # 时间窗口约束：agent 只能在窗口内某个时间点发言一次
        t_min = max(1, chapter_number - window_size + 1)
        t_max = chapter_number
        assigned_t = random.randint(t_min, t_max)

        schedule[assigned_t].append(sa)

    total_agents = len(agent_by_id)
    print(f"[SYSTEM][{task_name}] Created {total_agents} one-shot agents.")

    # 8. 创建环境 & 数据库（每个实验用独立 db 文件）
    db_path = str(DATA_ROOT / f"{task_name}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    platform = Platform(
        db_path=db_path,
        sandbox_clock=Clock(k=60),
        show_score=True,
        allow_self_rating=False,
        recsys_type="reddit",
        refresh_rec_post_count=5,
        max_rec_post_len=10,
        use_openai_embedding=False,
    )

    env = oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=db_path,
    )

    await env.reset()

    # 9. 连载时间推进：每个时间节点只唤醒 schedule[t] 中的 agent
    for t in range(1, num_chapters + 1):
        active_agents = schedule.get(t, [])
        actions = {agent: LLMAction() for agent in active_agents}

        print(
            f"[SYSTEM][{task_name}] Time node {t}/{num_chapters}: "
            f"{len(active_agents)} agents scheduled to act."
        )

        if actions:
            await env.step(actions)

    # 10. 讨论结束后，统一对所有 agent 做 INTERVIEW
    interview_actions: Dict[SocialAgent, ManualAction] = {}
    for user_id, agent in env.agent_graph.get_agents():
        profile = agent.user_info.profile
        prompt = make_chapter_rating_prompt(
            book_name=profile["book_name"],
            story_so_far_summary=profile["story_so_far_summary"],
            chapter_text=profile["chapter_text"],
        )
        interview_actions[agent] = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": prompt},
        )

    print(f"[SYSTEM][{task_name}] Launching interviews for {len(interview_actions)} agents...")
    await env.step(interview_actions)

    # 11. 从 trace 中取出 INTERVIEW 结果，按章节聚合 + 汇总到一个文件
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_id, info
        FROM trace
        WHERE action = ?
        """,
        (ActionType.INTERVIEW.value,),
    )
    rows = cursor.fetchall()
    conn.close()

    chapter_score_lists: Dict[int, List[int]] = {}
    interview_records: List[Dict[str, Any]] = []

    for user_id, info_json in rows:
        info = json.loads(info_json)
        score, reason = parse_interview_score(info)

        agent = agent_by_id.get(user_id)
        if agent is None:
            continue

        profile = agent.user_info.profile
        chapter_number = int(profile["chapter_number"])

        # 复制一份 profile，并删除 story_so_far_summary / chapter_text 节省空间
        profile_copy = dict(profile)
        profile_copy.pop("story_so_far_summary", None)
        profile_copy.pop("chapter_text", None)

        interview_records.append(
            {
                "user_id": user_id,
                "chapter_number": chapter_number,
                "profile": profile_copy,          # 保留 persona + role 等信息
                "raw_info": info,                 # 原始 trace.info（包含 prompt/response）
                "parsed_score": score,
                "parsed_reason": reason,
            }
        )

        if score is None:
            continue

        chapter_score_lists.setdefault(chapter_number, []).append(score)

    # 写出 interview 总表
    interviews_file = result_dir / "interview_records.json"
    with open(interviews_file, "w", encoding="utf-8") as f:
        json.dump(interview_records, f, ensure_ascii=False, indent=2)

    print(f"[SYSTEM][{task_name}] All interview records saved to: {interviews_file}")

    await env.close()

    # 12. 统计每章平均分（对所有使用到的章节；没有评分的就 num_scores=0, mean=0）
    chapter_scores: List[Dict[str, Any]] = []
    for chapter in book_data:
        chapter_number = int(chapter["Number"])
        scores = chapter_score_lists.get(chapter_number, [])
        mean_score = float(sum(scores) / len(scores)) if scores else 0.0

        chapter_scores.append(
            {
                "chapter_number": chapter_number,
                "score_mean": mean_score,
                "num_scores": len(scores),
            }
        )
        print(
            f"[SYSTEM][{task_name}] Chapter {chapter_number}: "
            f"{len(scores)} scores, mean = {mean_score:.3f}"
        )

    scores_file = result_dir / "chapter_scores_serialized.json"
    with open(scores_file, "w", encoding="utf-8") as f:
        json.dump(chapter_scores, f, ensure_ascii=False, indent=2)

    print(f"[SYSTEM][{task_name}] Chapter scores saved to: {scores_file}")

    # 13. 备份数据库到本次实验文件夹下
    backup_db_path = result_dir / "simulation.db"
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_db_path)
        print(f"[SYSTEM][{task_name}] Database backup saved to: {backup_db_path}")
    else:
        print(f"[SYSTEM][{task_name}] WARNING: db file not found for backup: {db_path}")


# ===== 批量实验入口 =====

async def run_batch(
    experiment_configs: List[Dict[str, Any]],
    max_concurrent: int = 2,
) -> None:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_with_semaphore(cfg: Dict[str, Any], idx: int) -> None:
        name = cfg.get("task_name", f"experiment_{idx}")
        async with semaphore:
            print(f"\n====================")
            print(f"[BATCH] Starting experiment {idx + 1}/{len(experiment_configs)}: {name}")
            print(f"====================\n")
            await run_one_experiment(cfg)

    tasks = [
        _run_with_semaphore(cfg, idx)
        for idx, cfg in enumerate(experiment_configs)
    ]
    await asyncio.gather(*tasks)



if __name__ == "__main__":
    # 用法 1：命令行传入一个 JSON 文件路径，里面是 [{...}, {...}] 的配置列表
    #   python test.py batch_configs.json
    #
    # 用法 2：不传参数，跑一轮默认配置，结果写到 result/default_experiment/
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_list = json.load(f)
        # 如果传的是单个 dict，也兼容一下
        if isinstance(cfg_list, dict):
            cfg_list = [cfg_list]
        asyncio.run(run_batch(cfg_list))
    else:
        default_config = {
            "task_name": "default_experiment",
            # 也可以在这里覆盖默认参数，比如：
            # "num_agents": 30,
            # "window_size": 5,
            # "max_chapters": 10,
        }
        asyncio.run(run_one_experiment(default_config))
