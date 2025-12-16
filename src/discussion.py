from __future__ import annotations
from typing import List, Optional, Dict
from .schemas import DiscussionMessage
from .json_utils import extract_json_object
from .prompts import discussion_message_prompt


def run_discussion(
    llm,
    meta: Dict,
    agent_uuid: str,
    persona_text: Optional[str],
    agent_stance: str,
    global_messages: List[str],
    rounds: int,
    window: int,
    llm_cfg: Dict,
) -> List[DiscussionMessage]:
    """
    让某个 agent 参与若干轮讨论。
    global_messages 是所有人共享的讨论记录（纯字符串列表即可）。
    返回该 agent 发出的消息（带 round 信息）。
    """
    out: List[DiscussionMessage] = []

    for r in range(1, rounds + 1):
        tail = global_messages[-window:]
        p = discussion_message_prompt(meta=meta, persona_text=persona_text, agent_stance=agent_stance, discussion_tail=tail)
        text = llm.chat_json(
            system=p["system"],
            user=p["user"],
            temperature=llm_cfg["temperature"],
            top_p=llm_cfg["top_p"],
            max_tokens=llm_cfg["max_tokens"],
            # 如果你的平台支持 json_object，可在这里打开：
            response_format=None,
        )
        obj = extract_json_object(text)
        msg = str(obj.get("message", "")).strip()
        if not msg:
            msg = "(no message)"

        global_messages.append(msg)
        out.append(DiscussionMessage(agent_uuid=agent_uuid, round=r, message=msg))

    return out
