from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Persona(BaseModel):
    uuid: str
    persona: str
    sex: Optional[str] = None
    age: Optional[int] = None
    education_level: Optional[str] = None
    occupation: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None


class BookMeta(BaseModel):
    book_name: str
    intro: str


class Chapter(BaseModel):
    Number: int
    text: str


class InterestDecision(BaseModel):
    keep: bool
    interest_score: float = Field(ge=0.0, le=100.0)
    reason: str


class ChapterEval(BaseModel):
    chapter_index: int
    score: float
    plot_summary: str
    comment: str


class IncrementalStep(BaseModel):
    chapter_index: int
    score: float
    summary: str
    review: str


class DiscussionMessage(BaseModel):
    agent_uuid: str
    round: int
    message: str


class AgentResult(BaseModel):
    agent_uuid: str
    kept: bool
    filtered_reason: Optional[str] = None

    interest: Optional[InterestDecision] = None

    # 三种方法分别填其中一种轨迹
    chapter_evals: Optional[List[ChapterEval]] = None
    incremental_steps: Optional[List[IncrementalStep]] = None
    global_summary: Optional[Dict[str, Any]] = None

    discussion: List[DiscussionMessage] = Field(default_factory=list)

    pre_discussion_score: Optional[float] = None
    post_discussion_score: Optional[float] = None
    final_review: Optional[str] = None


class RunOutput(BaseModel):
    book_name: str
    metadata: BookMeta
    config: Dict[str, Any]
    agents: List[AgentResult]

    aggregate: Dict[str, Any]
