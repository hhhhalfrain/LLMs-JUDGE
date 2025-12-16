from __future__ import annotations
from pydantic import BaseModel, Field
import os
import yaml


class LLMConfig(BaseModel):
    base_url: str
    api_key_env: str = "DASHSCOPE_API_KEY"
    model: str
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 800
    timeout_sec: int = 120


class PathsConfig(BaseModel):
    personas_json: str
    books_root: str
    output_root: str


class ExperimentConfig(BaseModel):
    method: str = Field(description="aggregation|incremental|summary_based")
    use_persona: bool = True
    use_discussion: bool = False
    use_interest_filter: bool = True

    discussion_rounds: int = 2
    discussion_window: int = 10

    n_agents: int = 8
    score_decimals: int = 1
    discussion_affects_score: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig
    paths: PathsConfig
    experiment: ExperimentConfig


def load_config(path: str) -> AppConfig:
    """读取 YAML 配置并解析为强类型对象（避免到处传 dict）"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def get_api_key(cfg: AppConfig) -> str:
    """从环境变量获取 DashScope API Key"""
    key = os.getenv(cfg.llm.api_key_env, "")
    if not key:
        raise RuntimeError(
            f"环境变量 {cfg.llm.api_key_env} 未设置。请先 export {cfg.llm.api_key_env}=..."
        )
    return key
