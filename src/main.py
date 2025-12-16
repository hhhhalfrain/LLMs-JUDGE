from __future__ import annotations
import argparse

from dotenv import load_dotenv

from .config import load_config, get_api_key
from .llm_client import LLMClient
from .runner import run_all_books


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    api_key = get_api_key(cfg)

    llm = LLMClient(
        base_url=cfg.llm.base_url,
        api_key=api_key,
        model=cfg.llm.model,
        timeout_sec=cfg.llm.timeout_sec,
    )

    run_all_books(cfg, llm)


if __name__ == "__main__":
    main()
