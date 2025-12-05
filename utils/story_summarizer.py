from typing import Dict, Any, List
from pathlib import Path
import asyncio
import concurrent.futures
import json

from llm_base import LLMBase


# ========== 全局配置（你只需要改这里） ==========

# 输入根目录: ../data/books/{bookname}/{bookname}_{N}.json
BOOKS_ROOT = Path("../data/books")

# 所有输出统一放在这个目录
OUTPUT_ROOT = Path("../data/books_summaries")

# 文件名中的 {N}，例如 {bookname}_3000.json
N = 3000

# 同时并发处理多少本书（线程数）
MAX_WORKERS = 30


# 书单：逐本并发处理
CANONICAL_TITLES = [
    # 高分 15 本
    "Fourth Wing (The Empyrean, #1)",
    "Iron Flame (The Empyrean, #2)",
    "Divine Rivals (Letters of Enchantment, #1)",
    "Yours Truly (Part of Your World, #2)",
    "The Frozen River",
    "Knockemout 3 Things We Left Behind",
    "The Covenant of Water",
    "The Women",
    "Funny Story",
    "Just for the Summer",
    "The God of the Woods",
    "All the Colors of the Dark",
    "Quicksilver (Fae & Alchemy, #1)",
    "James",
    "Onyx Storm (The Empyrean, #3)",

    # 有争议 / 低分 15 本
    "Yellowface",
    "The Pumpkin Spice Café (Dream Harbor, #1)",
    "The Husbands",
    "Society of Lies",
    "The Blue Hour",
    "The Widow's Husband's Secret Lie",
    "Incidents Around the House",
    "A Tempest of Tea (Blood and Tea, #1)",
    "Only If You're Lucky",
    "The Night We Lost Him",
    "A Novel Love Story",
    "The Christmas Tree Farm (Dream Harbor, #3)",
    "No One Can Know",
    "I Hope This Finds You Well",
    "The Striker (Gods of the Game, #1)",
]


class StorySummarizer(LLMBase):

    SUMMARY_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "chapter_summary": {
                "type": "string",
                "description": "Summary of the current chapter only."
            },
            "story_so_far_summary": {
                "type": "string",
                "description": "Updated summary of the whole story so far (all previous chapters plus the current one)."
            }
        },
        "required": ["chapter_summary", "story_so_far_summary"],
        "additionalProperties": False
    }

    def get_summary(self, previous_story_summary: str, chapter_text: str) -> Dict[str, str]:
        """
        previous_story_summary: 已有的“截至上一章的全书总结”（可以为空字符串）。
        chapter_text: 当前章节的全文。
        返回:
          - chapter_summary: 本节全文总结
          - story_so_far_summary: 将 previous_story_summary 与本节内容合并后的“截至本节的全书总结”
        """
        system_prompt = (
            "You are an editor skilled at summarizing long-form fiction.\n"
            "Given:\n"
            "1) A running summary of all previous chapters (may be empty if this is the first chapter).\n"
            "2) The full text of the current chapter.\n\n"
            "Your tasks:\n"
            "1) Provide a concise summary of ONLY the current chapter (max 2000 characters).\n"
            "2) Provide an UPDATED concise summary of the whole story so far, i.e., merge the\n"
            "   previous story summary with what happens in this chapter (max 3000 characters).\n\n"
            "Notes:\n"
            "- If there is no previous summary, treat this as the first chapter and base the\n"
            "  story-so-far summary only on the current chapter.\n"
            "- If the current chapter text is empty, return empty strings for both summaries.\n"
        )

        if not chapter_text.strip():
            # 当前章节为空时，不新增总结，只把已有 story_so_far_summary 原样返回
            return {
                "chapter_summary": "",
                "story_so_far_summary": previous_story_summary.strip(),
                "chapter_text": chapter_text,
            }

        user_prompt = (
            f"[Story-so-far summary BEFORE this chapter]:\n"
            f"{previous_story_summary.strip() if previous_story_summary.strip() else 'No previous summary; this is the first chapter.'}\n\n"
            f"[Full text of CURRENT chapter]:\n{chapter_text}\n\n"
            f"Generate both the 'chapter_summary' and 'story_so_far_summary' according to the instructions."
        )

        result = self.call_structured_json(
            model=self.STRONG_TEXT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=self.SUMMARY_SCHEMA,
            temperature=0.4,
        )

        return {
            "chapter_summary": result.get("chapter_summary", "").strip(),
            "story_so_far_summary": result.get("story_so_far_summary", "").strip(),
            "chapter_text": chapter_text,
        }


# ========= 工具函数 =========

def safe_write_json(path: Path, data: Any) -> None:
    """
    原子写 JSON：先写到 .tmp，再 rename，避免中途崩溃导致文件半截。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def process_one_book(book_title: str) -> None:
    """
    单本书的同步处理逻辑：顺序遍历章节，但支持断点续跑。

    输入路径格式:
      ../data/books/{bookname}/{bookname}_{N}.json

    输出路径统一到:
      ../data/books_summaries/{bookname}_{N}_chapter_summaries.json
    """
    input_path = BOOKS_ROOT / book_title / f"{book_title}_{N}.json"
    output_path = OUTPUT_ROOT / f"{book_title}_{N}_chapter_summaries.json"

    if not input_path.exists():
        print(f"[{book_title}] 输入文件不存在: {input_path}")
        return

    print(f"[{book_title}] 读取章节: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        chapters: List[Dict[str, Any]] = json.load(f)

    # ---- 断点续跑：检测已有输出 ----
    processed_results: List[Dict[str, Any]] = []
    story_so_far_summary: str = ""
    start_index: int = 0

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                processed_results = json.load(f)

            if processed_results:
                processed_count = len(processed_results)
                start_index = processed_count
                story_so_far_summary = processed_results[-1].get(
                    "story_so_far_summary", ""
                )

                print(
                    f"[{book_title}] 检测到已有进度: 已完成 "
                    f"{processed_count}/{len(chapters)} 章，从第 {start_index + 1} 章继续。"
                )

                if processed_count >= len(chapters):
                    print(f"[{book_title}] 已全部处理完成，跳过。")
                    return
        except Exception as e:
            print(
                f"[{book_title}] 读取已有输出文件失败，将从头重跑该书: {e}"
            )
            processed_results = []
            story_so_far_summary = ""
            start_index = 0
    else:
        print(f"[{book_title}] 没有检测到历史输出，从头开始。")

    summarizer = StorySummarizer()

    # ---- 顺序处理该书的剩余章节 ----
    for idx in range(start_index, len(chapters)):
        chapter = chapters[idx]
        number = chapter.get("Number", idx + 1)
        text = chapter.get("text", "")

        print(f"[{book_title}] Summarizing chapter/segment: {number} (index {idx})")

        try:
            summary = summarizer.get_summary(story_so_far_summary, text)
        except Exception as e:
            # 网络问题 / API 报错等：保留当前已完成进度，抛异常给上层
            print(
                f"[{book_title}] 在处理章节 {number} 时出错，"
                f"已保留之前的进度: {e}"
            )
            safe_write_json(output_path, processed_results)
            raise

        story_so_far_summary = summary["story_so_far_summary"]

        result = {
            "Number": number,
            "chapter_text": summary["chapter_text"],
            "chapter_summary": summary["chapter_summary"],
            "story_so_far_summary": story_so_far_summary,
        }
        processed_results.append(result)

        # 每处理一章就立刻落盘，保证断点续跑粒度到“章”
        safe_write_json(output_path, processed_results)

    print(f"[{book_title}] 完成。结果已写入: {output_path}")


def summarize_book_sync(book_title: str) -> None:
    """
    给 ThreadPoolExecutor 调用的同步函数（单本书）。
    """
    print(f"=== 开始处理: {book_title} ===")
    process_one_book(book_title)
    print(f"=== 结束处理: {book_title} ===")


# ========= 异步总控 =========

async def main_async() -> None:
    """
    异步总控：
    - 使用 asyncio + ThreadPoolExecutor
    - 每本书在一个线程中顺序 summarise
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("配置：")
    print(f"  BOOKS_ROOT = {BOOKS_ROOT}")
    print(f"  OUTPUT_ROOT = {OUTPUT_ROOT}")
    print(f"  N = {N}")
    print(f"  MAX_WORKERS = {MAX_WORKERS}")
    print(f"  书本数量 = {len(CANONICAL_TITLES)}")
    print("开始并发处理所有书籍...\n")

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [
            loop.run_in_executor(executor, summarize_book_sync, title)
            for title in CANONICAL_TITLES
        ]

        # return_exceptions=True 保证单本书失败不影响其他任务继续
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for title, res in zip(CANONICAL_TITLES, results):
            if isinstance(res, Exception):
                print(f"[ERROR] 书籍 '{title}' 处理失败: {res}")
            else:
                print(f"[OK] 书籍 '{title}' 处理完成。")


if __name__ == "__main__":
    # 直接运行此文件即可：python story_summarizer.py
    asyncio.run(main_async())
