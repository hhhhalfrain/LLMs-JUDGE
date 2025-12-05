# experiments.py
import asyncio
from experiment import run_batch

# 这里硬编码多个实验配置
EXPERIMENTS = [
    # {
    #     "task_name": "win5_agents30",
    #     "book_path": "data/books/Things Have Gotten Worse Since We Last Spoke and Other -- Eric LaRocca_3000_chapter_summaries.json",
    #     "book_name": "Things Have Gotten Worse Since We Last Spoke and Other -- Eric LaRocca",
    #     "persona_path": "data/reddit/personas_sample_100.json",
    #     "num_agents": 10,
    #     "window_size": 5,
    #     "max_chapters": None,
    #     "random_seed": 42,
    # },
]


async def main():
    CANONICAL_TITLES = [
        "Fourth Wing (The Empyrean, #1)",
        # "Iron Flame (The Empyrean, #2)",
        # "Divine Rivals (Letters of Enchantment, #1)",
        "Yours Truly (Part of Your World, #2)",
        # "The Frozen River",
        # "Knockemout 3 Things We Left Behind",
        "The Covenant of Water",
        "The Women",
        "Funny Story",
        "Just for the Summer",
        "The God of the Woods",
        "All the Colors of the Dark",
        "Quicksilver (Fae & Alchemy, #1)",
        "James",
        # "Onyx Storm (The Empyrean, #3)",

        "Yellowface",
        # "The Pumpkin Spice Café (Dream Harbor, #1)",
        "The Husbands",
        # "Society of Lies",
        # "The Blue Hour",
        "The Widow's Husband's Secret Lie",
        "Incidents Around the House",
        "A Tempest of Tea (Blood and Tea, #1)",
        "Only If You're Lucky",
        "The Night We Lost Him",
        "A Novel Love Story",
        # "The Christmas Tree Farm (Dream Harbor, #3)",
        # "No One Can Know",
        "I Hope This Finds You Well",
        "The Striker (Gods of the Game, #1)",
    ]
    for canonical_title in CANONICAL_TITLES:
        experiment = {
            "task_name": canonical_title,
            "book_path": f"data/books_summaries/{canonical_title}_3000_chapter_summaries.json",
            "book_name": canonical_title,
            "persona_path": "data/reddit/personas_sample_100.json",
            "num_agents": 100,
            "window_size": 5,
            "max_chapters": None,
            "random_seed": 42,
        }
        EXPERIMENTS.append(experiment)
    await run_batch(EXPERIMENTS, max_concurrent=30)


if __name__ == "__main__":
    asyncio.run(main())
