import asyncio
import json
import random

from datasets import load_dataset
from assistant import Assistant


def get_name_sync(persona: str):
    """
    原来的同步版本：调用 Assistant().ask
    放在线程池里跑，用于 I/O 并发。
    """
    a = Assistant()
    system = "You are a helpful assistant that extracts names from personas."
    prompt = f"Extract the full name and take a username from the following persona:\n\n{persona}"
    schema = {
        "type": "object",
        "properties": {
            "full_name": {
                "type": "string",
                "description": "The full name of the person."
            },
            "username": {
                "type": "string",
                "description": "A username derived from the full name."
            }
        },
        "required": ["full_name", "username"],
        "additionalProperties": False
    }
    result = a.ask(prompt=prompt, system=system, schema=schema)
    return result["full_name"], result["username"]


def process_one_persona_sync(data):
    """
    单条 persona 的同步处理逻辑。
    方便丢进 asyncio.to_thread 中执行。
    """
    persona = {
        "uuid": data["uuid"],
        "persona": data["persona"],
        "sex": data["sex"],
        "age": data["age"],
        "education_level": data["education_level"],
        "occupation": data["occupation"],
    }

    full_name, username = get_name_sync(data["persona"])
    persona["name"] = full_name
    persona["username"] = username

    return persona


async def process_one_persona(data, sem: asyncio.Semaphore):
    """
    异步包装：
    - 用 Semaphore 控制同时并发的数量（防止打爆后端 / 本机）
    - 用 asyncio.to_thread 把同步 CPU/I/O 丢到线程池中
    """
    async with sem:
        # 把同步函数放到默认线程池执行
        return await asyncio.to_thread(process_one_persona_sync, data)


async def main():
    # 加载数据集
    ds = load_dataset("nvidia/Nemotron-Personas-USA")["train"]

    # 随机抽样 200 条
    idx = random.sample(range(len(ds)), 200)
    samples = ds.select(idx)
    data_list = samples.to_list()

    # 排除其中age<10的数据
    data_list = [data for data in data_list if data["age"] is not None and data["age"] >= 10]

    data_list= data_list[:100]  # 只取前100条

    # 控制最大并发量，例如 8 个并发请求
    sem = asyncio.Semaphore(32)

    # 为每条数据创建一个异步任务
    tasks = [
        asyncio.create_task(process_one_persona(data, sem))
        for data in data_list
    ]

    # 并发执行并收集所有结果
    persona_list = await asyncio.gather(*tasks)

    # 写出到 json 文件
    with open("../data/persona/personas_sample.json", "w", encoding="utf-8") as f:
        json.dump(persona_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
