from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any, Optional, Iterable, Tuple
import json
import time
import re
from openai import AsyncOpenAI
from .clientpool import MultiKeyClientPool, safe_ask
import asyncio

SYSTEM_PROMPT = """
You are a Answer Agent. Your goal is to provide accurate and informative answers to user questions based on your knowledge and reasoning abilities.
"""

class GenAgent:
    def __init__(
        self,
        api_key: List[str],
        model: str = "meta/llama-3.2-3b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ):
        self.client = MultiKeyClientPool(
            api_keys=api_key,
            base_url=base_url
        )
        self.api_key = api_key
        self.system_prompt = SYSTEM_PROMPT
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    async def generate(self, problem: str) -> str:
        completion = await safe_ask(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        return completion.choices[0].message.content

if __name__ == "__main__":
    import random
    with open("api.json") as f:
        api_data = json.load(f)
    agent = GenAgent(api_key=api_data["api_keys"])
    question = "What are the latest advancements in artificial intelligence in 2024?"
    answer = asyncio.run(agent.generate(question))
    print("Answer:")
    print(answer)