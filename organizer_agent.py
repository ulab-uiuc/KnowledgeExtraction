from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict, Any, Optional, Iterable, Tuple
import json
import time
import re
from openai import OpenAI

"currently it's useless, but we may find it useful later"

SYSTEM_PROMPT = """

"""

class OrganizerAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "meta/llama-3.2-3b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 8192,
    ):
        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = api_key
        )
        self.api_key = api_key
        self.system_prompt = SYSTEM_PROMPT
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate(self, problem: str):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem}
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            n=1,
            max_tokens=self.max_tokens,
            stream=False
        )
        return completion.choices[0].message.content

if __name__ == "__main__":
    import random
    with open("api.json") as f:
        api_data = json.load(f)
    api_key = random.choice(api_data["api_keys"])
    agent = AnswerAgent(api_key=api_key)
    question = "What are the latest advancements in artificial intelligence in 2024?"
    answer = agent.generate(question)
    print("Answer:")
    print(answer)