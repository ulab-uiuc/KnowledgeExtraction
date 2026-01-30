import asyncio
from openai import AsyncOpenAI
from .clientpool import MultiKeyClientPool
import sys

SYSTEM_PROMPT = "You are a Answer Agent. Provide accurate and informative answers."

class GenAgent:
    _global_semaphore = asyncio.Semaphore(200) 

    def __init__(self, api_key, model="meta/llama-3.1-405b-instruct", base_url="https://integrate.api.nvidia.com/v1", **kwargs):
        keys = [api_key] if isinstance(api_key, str) else api_key
        self.client_pool = MultiKeyClientPool(api_keys=keys, base_url=base_url)
        self.model = model
        self.total_tokens = 0
        self.system_prompt = SYSTEM_PROMPT

    async def generate(self, problem: str) -> str:
        async with self._global_semaphore:
            for attempt in range(50):
                client = self.client_pool.get()
                if not client:
                    raise RuntimeError("No available clients in pool")
                try:
                    completion = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": problem}],
                        temperature=0.7,
                        max_tokens=4096,
                        timeout=600
                    )
                    self.total_tokens += getattr(completion.usage, 'total_tokens', 0)
                    return completion.choices[0].message.content
                except Exception as e:
                    err_str = str(e).upper()
                    if "403" in err_str or "401" in err_str:
                        self.client_pool.mark_bad(client)
                        continue
                    if "429" in err_str:
                        import random
                        await asyncio.sleep(2.0 + random.random() * 2.0)
                        continue
                    await asyncio.sleep(1.0)
                    continue
            raise RuntimeError(f"GenAgent exhausted 50 retries for {self.model}")
