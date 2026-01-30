import asyncio
from openai import AsyncOpenAI
import itertools
import time
import sys

class MultiKeyClientPool:
    def __init__(self, api_keys, base_url="https://integrate.api.nvidia.com/v1"):
        self.clients = [AsyncOpenAI(base_url=base_url, api_key=k) for k in api_keys]
        self._cycle = itertools.cycle(self.clients)
        self.bad_clients = set()
        self.client_to_idx = {c: i for i, c in enumerate(self.clients)}

    def get(self):
        # 优先返回没被标记的，如果全挂了，就返回第一个试试
        for _ in range(len(self.clients)):
            client = next(self._cycle)
            if client not in self.bad_clients:
                return client
        return self.clients[0]

    def mark_bad(self, client):
        if client not in self.bad_clients:
            idx = self.client_to_idx.get(client, "?")
            self.bad_clients.add(client)
            print(f"      [ClientPool] Key [{idx:02}] marked temporary BAD. Active: {len(self.clients)-len(self.bad_clients)}/{len(self.clients)}")

async def safe_ask(client_pool, model, messages, temperature=1.0, top_p=0.9, max_tokens=4096):
    # evaluate 使用独立的重试逻辑
    for attempt in range(10):
        client = client_pool.get()
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=45
            )
        except Exception as e:
            err_str = str(e).upper()
            if "403" in err_str or "401" in err_str:
                client_pool.mark_bad(client)
            await asyncio.sleep(1.0)
            continue
    raise RuntimeError(f"safe_ask failed")

async def safe_embed(client_pool, model, messages, **kwargs):
    # Embedding 同样增加并发锁思想
    last_err = None
    for attempt in range(5):
        client = client_pool.get()
        try:
            # Increased timeout for heavy local loads
            return await client.embeddings.create(model=model, input=messages, timeout=60, **kwargs)
        except Exception as e:
            last_err = e
            await asyncio.sleep(0.5 * (attempt + 1))
            continue
    raise RuntimeError(f"Embedding failed: {last_err}")
