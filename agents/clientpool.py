from openai import AsyncOpenAI
import itertools

class MultiKeyClientPool:
    def __init__(self, api_keys, base_url="https://integrate.api.nvidia.com/v1"):
        if not api_keys:
            # Handle local case where keys might not be needed
            self.clients = [AsyncOpenAI(base_url=base_url, api_key="none")]
        else:
            self.clients = [
                AsyncOpenAI(base_url=base_url, api_key=k)
                for k in api_keys
            ]
        self._cycle = itertools.cycle(self.clients)

    def get(self):
        return next(self._cycle)

async def safe_ask(client_pool, model, messages, temperature=1.0, top_p=0.95, max_tokens=16384):
    for _ in range(len(client_pool.clients)):
        client = client_pool.get()
        try:
            return await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Key failed, try next. error={e}")
    raise RuntimeError("All API keys failed")

async def safe_embed(client_pool, model, messages, **kwargs):
    for _ in range(len(client_pool.clients)):
        client = client_pool.get()
        try:
            # SGLang/Standard OpenAI don't need extra_body for Nvidia
            # But we allow passing extra parameters via kwargs if needed
            return await client.embeddings.create(
                input=messages,
                model=model,
                encoding_format="float",
                **kwargs
            )
        except Exception as e:
            print(f"Embedding request failed, try next. error={e}")
    raise RuntimeError("All embedding clients failed")
