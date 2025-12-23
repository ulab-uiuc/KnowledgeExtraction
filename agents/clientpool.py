from openai import AsyncOpenAI
import itertools

class MultiKeyClientPool:
    def __init__(self, api_keys, base_url="https://integrate.api.nvidia.com/v1"):
        self.clients = [
            AsyncOpenAI(base_url=base_url, api_key=k)
            for k in api_keys
        ]
        self._cycle = itertools.cycle(self.clients)

    def get(self):
        return next(self._cycle)

# # 使用
# api_keys = [
#     "key1",
#     "key2",
#     "key3",
# ]

# client_pool = MultiKeyClientPool(api_keys)


# async def ask(client, model, messages, temperature, top_p,max_tokens):
#     return await client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         top_p=top_p,
#         max_tokens=max_tokens,
#     )
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

async def safe_embed(client_pool, model, messages, temperature=1.0, top_p=0.95, max_tokens=16384):
    for _ in range(len(client_pool.clients)):
        client = client_pool.get()
        try:
            return await client.embeddings.create(
                input=messages,
                model=model,
                encoding_format="float",
                extra_body={"input_type": "query", "truncate": "NONE"}
            )
        except Exception as e:
            print(f"Key failed, try next. error={e}")
    raise RuntimeError("All API keys failed")

