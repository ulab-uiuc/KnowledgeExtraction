import asyncio
import json
import os
from openai import AsyncOpenAI

async def test_key(name, key, model, base_url, is_embed=False, embed_model=None, embed_url=None):
    if is_embed:
        client = AsyncOpenAI(api_key=key, base_url=embed_url)
        target_model = embed_model
    else:
        client = AsyncOpenAI(api_key=key, base_url=base_url)
        target_model = model
        
    try:
        if is_embed:
            await client.embeddings.create(
                input=["health check"],
                model=target_model
            )
        else:
            await client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
        return True, None
    except Exception as e:
        return False, str(e)

async def main():
    if not os.path.exists("api.json"):
        print("api.json not found")
        return
    
    with open("api.json") as f:
        data = json.load(f)
    
    keys = data["api_keys"]
    gen_model = "meta/llama-3.1-405b-instruct"
    gen_url = "https://integrate.api.nvidia.com/v1"
    
    embed_config = data.get("embed_config", {})
    embed_model = embed_config.get("model", "Qwen/Qwen3-Embedding-8B")
    embed_url = embed_config.get("base_url", "http://localhost:30000/v1")
    
    print(f"Testing {len(keys)} keys...")
    print(f"Gen Model: {gen_model} via {gen_url}")
    print(f"Emb Model: {embed_model} via {embed_url}\n")
    
    for i, key in enumerate(keys):
        # Test Gen
        gen_ok, gen_err = await test_key(f"Key_{i}", key, gen_model, gen_url, is_embed=False)
        # Test Embed
        emb_ok, emb_err = await test_key(f"Key_{i}", key, gen_model, gen_url, is_embed=True, embed_model=embed_model, embed_url=embed_url)
        
        if gen_ok and emb_ok:
            status = "PASSED"
        elif not gen_ok and not emb_ok:
            status = "FAILED BOTH"
        elif not gen_ok:
            status = "FAILED GEN"
        else:
            status = "FAILED EMB"
            
        print(f"[{i:02d}] {status}")
        if not gen_ok: print(f"    - Gen Error: {gen_err}")
        if not emb_ok: print(f"    - Emb Error: {emb_err}")

if __name__ == "__main__":
    asyncio.run(main())
