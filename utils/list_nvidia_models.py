import asyncio
import json
import os
from openai import AsyncOpenAI

async def main():
    if not os.path.exists("api.json"):
        print("Error: api.json not found")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    
    api_keys = api_data.get("api_keys", [])
    if not api_keys:
        print("Error: No API keys found in api.json")
        return

    # Use the first key to list models
    client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_keys[0]
    )

    try:
        print("Fetching available models from NVIDIA API...")
        response = await client.models.list()
        
        # Sort and filter for chat/instruct models
        models = sorted([m.id for m in response.data])
        
        print("\nAvailable Models:")
        print("-" * 60)
        for model in models:
            # Highlight some popular large models
            if "70b" in model.lower() or "405b" in model.lower() or "large" in model.lower():
                print(f"* {model} (Large)")
            else:
                print(f"  {model}")
        print("-" * 60)
        print(f"Total models found: {len(models)}")
        
    except Exception as e:
        print(f"Failed to fetch models: {e}")

if __name__ == "__main__":
    asyncio.run(main())


