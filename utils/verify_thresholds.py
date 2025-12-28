import asyncio
import json
import os
import numpy as np
import random
from agents.clientpool import MultiKeyClientPool, safe_ask
from core.processor import KnowledgeProcessor

import argparse

async def check_is_same(client_pool, text1, text2):
    prompt = f"""
    Do these two bullet points describe the same fundamental concept or property?
    
    Point A: {text1}
    Point B: {text2}
    
    Criteria for 'YES':
    1. They are semantically equivalent despite different phrasing.
    2. One is a slightly more detailed version of the same definition.
    3. They refer to the same theorem, property, or code behavior.
    
    Criteria for 'NO':
    1. They describe different concepts.
    2. They describe different properties or behaviors of the same object.
    3. One is a general category and the other is a specific sub-topic.
    
    Answer only 'YES' or 'NO'.
    """
    try:
        # Using a much larger model for better reasoning
        response = await safe_ask(
            client_pool,
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        ans = response.choices[0].message.content.strip().upper()
        return "YES" in ans and "NO" not in ans
    except:
        return False

async def main():
    # 0. Parse arguments
    parser = argparse.ArgumentParser(description="Verify Embedding Thresholds")
    parser.add_argument("--query", type=str, default="Linear Algebra", help="Domain to verify")
    parser.add_argument("--is_code", action="store_true", help="Whether the domain is code-related")
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists("api.json"):
        print("错误: 未找到 api.json 文件")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    
    # 3. Determine Embedding Strategy
    domain_query = args.query
    is_code_domain = args.is_code
    config_key = "code_embed_config" if is_code_domain else "embed_config"
    embed_config = api_data.get(config_key, api_data.get("embed_config", {}))
    
    if not embed_config:
        print(f"错误: api.json 中未发现 {config_key} 配置")
        return

    client_pool = MultiKeyClientPool(api_keys=api_data["api_keys"])
    target_keys = embed_config.get("api_keys", api_data["api_keys"])
    embed_pool = MultiKeyClientPool(api_keys=target_keys, base_url=embed_config.get("base_url"))
    processor = KnowledgeProcessor(None, embed_client_pool=embed_pool, embed_model=embed_config.get("model"))

    # 加载数据
    query_id = domain_query.lower().replace(" ", "_")
    output_dir = os.path.join("results", query_id)
    all_points = set()
    for f in os.listdir(output_dir):
        if f.endswith("_raw.json"):
            with open(os.path.join(output_dir, f), "r") as fin:
                all_points.update(json.load(fin))
    
    unique_points = list(all_points)
    print(f"Total unique points: {len(unique_points)}")
    embeddings = await processor.get_embeddings(unique_points)
    
    ranges = [
        (0.95, 1.00),
        (0.90, 0.95),
        (0.85, 0.90),
        (0.80, 0.85),
        (0.70, 0.80)
    ]
    
    results = {}
    
    for low, high in ranges:
        print(f"\nScanning range [{low:.2f} - {high:.2f}]...")
        pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = processor.cosine_similarity(embeddings[i], embeddings[j])
                if low <= sim < high:
                    pairs.append((unique_points[i], unique_points[j], sim))
        
        sample_size = min(len(pairs), 10)
        samples = random.sample(pairs, sample_size)
        
        yes_count = 0
        print(f"Testing {sample_size} samples...")
        for t1, t2, s in samples:
            is_same = await check_is_same(client_pool, t1, t2)
            if is_same: yes_count += 1
            status = " [MATCH]" if is_same else " [DIFF ]"
            print(f"  {s:.4f} | {t1[:40]}... vs {t2[:40]}... -> {status}")
        
        results[f"{low}-{high}"] = (yes_count, sample_size)

    print("\n" + "="*50)
    print("THRESHOLD VERIFICATION REPORT")
    print("="*50)
    print(f"{'Range':15} | {'Mergable Rate':15} | {'Suggestion'}")
    print("-" * 50)
    for r, (y, s) in results.items():
        rate = y / s if s > 0 else 0
        suggestion = "Auto-Merge" if rate > 0.8 else ("Grey Zone" if rate > 0.2 else "Distinct")
        print(f"{r:15} | {rate:14.1%} | {suggestion}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())

