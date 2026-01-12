import asyncio
import json
import os
import numpy as np
from core.processor import KnowledgeProcessor
from agents.clientpool import MultiKeyClientPool

async def verify_thresholds(results_dir):
    # 1. Load configuration
    if not os.path.exists("api.json"):
        print("Error: api.json not found")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    
    api_keys = api_data["api_keys"]
    
    # Use code embedding config if "code" is in path
    is_code = "code" in results_dir.lower()
    config_key = "code_embed_config" if is_code else "embed_config"
    embed_config = api_data.get(config_key, api_data.get("embed_config", {}))
    
    if not embed_config:
        print(f"Error: No {config_key} configuration found in api.json")
        return

    # Initialize processor
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    embed_client_pool = MultiKeyClientPool(
        api_keys=embed_config.get("api_keys", api_keys),
        base_url=embed_config.get("base_url")
    )
    
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=embed_client_pool,
        embed_model=embed_config.get("model")
    )

    # 2. Load data
    union_set_path = os.path.join(results_dir, "union_set.json")
    if not os.path.exists(union_set_path):
        print(f"Error: {union_set_path} not found")
        return
        
    with open(union_set_path, "r") as f:
        union_set = json.load(f)

    # 3. Analyze internal similarities of clusters
    print(f"\nAnalyzing {len(union_set)} knowledge nodes from {results_dir}...")
    
    low_sim_merges = []
    
    for node in union_set:
        sources = node.get("detailed_sources", [])
        if len(sources) < 2:
            continue
            
        for s in sources:
            sim = s.get("similarity", 1.0)
            if 0 < sim < 0.85: # Threshold for investigation
                low_sim_merges.append({
                    "rep": node["representative_text"],
                    "source": s["text"],
                    "sim": sim,
                    "pipeline": s["pipeline"]
                })

    if low_sim_merges:
        print(f"\nFound {len(low_sim_merges)} merges with similarity < 0.85:")
        # Sort by similarity ascending
        low_sim_merges.sort(key=lambda x: x["sim"])
        for m in low_sim_merges[:10]: # Show top 10 most questionable merges
            print(f"\nSimilarity: {m['sim']:.4f} (Pipeline: {m['pipeline']})")
            print(f"  Node:   {m['rep'][:100]}...")
            print(f"  Merged: {m['source'][:100]}...")
    else:
        print("\nAll merges look high-confidence (>0.85 similarity).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Results directory to verify")
    args = parser.parse_args()
    
    asyncio.run(verify_thresholds(args.dir))
