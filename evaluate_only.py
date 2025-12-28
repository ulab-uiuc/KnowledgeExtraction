import asyncio
import json
import os
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge

import argparse

async def main():
    # 0. Parse arguments
    parser = argparse.ArgumentParser(description="LLM Knowledge Extraction Evaluation Only")
    parser.add_argument("--query", type=str, default="Linear Algebra", help="Domain to evaluate")
    parser.add_argument("--is_code", action="store_true", help="Whether the domain is code-related")
    args = parser.parse_args()

    # Load configuration
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    
    # Initialize shared components
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    
    # Determine Embedding Strategy
    domain_query = args.query
    is_code_domain = args.is_code
    config_key = "code_embed_config" if is_code_domain else "embed_config"
    embed_config = api_data.get(config_key, api_data.get("embed_config", {}))
    
    if embed_config:
        print(f"Using {'CODE' if is_code_domain else 'TEXT'} embedding model: {embed_config.get('model')} at {embed_config.get('base_url')}")
        # Fix: Use main api_keys if the specific config doesn't have its own
        target_keys = embed_config.get("api_keys", api_keys)
        embed_client_pool = MultiKeyClientPool(
            api_keys=target_keys, 
            base_url=embed_config.get("base_url")
        )
        embed_model = embed_config.get("model")
        # Load custom thresholds if present
        threshold = embed_config.get("threshold", 0.90)
        candidate_threshold = embed_config.get("candidate_threshold", 0.80)
    else:
        embed_client_pool = client_pool
        embed_model = "nvidia/nv-embedcode-7b-v1" if is_code_domain else "nvidia/nv-embed-v1"
        threshold = 0.90
        candidate_threshold = 0.80

    processor = KnowledgeProcessor(
        client_pool=client_pool, 
        embed_client_pool=embed_client_pool,
        embed_model=embed_model,
        threshold=threshold,
        candidate_threshold=candidate_threshold
    )
    judge = DomainJudge(client_pool=client_pool)
    
    # User Input
    query_id = domain_query.lower().replace(" ", "_")
    output_dir = os.path.join("results", query_id)
    
    print(f"=== EVALUATION ONLY MODE ===")
    print(f"Loading raw results from: {output_dir}")

    # 1. Identify existing raw outputs
    raw_files = [f for f in os.listdir(output_dir) if f.endswith("_raw.json")]
    if not raw_files:
        print("No _raw.json files found. Please run main.py first to generate data.")
        return

    pipeline_raw_outputs = {}
    for f in raw_files:
        name = f.replace("_raw.json", "")
        with open(os.path.join(output_dir, f), "r") as f_in:
            pipeline_raw_outputs[name] = json.load(f_in)

    # 2. Build Global Union Set
    print("\nBuilding global deduplicated union set...")
    await processor.build_union_set(output_dir) 
    
    # 3. Global Domain Audit
    print(f"Auditing {len(processor.union_set)} unique knowledge nodes...")
    unique_texts = [node["representative_text"] for node in processor.union_set]
    audit_results = await judge.check_batch(domain_query, unique_texts)
    
    valid_nodes_count = 0
    for node, is_valid in zip(processor.union_set, audit_results):
        node["is_in_domain"] = is_valid
        if is_valid:
            valid_nodes_count += 1
            
    print(f"Audit complete: {valid_nodes_count} valid / {len(processor.union_set)} total nodes.")
    processor.save_union_set(os.path.join(output_dir, "union_set.json"))
    
    # 4. Metric Calculation
    total_valid_nodes = valid_nodes_count
    pipeline_metrics = {}
    
    for name, raw_points in pipeline_raw_outputs.items():
        covered_indices = processor.get_pipeline_coverage(name)
        valid_covered_indices = [i for i in covered_indices if processor.union_set[i]["is_in_domain"]]
        
        recall = len(valid_covered_indices) / total_valid_nodes if total_valid_nodes > 0 else 0
        
        valid_raw_count = 0
        for node_idx in covered_indices:
            node = processor.union_set[node_idx]
            if node["is_in_domain"]:
                for source in node["source_entries"]:
                    if source["pipeline"] == name:
                        valid_raw_count += 1
        
        accuracy = valid_raw_count / len(raw_points) if raw_points else 0
        pipeline_metrics[name] = {"recall": recall, "accuracy": accuracy, "raw_total": len(raw_points)}

    # 5. Final Leaderboard
    print("\n" + "="*75)
    print(f"LEADERBOARD (Saturation Extraction) for Domain: {domain_query}")
    print("="*75)
    print(f"{'Pipeline':30} | {'Recall':10} | {'Accuracy':10} | {'Raw Points'}")
    print("-" * 75)
    
    sorted_keys = sorted(pipeline_metrics.keys(), key=lambda x: pipeline_metrics[x]['recall'], reverse=True)
    for name in sorted_keys:
        m = pipeline_metrics[name]
        print(f"{name:30} | {m['recall']:7.2%} | {m['accuracy']:7.2%} | {m['raw_total']}")
    print("="*75)
    print(f"Global Pseudo Ground Truth Size: {total_valid_nodes} unique points.")

if __name__ == "__main__":
    asyncio.run(main())

