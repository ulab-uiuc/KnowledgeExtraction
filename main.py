import asyncio
import json
import os
import importlib
import inspect
from agents.clientpool import MultiKeyClientPool
from agents.call_agent import GenAgent
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge
from core.evaluator import Evaluator
from core.cleaner import KnowledgeCleaner
from pipelines.base import BasePipeline

import argparse

async def main():
    # 0. Parse arguments
    parser = argparse.ArgumentParser(description="LLM Knowledge Extraction Evaluation Framework")
    parser.add_argument("--query", type=str, default="Linear Algebra", help="Domain to extract knowledge from")
    parser.add_argument("--is_code", action="store_true", help="Whether the domain is code-related")
    args = parser.parse_args()

    # 1. User Input
    domain_query = args.query
    is_code_domain = args.is_code
    query_id = domain_query.lower().replace(" ", "_")
    output_dir = os.path.join("results", query_id)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load configuration
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    
    # Initialize shared components
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    
    # 3. Determine Embedding Strategy
    # If it's a code domain, look for code-specific config, otherwise use default
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
        # Fallback to Nvidia default
        embed_client_pool = client_pool
        embed_model = "nvidia/nv-embedcode-7b-v1" if is_code_domain else "nvidia/nv-embed-v1"
        threshold = 0.90
        candidate_threshold = 0.80

    gen_agent = GenAgent(api_key=api_keys)
    processor = KnowledgeProcessor(
        client_pool=client_pool, 
        embed_client_pool=embed_client_pool,
        embed_model=embed_model,
        threshold=threshold,
        candidate_threshold=candidate_threshold
    )
    judge = DomainJudge(client_pool=client_pool)
    
    print(f"Starting extraction for domain: {domain_query} (Type: {'Code' if is_code_domain else 'Text'})")
    
    # 2. Auto-discovery of Pipelines
    active_pipelines = []
    pipeline_dir = "pipelines"
    
    target_files = [
        "p2_sequential.py", 
        "p3_reflection.py", 
        "p4_taxonomy_explorer.py",
        "p5_debate.py",
        "p6_camel.py"
    ] 
    
    for filename in target_files:
        module_name = f"pipelines.{filename[:-3]}"
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BasePipeline) and obj != BasePipeline:
                # Initialize with processor for saturation check
                active_pipelines.append((name, obj(gen_agent, processor)))

    # --- PHASE 1: SATURATION GENERATION ---
    print("\n=== PHASE 1: GENERATION (Saturation Mode) ===")
    pipeline_raw_outputs = {} # {pipeline_name: [raw_points]}
    
    for name, pipeline in active_pipelines:
        print(f"Running pipeline: {name}...")
        try:
            raw_points = await pipeline.run(domain_query)
            pipeline_raw_outputs[name] = raw_points
            
            # Save raw points immediately
            with open(os.path.join(output_dir, f"{name}_raw.json"), "w") as f:
                json.dump(raw_points, f, indent=2)
            print(f"  {name} finished. Total raw points: {len(raw_points)}")
        except Exception as e:
            print(f"  Error running {name}: {e}")
            import traceback
            traceback.print_exc()

    if not pipeline_raw_outputs:
        print("No outputs generated. Exiting.")
        return

    # --- PHASE 2: GLOBAL EVALUATION ---
    print("\n=== PHASE 2: EVALUATION (Post-Audit) ===")
    
    # 1. Build Global Union Set (Deduplicate ALL raw points from ALL pipelines)
    # We use processor.build_union_set but we need to point it to the raw files
    print("Building global deduplicated union set...")
    await processor.build_union_set(output_dir) 
    
    # 2. Global Domain Audit (Only judge each unique node ONCE)
    print(f"Auditing {len(processor.union_set)} unique knowledge nodes...")
    unique_texts = [node["representative_text"] for node in processor.union_set]
    audit_results = await judge.check_batch(domain_query, unique_texts)
    
    # Apply audit results to the union set
    valid_nodes_count = 0
    for node, is_valid in zip(processor.union_set, audit_results):
        node["is_in_domain"] = is_valid
        if is_valid:
            valid_nodes_count += 1
            
    print(f"Audit complete: {valid_nodes_count} valid / {len(processor.union_set)} total nodes.")
    
    # Save the audited union set
    processor.save_union_set(os.path.join(output_dir, "union_set.json"))
    
    # 3. Metric Calculation
    # Recall = (Pipeline's valid nodes) / (Total valid nodes in union set)
    # Accuracy = (Pipeline's valid raw points) / (Pipeline's total raw points)
    
    total_valid_nodes = valid_nodes_count
    pipeline_metrics = {}
    
    for name, raw_points in pipeline_raw_outputs.items():
        # Find which nodes in the union set this pipeline contributed to
        covered_indices = processor.get_pipeline_coverage(name)
        
        # Filter these nodes by audit result
        valid_covered_indices = [i for i in covered_indices if processor.union_set[i]["is_in_domain"]]
        
        recall = len(valid_covered_indices) / total_valid_nodes if total_valid_nodes > 0 else 0
        
        # For Accuracy, we need to check each raw point of this pipeline
        # (Alternatively, use the audit result of the nodes they mapped to)
        valid_raw_count = 0
        # A raw point is valid if the node it was merged into is valid
        # We can map raw points back to nodes during build_union_set if needed, 
        # but here's a simpler way:
        for node_idx in covered_indices:
            node = processor.union_set[node_idx]
            if node["is_in_domain"]:
                # Count how many raw points from this pipeline went into this valid node
                for source in node["source_entries"]:
                    if source["pipeline"] == name:
                        valid_raw_count += 1
        
        accuracy = valid_raw_count / len(raw_points) if raw_points else 0
        pipeline_metrics[name] = {"recall": recall, "accuracy": accuracy, "raw_total": len(raw_points)}

    # --- FINAL LEADERBOARD ---
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
