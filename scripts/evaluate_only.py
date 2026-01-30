import asyncio
import json
import os
import argparse
import sys

# DEBUG: Print immediately before any heavy imports
print(">>> Script started, importing heavy modules...", flush=True)

import numpy as np
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge

async def main():
    # ... rest of the code ...
    # 0. Parse arguments
    parser = argparse.ArgumentParser(description="Multi-Model Knowledge Extraction Evaluation")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing _raw.json files")
    parser.add_argument("--query", type=str, required=True, help="The domain query for the judge")
    args = parser.parse_args()

    # Force unbuffered output for real-time logs
    print(f"=== INITIALIZING EVALUATION SYSTEM ===", flush=True)
    
    results_dir = args.dir
    domain_query = args.query
    
    if not os.path.exists("api.json"):
        print("Error: api.json not found.", flush=True)
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    
    # 1. Initialize Processor (this will load disk cache)
    print(f"[*] Loading KnowledgeProcessor and Disk Cache...", end="", flush=True)
    embed_config = api_data.get("embed_config", {})
    embed_client_pool = MultiKeyClientPool(
        api_keys=embed_config.get("api_keys", api_keys), 
        base_url=embed_config.get("base_url")
    )
    
    processor = KnowledgeProcessor(
        client_pool=client_pool, 
        embed_client_pool=embed_client_pool,
        embed_model=embed_config.get("model", "nvidia/nv-embed-v1")
    )
    print(" Done.", flush=True)

    judge = DomainJudge(client_pool=client_pool, model="meta/llama-3.1-8b-instruct")

    print(f"\n[Target Directory]: {results_dir}")
    print(f"[Domain Query]:     {domain_query}", flush=True)

    # 2. Load all raw outputs
    print(f"[*] Scanning for raw result files...", flush=True)
    raw_files = [f for f in os.listdir(results_dir) if f.endswith("_raw.json")]
    if not raw_files:
        print("Error: No _raw.json files found in the specified directory.", flush=True)
        return

    model_data = {} 
    total_raw_points_count = 0
    for f in sorted(raw_files):
        print(f"    - Loading {f}...", end="", flush=True)
        parts = f.replace("_raw.json", "").split("_")
        model_name = parts[0]
        pipeline_name = parts[1]
        
        if model_name not in model_data:
            model_data[model_name] = {}
            
        with open(os.path.join(results_dir, f), "r") as f_in:
            points = json.load(f_in)
            model_data[model_name][pipeline_name] = points
            total_raw_points_count += len(points)
        print(f" ({len(points)} points)")

    print(f"\n[*] Total raw points to process: {total_raw_points_count}", flush=True)

    # 3. Build Global Union Set
    print("\n[Phase 1] Building Global Pseudo Ground Truth (Deduplicating)...", flush=True)
    await processor.build_union_set(results_dir) 
    
    # 4. Global Domain Audit
    print(f"\n[Phase 2] Auditing {len(processor.union_set)} unique knowledge nodes...", flush=True)
    unique_texts = [node["representative_text"] for node in processor.union_set]
    audit_results = await judge.check_batch(domain_query, unique_texts)
    
    valid_nodes_count = 0
    for node, is_valid in zip(processor.union_set, audit_results):
        node["is_in_domain"] = is_valid
        if is_valid:
            valid_nodes_count += 1
            
    print(f"Audit complete: {valid_nodes_count} valid / {len(processor.union_set)} total nodes.", flush=True)
    processor.save_union_set(os.path.join(results_dir, "union_set.json"))
    
    # 5. Calculate Metrics
    print("\n[Phase 3] Calculating Final Metrics...", flush=True)
    global_total_valid = valid_nodes_count
    model_metrics = {}

    for model_name, pipelines in model_data.items():
        model_covered_indices = []
        for p_name in pipelines:
            model_covered_indices.extend(processor.get_pipeline_coverage(f"{model_name}_{p_name}"))
        
        model_covered_indices = list(set(model_covered_indices))
        model_valid_indices = [i for i in model_covered_indices if processor.union_set[i]["is_in_domain"]]
        
        model_recall = len(model_valid_indices) / global_total_valid if global_total_valid > 0 else 0
        
        total_raw = 0
        total_valid_raw = 0
        pipeline_stats = {}
        
        for p_name, raw_points in pipelines.items():
            full_name = f"{model_name}_{p_name}"
            p_covered = processor.get_pipeline_coverage(full_name)
            p_valid_covered = [i for i in p_covered if processor.union_set[i]["is_in_domain"]]
            p_recall_vs_model = len(p_valid_covered) / len(model_valid_indices) if model_valid_indices else 0
            
            p_valid_raw_count = 0
            for node_idx in p_covered:
                node = processor.union_set[node_idx]
                if node["is_in_domain"]:
                    for source in node["source_entries"]:
                        if source["pipeline"] == full_name:
                            p_valid_raw_count += 1
            
            p_acc = p_valid_raw_count / len(raw_points) if raw_points else 0
            pipeline_stats[p_name] = {"recall_internal": p_recall_vs_model, "accuracy": p_acc, "count": len(raw_points)}
            total_raw += len(raw_points)
            total_valid_raw += p_valid_raw_count

        model_metrics[model_name] = {
            "global_recall": model_recall,
            "overall_accuracy": model_accuracy if 'model_accuracy' in locals() else (total_valid_raw/total_raw if total_raw else 0),
            "pipelines": pipeline_stats,
            "valid_knowledge_count": len(model_valid_indices)
        }

    # 6. Final Report
    print("\n" + "="*85)
    print(f"FINAL LEADERBOARD: {domain_query}")
    print("="*85)
    print(f"{'Model':25} | {'Global Recall':15} | {'Accuracy':12} | {'Valid Points'}")
    print("-" * 85)
    
    sorted_models = sorted(model_metrics.keys(), key=lambda x: model_metrics[x]['global_recall'], reverse=True)
    for m_name in sorted_models:
        m = model_metrics[m_name]
        print(f"{m_name:25} | {m['global_recall']:15.2%} | {m['overall_accuracy']:12.2%} | {m['valid_knowledge_count']}")
    
    print("\n" + "-"*85)
    print("PIPELINE EFFICIENCY (Internal Recall within each model):")
    print("-" * 85)
    p_efficiency = {}
    for m_name, m_data in model_metrics.items():
        for p_name, p_data in m_data["pipelines"].items():
            if p_name not in p_efficiency: p_efficiency[p_name] = []
            p_efficiency[p_name].append(p_data["recall_internal"])
    
    for p_name in sorted(p_efficiency.keys()):
        avg_eff = sum(p_efficiency[p_name]) / len(p_efficiency[p_name])
        print(f"{p_name:30} | Average Internal Recall: {avg_eff:.2%}")
    
    print("="*85)
    print(f"Global Pseudo Ground Truth (Union Set) Size: {global_total_valid} unique valid points.")

if __name__ == "__main__":
    asyncio.run(main())
