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

async def main():
    # Load configuration
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    
    # Initialize shared components
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    gen_agent = GenAgent(api_key=api_keys)
    # Hybrid Processor: Threshold 0.92 for auto-merge, 0.75 for LLM check
    processor = KnowledgeProcessor(client_pool=client_pool, threshold=0.92, candidate_threshold=0.75)
    judge = DomainJudge(client_pool=client_pool)
    
    # 1. User Input
    domain_query = "Linear Algebra" 
    query_id = domain_query.lower().replace(" ", "_")
    output_dir = os.path.join("results", query_id)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting extraction for domain: {domain_query}")
    print(f"Results will be saved to: {output_dir}\n")
    
    # 2. Auto-discovery of Pipelines
    standard_pipelines = []
    pipeline_dir = "pipelines"
    for filename in os.listdir(pipeline_dir):
        if filename.endswith(".py") and filename not in ["base.py"] and not filename.startswith("__"):
            # Skip old/redundant turn-based files
            if any(x in filename for x in ["turns", "baseline"]): continue 
            
            module_name = f"pipelines.{filename[:-3]}"
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasePipeline) and obj != BasePipeline:
                    standard_pipelines.append((name, obj(gen_agent)))
    
    all_outputs = {}
    pipeline_raw_counts = {}
    rejected_log = {} # To store points rejected by the judge
    
    # 3. Sequential Multi-Turn Experiment (Serves as 1-4 turns)
    print("Running Sequential Multi-Turn Experiment (1-4 turns)...")
    history = []
    cumulative_raw_points = []
    
    base_prompt = f"List all atomic knowledge points about '{domain_query}' in bullet points."
    
    for turn in range(1, 5):
        print(f"  Turn {turn}...")
        if turn == 1:
            response = await gen_agent.generate(base_prompt)
            current_turn_points = KnowledgeCleaner.clean_bullet_points(response)
            history.append(f"User: {base_prompt}\nAssistant: {response}")
        else:
            follow_up = "What else? Please provide more specific and in-depth points that were not mentioned above."
            context = "\n".join(history)
            response = await gen_agent.generate(f"{context}\nUser: {follow_up}")
            current_turn_points = KnowledgeCleaner.clean_bullet_points(response)
            history.append(f"User: {follow_up}\nAssistant: {response}")
        
        cumulative_raw_points.extend(current_turn_points)
        cumulative_raw_points = list(dict.fromkeys(cumulative_raw_points))
        
        # --- Pre-filtering with Judge ---
        print(f"    Verifying {len(cumulative_raw_points)} points for Turn {turn} (Parallel)...")
        judge_results = await judge.check_batch(domain_query, cumulative_raw_points)
        
        filtered_points = []
        rejected_points = []
        for p, is_valid in zip(cumulative_raw_points, judge_results):
            if is_valid:
                filtered_points.append(p)
            else:
                rejected_points.append(p)
        
        name = f"Sequential_Turn_{turn}"
        all_outputs[name] = filtered_points
        pipeline_raw_counts[name] = len(cumulative_raw_points)
        rejected_log[name] = rejected_points
        
        with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
            json.dump(filtered_points, f, indent=2)
        print(f"    Turn {turn} summary: {len(filtered_points)} in-domain / {len(cumulative_raw_points)} total.")

    # 4. Run Other Standard Pipelines (like Reflection)
    print("\nRunning other standard pipelines...")
    for name, pipeline in standard_pipelines:
        print(f"Running pipeline: {name}...")
        try:
            raw_points = await pipeline.run(domain_query)
            # --- Pre-filtering with Judge ---
            print(f"  Verifying {len(raw_points)} points (Parallel)...")
            judge_results = await judge.check_batch(domain_query, raw_points)
            
            filtered_points = []
            rejected_points = []
            for p, is_valid in zip(raw_points, judge_results):
                if is_valid:
                    filtered_points.append(p)
                else:
                    rejected_points.append(p)
            
            all_outputs[name] = filtered_points
            pipeline_raw_counts[name] = len(raw_points)
            rejected_log[name] = rejected_points
            
            with open(os.path.join(output_dir, f"{name}.json"), "w") as f:
                json.dump(filtered_points, f, indent=2)
            print(f"  {name} summary: {len(filtered_points)} in-domain / {len(raw_points)} total.")
        except Exception as e:
            print(f"  Error running {name}: {e}")
    
    # Save rejected points for inspection
    with open(os.path.join(output_dir, "rejected_points.json"), "w") as f:
        json.dump(rejected_log, f, indent=2)
    print(f"\nRejected points log saved to: {os.path.join(output_dir, 'rejected_points.json')}")
    
    # 5. Global Processing (Deduplication & Union Set Construction)
    print("\nProcessing union set and deduplication (Embedding + LLM Hybrid)...")
    # Clean output_dir of old files
    current_files = [f"{name}.json" for name in all_outputs.keys()] + ["rejected_points.json"]
    for f in os.listdir(output_dir):
        if f.endswith(".json") and f != "union_set.json" and f not in current_files:
            os.remove(os.path.join(output_dir, f))
            
    await processor.build_union_set(output_dir)
    processor.save_union_set(os.path.join(output_dir, "union_set.json"))
    
    union_size = processor.get_union_size()
    print(f"Total unique knowledge points found: {union_size}")
    
    # 6. Metric Calculation
    print("\nCalculating metrics...")
    pipeline_metrics = {}
    for name, points in all_outputs.items():
        covered_indices = processor.get_pipeline_coverage(name)
        recall = len(covered_indices) / union_size if union_size > 0 else 0
        accuracy = len(points) / pipeline_raw_counts[name] if pipeline_raw_counts[name] > 0 else 0
        pipeline_metrics[name] = {"recall": recall, "accuracy": accuracy}
    
    # 7. Final Report (Leaderboard)
    print("\n" + "="*65)
    print(f"LEADERBOARD for Domain: {domain_query}")
    print("="*65)
    seq_keys = [f"Sequential_Turn_{i}" for i in range(1, 5)]
    other_keys = sorted([k for k in pipeline_metrics.keys() if k not in seq_keys], 
                        key=lambda x: pipeline_metrics[x]['recall'], reverse=True)
    
    for name in seq_keys + other_keys:
        if name not in pipeline_metrics: continue
        m = pipeline_metrics[name]
        print(f"{name:30} | Recall: {m['recall']:7.2%} | Accuracy: {m['accuracy']:7.2%}")
    print("="*65)

if __name__ == "__main__":
    asyncio.run(main())
