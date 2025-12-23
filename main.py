import asyncio
import json
import os
from agents.clientpool import MultiKeyClientPool
from agents.call_agent import GenAgent
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge
from core.evaluator import Evaluator
from pipelines.p1_baseline import BaselinePipeline

async def main():
    # Load configuration
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    
    # Initialize shared components
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    gen_agent = GenAgent(api_key=api_keys)
    processor = KnowledgeProcessor(client_pool=client_pool)
    judge = DomainJudge(client_pool=client_pool)
    
    # 1. User Input
    domain_query = "Linear Algebra" # Example query
    print(f"Starting extraction for domain: {domain_query}\n")
    
    # 2. Define Pipelines
    # In a real scenario, we could use inspection to load all classes in pipelines/
    pipelines = {
        "Baseline": BaselinePipeline(gen_agent)
    }
    
    all_outputs = {}
    
    # 3. Run Pipelines
    for name, pipeline in pipelines.items():
        print(f"Running pipeline: {name}...")
        points = await pipeline.run(domain_query)
        all_outputs[name] = points
        print(f"  Extracted {len(points)} points.")
    
    # 4. Global Processing (Deduplication & Union Set Construction)
    print("\nProcessing union set and deduplication...")
    pipeline_indices = {}
    for name, points in all_outputs.items():
        indices = await processor.process_pipeline_output(name, points)
        pipeline_indices[name] = indices
    
    union_size = processor.get_union_size()
    print(f"Total unique knowledge points found: {union_size}")
    
    # 5. Accuracy Verification (Judging)
    print("Verifying domain relevance...")
    pipeline_metrics = {}
    for name, points in all_outputs.items():
        judge_results = await judge.check_batch(domain_query, points)
        metrics = Evaluator.calculate_metrics(pipeline_indices[name], union_size, judge_results)
        pipeline_metrics[name] = metrics
    
    # 6. Final Report
    print("\n" + "="*30)
    print("KNOWLEDGE EXTRACTION LEADERBOARD")
    print("="*30)
    for name, metrics in pipeline_metrics.items():
        print(f"Pipeline: {name}")
        print(f"  Recall (Coverage): {metrics['recall']:.2%}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())

