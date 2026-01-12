import asyncio
import json
import os
import importlib
import inspect
from agents.clientpool import MultiKeyClientPool
from agents.call_agent import GenAgent
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge
from pipelines.base import BasePipeline

# --- Experiment Configuration ---
CATEGORIES = {
    "Deep Learning": [
        "Transformer Architectures",
        "Generative Models",
        "Deep Learning Theory"
    ],
    "Reinforcement Learning": [
        "Markov Decision Processes",
        "Policy Gradient Methods",
        "Hierarchical Reinforcement Learning"
    ],
    "Trustworthy ML": [
        "Algorithmic Fairness",
        "Adversarial Robustness",
        "Model Interpretability"
    ]
}

LLAMA_MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct"
]

PIPELINES_TO_RUN = [
    "p2_sequential",
    "p3_reflection",
    "p4_taxonomy_explorer",
    "p5_debate"
]

JUDGE_MODEL = "meta/llama-3.1-8b-instruct"

async def run_single_experiment(category_name, sub_category, model_name, client_pool, api_keys):
    """Run all pipelines for a single (Area + Model) combination"""
    query_id = sub_category.lower().replace(" ", "_")
    cat_id = category_name.lower().replace(" ", "_")
    output_dir = os.path.join("results", cat_id, query_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n>>> Running: [{category_name}] {sub_category} | Model: {model_name}")

    # 1. Initialize components
    gen_agent = GenAgent(api_key=api_keys, model=model_name)
    
    # Use default embedding settings
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=client_pool,
        embed_model="nvidia/nv-embed-v1"
    )
    
    judge = DomainJudge(client_pool=client_pool, model=JUDGE_MODEL)

    # 2. Load selected Pipelines
    active_pipelines = []
    for p_file in PIPELINES_TO_RUN:
        module_name = f"pipelines.{p_file}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasePipeline) and obj != BasePipeline:
                    # Initialize with model name for internal use
                    active_pipelines.append((name, obj(gen_agent, processor, model=model_name)))
        except Exception as e:
            print(f"  Error loading pipeline {p_file}: {e}")

    # 3. Run each Pipeline and save raw outputs
    pipeline_raw_outputs = {}
    model_slug = model_name.split("/")[-1] # e.g., llama-3.1-8b-instruct

    for name, pipeline in active_pipelines:
        raw_file = os.path.join(output_dir, f"{model_slug}_{name}_raw.json")
        
        # Resume support: skip if file exists
        if os.path.exists(raw_file):
            print(f"  Pipeline {name} already exists. Skipping.")
            with open(raw_file, "r") as f:
                pipeline_raw_outputs[name] = json.load(f)
            continue

        print(f"  Running pipeline: {name}...")
        try:
            raw_points = await pipeline.run(sub_category)
            pipeline_raw_outputs[name] = raw_points
            
            with open(raw_file, "w") as f:
                json.dump(raw_points, f, indent=2)
            print(f"    Finished. Points: {len(raw_points)}")
        except Exception as e:
            print(f"    Error in {name}: {e}")

    # 4. Global Evaluation and Auditing
    # This phase is typically handled in a separate post-processing step
    # to avoid overwhelming the API during massive extraction runs.
    print(f"  Extraction for {model_slug} on {sub_category} done.")

async def main():
    # Load API Keys
    if not os.path.exists("api.json"):
        print("Error: api.json not found.")
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)

    # Main loop through categories and models
    for cat_name, sub_cats in CATEGORIES.items():
        for sub_cat in sub_cats:
            for model_name in LLAMA_MODELS:
                await run_single_experiment(cat_name, sub_cat, model_name, client_pool, api_keys)

if __name__ == "__main__":
    asyncio.run(main())
