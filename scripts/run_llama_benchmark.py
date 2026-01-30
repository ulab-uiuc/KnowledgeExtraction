import asyncio
import json
import os
import importlib
import inspect
import pickle
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

async def run_single_experiment(category_name, sub_category, model_name, client_pool, api_data):
    """Run all pipelines for a single (Area + Model) combination"""
    api_keys = api_data["api_keys"]
    embed_config = api_data.get("embed_config", {})
    
    query_id = sub_category.lower().replace(" ", "_")
    cat_id = category_name.lower().replace(" ", "_")
    output_dir = os.path.join("results", cat_id, query_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n" + "="*60, flush=True)
    print(f"[PROCESS] Category: {category_name} | Area: {sub_category}", flush=True)
    print(f"[MODEL] {model_name}", flush=True)
    print("="*60 + "\n", flush=True)

    # 1. Initialize components
    gen_agent = GenAgent(api_key=api_keys, model=model_name)
    
    # Initialize embed client pool for potential local/custom embedding server
    embed_client_pool = MultiKeyClientPool(
        api_keys=[], # Local usually doesn't need keys
        base_url=embed_config.get("base_url", "https://integrate.api.nvidia.com/v1")
    )

    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=embed_client_pool,
        embed_model=embed_config.get("model", "Qwen/Qwen3-Embedding-8B")
    )

    # 2. Load selected Pipelines
    active_pipelines = []
    for p_file in PIPELINES_TO_RUN:
        module_name = f"pipelines.{p_file}"
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BasePipeline) and obj != BasePipeline:
                    active_pipelines.append((name, obj(gen_agent, processor, model=model_name)))
        except Exception as e:
            print(f"  [Error] Failed to load pipeline {p_file}: {e}", flush=True)

    # 3. Run each Pipeline
    model_slug = model_name.split("/")[-1]

    for name, pipeline in active_pipelines:
        raw_file = os.path.join(output_dir, f"{model_slug}_{name}_raw.json")
        emb_file = os.path.join(output_dir, f"{model_slug}_{name}.emb.pkl")
        
        # FIX/RESUME LOGIC
        if os.path.exists(raw_file):
            if os.path.exists(emb_file):
                print(f"  [Skip] {name}: Files already exist.", flush=True)
                continue
            else:
                print(f"  [Fix] {name}: JSON exists but PKL missing. Repairing...", flush=True)
                with open(raw_file, "r") as f:
                    raw_points = json.load(f)
                if raw_points:
                    await processor.get_embeddings(raw_points)
                    processor.save_embeddings(emb_file)
                    print(f"    -> Repair complete: {len(raw_points)} embeddings saved.", flush=True)
                continue

        print(f"  [Run] Pipeline: {name}...", flush=True)
        try:
            # IMPORTANT: We don't clear the whole cache if we want to reuse embeddings
            # but we need to track what's new for THIS pipeline.
            # BasePipeline.run will populate processor.embed_cache
            
            async def save_callback(current_points):
                # Save JSON
                with open(raw_file, "w") as f:
                    json.dump(current_points, f, indent=2)
                # Save only the relevant embeddings to the PKL to keep it clean
                current_vectors = {p: processor.embed_cache[p] for p in current_points if p in processor.embed_cache}
                with open(emb_file, 'wb') as f_emb:
                    pickle.dump(current_vectors, f_emb)

            pipeline.set_callback(save_callback)
            run_result = await pipeline.run(sub_category)
            raw_points = run_result["points"]
            total_tokens = run_result["total_tokens"]
            
            # Save token count
            token_file = os.path.join(output_dir, f"{model_slug}_{name}.tokens.json")
            with open(token_file, "w") as f_tok:
                json.dump({"total_tokens": total_tokens}, f_tok)
            
            # Ensure final save
            await save_callback(raw_points)
            print(f"  [Done] {name}: {len(raw_points)} points. Cost: {total_tokens} tokens.", flush=True)
            
        except Exception as e:
            print(f"  [Error] {name} failed: {e}", flush=True)

async def main():
    if not os.path.exists("api.json"):
        print("Error: api.json not found.", flush=True)
        return
        
    with open("api.json") as f:
        api_data = json.load(f)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)

    # You can re-order or filter CATEGORIES here if you want to focus on specific areas
    for cat_name, sub_cats in CATEGORIES.items():
        for sub_cat in sub_cats:
            for model_name in LLAMA_MODELS:
                await run_single_experiment(cat_name, sub_cat, model_name, client_pool, api_data)

if __name__ == "__main__":
    asyncio.run(main())
