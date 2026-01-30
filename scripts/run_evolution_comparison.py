import asyncio
import json
import os
import pickle
import importlib
from tqdm import tqdm
from agents.call_agent import GenAgent
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor

# --- Evolution Experiment Configuration (RL Comparison) ---
EVOLUTION_MODELS = [
    "qwen/qwen2.5-7b-instruct",
    "qwen/qwen2.5-coder-7b-instruct"
]

# Local endpoint configuration for models that can't use NVIDIA API
# Note: qwen2.5-coder-7b-instruct works with NVIDIA API, only instruct model needs local deployment
# Using port 30002 (GPU 3) to avoid conflict with cross_series experiment on port 30001 (GPU 7)
LOCAL_ENDPOINTS = {
    "qwen/qwen2.5-7b-instruct": "http://localhost:30002/v1"
}

OUTPUT_BASE_DIR = "results/evolution_comparison"
MAX_TURNS = 15

TARGET_DOMAINS = {
    "Deep_Learning": "Deep Learning",
    "Machine_Learning_Systems": "Machine Learning Systems",
    "Probabilistic_Methods": "Probabilistic Methods"
}

COMPARISON_CONFIG = {
    "id": "P4_Taxonomy_L5W5",
    "module": "pipelines.p4_taxonomy_explorer",
    "class": "RecursiveTaxonomyExplorer",
    "params": {
        "l1_width": 5,
        "l2_width": 5
    }
}

async def run_evolution_task(domain_id, domain_query, model_name, api_keys, processor, pbar):
    """Executes a single extraction pipeline line"""
    folder_name = model_name.replace("/", "_")
    output_dir = os.path.join(OUTPUT_BASE_DIR, domain_id, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    exp_id = COMPARISON_CONFIG["id"]
    traj_file = os.path.join(output_dir, f"{exp_id}.trajectory.json")
    raw_file = os.path.join(output_dir, f"{exp_id}_raw.json")
    emb_file = os.path.join(output_dir, f"{exp_id}.emb.pkl")
    token_file = os.path.join(output_dir, f"{exp_id}.tokens.json")
    
    base_desc = pbar.desc.strip()
    resume_traj = None
    start_turn = 0
    if os.path.exists(traj_file):
        try:
            with open(traj_file, "r") as f: resume_traj = json.load(f)
            if isinstance(resume_traj, list) and len(resume_traj) > 0:
                last_meta = resume_traj[-1].get("meta", {})
                raw_val = last_meta.get("turn", 0)
                start_turn = 0 if raw_val == "init" else int(raw_val)
                pbar.update(start_turn)
                # Skip if already completed
                if last_meta.get("is_completed") and start_turn >= MAX_TURNS:
                    pbar.set_description(f"DONE | {base_desc}")
                    pbar.refresh()
                    return
        except: pass

    # Use local endpoint if specified, otherwise use default NVIDIA API
    base_url = LOCAL_ENDPOINTS.get(model_name, "https://integrate.api.nvidia.com/v1")
    gen_agent = GenAgent(api_key=api_keys, model=model_name, base_url=base_url)
    try:
        module = importlib.import_module(COMPARISON_CONFIG["module"])
        pipeline_class = getattr(module, COMPARISON_CONFIG["class"])
        pipeline = pipeline_class(gen_agent, processor, model=model_name, **COMPARISON_CONFIG["params"])
        
        async def save_callback(current_trajectory):
            with open(traj_file, "w") as f: json.dump(current_trajectory, f, indent=2, ensure_ascii=False)
            last_snapshot = current_trajectory[-1]
            raw_turn = last_snapshot.get("meta", {}).get("turn", 0)
            turn_num = 0 if raw_turn == "init" else int(raw_turn)
            pbar.n = min(turn_num, MAX_TURNS)
            pbar.refresh()
            # Save _raw.json and .emb.pkl files
            all_points = last_snapshot.get("points", [])
            if all_points:
                with open(raw_file, "w") as f:
                    json.dump(all_points, f, indent=2, ensure_ascii=False)
                current_vectors = {p: processor.embed_cache[p] for p in all_points if p in processor.embed_cache}
                with open(emb_file, 'wb') as f_emb:
                    pickle.dump(current_vectors, f_emb)
            # Save token usage
            with open(token_file, "w") as f:
                json.dump({"total_tokens": gen_agent.total_tokens, "turn": turn_num}, f, indent=2)

        pipeline.set_callback(save_callback)
        await pipeline.run(domain_query, resume_trajectory=resume_traj, max_turns=MAX_TURNS)
        pbar.n = MAX_TURNS
        pbar.set_description(f"DONE | {base_desc}")
    except Exception as e:
        pbar.set_description(f"ERR  | {base_desc}")
        print(f"\n❌ Error in {domain_id}/{model_name}: {str(e)}")

async def main():
    if not os.path.exists("api.json"): return
    with open("api.json") as f: api_data = json.load(f)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    embed_config = api_data.get("embed_config", {})
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_model=embed_config.get("model"),
        embed_client_pool=MultiKeyClientPool(
            api_keys=api_keys,
            base_url=embed_config.get("base_url")
        )
    )

    print(f"\n🚀 Starting Qwen RL Comparison Experiment")
    all_tasks = []
    pbars = []
    pos = 0
    for domain_id, domain_query in TARGET_DOMAINS.items():
        domain_label = domain_id.replace("Machine_Learning_Systems", "ML-Sys").replace("Probabilistic_Methods", "Prob-Meth").replace("Deep_Learning", "Deep-L")
        for model_name in EVOLUTION_MODELS:
            model_label = "Coder" if "coder" in model_name.lower() else "Instruct"
            desc = f"RL-{model_label}|{domain_label}".ljust(20)
            pbar = tqdm(total=MAX_TURNS, desc=desc, position=pos, leave=True)
            pbars.append(pbar)
            all_tasks.append(run_evolution_task(domain_id, domain_query, model_name, api_keys, processor, pbar))
            pos += 1
    await asyncio.gather(*all_tasks)
    for p in pbars: p.close()

if __name__ == "__main__":
    asyncio.run(main())
