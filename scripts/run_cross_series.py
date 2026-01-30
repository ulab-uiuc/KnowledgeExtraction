#!/usr/bin/env python3
"""
Experiment 4: Cross-Series Model Comparison (~7B level)
Compare knowledge extraction across different model families at similar parameter counts.
"""
import asyncio
import json
import os
import pickle
import importlib
from tqdm import tqdm
from agents.call_agent import GenAgent
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor

# --- Cross-Series Experiment Configuration ---
CROSS_SERIES_MODELS = [
    "meta/llama-3.1-8b-instruct",               # Llama 3.1 (US)
    "qwen/qwen2.5-7b-instruct",                 # Qwen 2.5 (CN)
    # "mistralai/mistral-7b-instruct-v0.3",       # Mistral (EU)
    "deepseek-ai/deepseek-r1-distill-qwen-7b",  # DeepSeek R1 (Reasoning-enhanced)
]

# Local endpoint configuration for models that can't use NVIDIA API
LOCAL_ENDPOINTS = {
    "qwen/qwen2.5-7b-instruct": "http://localhost:30001/v1"
}

OUTPUT_BASE_DIR = "results/cross_series_7b"
MAX_TURNS = 15  # Temporarily set to 2 for debugging taxonomy node counts

TARGET_DOMAINS = {
    "Deep_Learning": "Deep Learning",
    "Machine_Learning_Systems": "Machine Learning Systems",
    "Probabilistic_Methods": "Probabilistic Methods"
}

PIPELINE_CONFIG = {
    "id": "P4_Taxonomy_L5W5",
    "module": "pipelines.p4_taxonomy_explorer",
    "class": "RecursiveTaxonomyExplorer",
    "params": {
        "l1_width": 5,
        "l2_width": 5
    }
}

async def run_extraction_task(domain_id, domain_query, model_name, api_keys, processor, pbar):
    """Executes a single extraction pipeline"""
    primary_folder = model_name.split("/")[-1]
    output_dir = os.path.join(OUTPUT_BASE_DIR, domain_id, primary_folder)
    os.makedirs(output_dir, exist_ok=True)

    exp_id = PIPELINE_CONFIG["id"]
    traj_file = os.path.join(output_dir, f"{exp_id}.trajectory.json")
    raw_file = os.path.join(output_dir, f"{exp_id}_raw.json")
    emb_file = os.path.join(output_dir, f"{exp_id}.emb.pkl")
    token_file = os.path.join(output_dir, f"{exp_id}.tokens.json")

    base_desc = pbar.desc.strip()
    resume_traj = None
    start_turn = 0

    # Check for existing trajectory to resume
    if os.path.exists(traj_file):
        try:
            with open(traj_file, "r") as f:
                resume_traj = json.load(f)
            if isinstance(resume_traj, list) and len(resume_traj) > 0:
                last_meta = resume_traj[-1].get("meta", {})
                raw_val = last_meta.get("turn", 0)
                start_turn = 0 if raw_val == "init" else int(raw_val)
                pbar.update(start_turn)
                if last_meta.get("is_completed") and start_turn >= MAX_TURNS:
                    pbar.set_description(f"DONE | {base_desc}")
                    pbar.refresh()
                    return
        except:
            pass

    # Use local endpoint if specified, otherwise use default NVIDIA API
    base_url = LOCAL_ENDPOINTS.get(model_name, "https://integrate.api.nvidia.com/v1")
    gen_agent = GenAgent(api_key=api_keys, model=model_name, base_url=base_url)

    try:
        module = importlib.import_module(PIPELINE_CONFIG["module"])
        pipeline_class = getattr(module, PIPELINE_CONFIG["class"])
        pipeline = pipeline_class(gen_agent, processor, model=model_name, **PIPELINE_CONFIG["params"])

        async def save_callback(current_trajectory):
            with open(traj_file, "w") as f:
                json.dump(current_trajectory, f, indent=2, ensure_ascii=False)
            last_snapshot = current_trajectory[-1]
            raw_turn = last_snapshot.get("meta", {}).get("turn", 0)
            turn_num = 0 if raw_turn == "init" else int(raw_turn)
            pbar.n = min(turn_num, MAX_TURNS)
            pbar.refresh()
            all_points = last_snapshot.get("points", [])
            if all_points:
                with open(raw_file, "w") as f:
                    json.dump(all_points, f, indent=2, ensure_ascii=False)
                current_vectors = {p: processor.embed_cache[p] for p in all_points if p in processor.embed_cache}
                with open(emb_file, 'wb') as f_emb:
                    pickle.dump(current_vectors, f_emb)
            with open(token_file, "w") as f:
                json.dump({"total_tokens": gen_agent.total_tokens, "turn": turn_num}, f, indent=2)

        pipeline.set_callback(save_callback)

        try:
            await pipeline.run(domain_query, resume_trajectory=resume_traj, max_turns=MAX_TURNS, silent=True)
            pbar.n = MAX_TURNS
            pbar.set_description(f"DONE | {base_desc}")
            pbar.refresh()
        except Exception as e:
            import traceback
            pbar.set_description(f"ERR  | {base_desc}")
            with open("cross_series_errors.log", "a") as err_f:
                err_f.write(f"[{model_name} | {domain_id}] Error: {str(e)}\n{traceback.format_exc()}\n")

    except Exception as e:
        import traceback
        pbar.set_description(f"OUT_ERR | {base_desc}")
        with open("cross_series_errors.log", "a") as err_f:
            err_f.write(f"[{model_name} | {domain_id}] Outer Error: {str(e)}\n{traceback.format_exc()}\n")


async def main():
    if not os.path.exists("api.json"):
        print("Error: api.json not found")
        return

    with open("api.json") as f:
        api_data = json.load(f)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    embed_config = api_data.get("embed_config", {})
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        judge_model="deepseek-ai/deepseek-v3.1",
        embed_model=embed_config.get("model"),
        embed_client_pool=MultiKeyClientPool(api_keys=api_keys, base_url=embed_config.get("base_url"))
    )

    print(f"\n{'='*60}")
    print(f"🚀 Experiment 4: Cross-Series Model Comparison (~7B)")
    print(f"{'='*60}")
    print(f"📂 Output: {OUTPUT_BASE_DIR}")
    print(f"🤖 Models: {[m.split('/')[-1] for m in CROSS_SERIES_MODELS]}")
    print(f"📊 Domains: {list(TARGET_DOMAINS.keys())}")
    print(f"🔄 Max Turns: {MAX_TURNS}")
    print(f"{'='*60}\n")

    # Create progress bars and tasks
    all_tasks = []
    pbars = []
    pos = 0

    for domain_id, domain_query in TARGET_DOMAINS.items():
        domain_label = domain_id.replace("Machine_Learning_Systems", "ML-Sys").replace("Probabilistic_Methods", "Prob").replace("Deep_Learning", "DL")
        for model_name in CROSS_SERIES_MODELS:
            model_label = model_name.split("/")[-1][:12]  # Truncate for display
            desc = f"{model_label}|{domain_label}".ljust(25)
            pbar = tqdm(total=MAX_TURNS, desc=desc, position=pos, leave=True)
            pbars.append(pbar)
            all_tasks.append(run_extraction_task(domain_id, domain_query, model_name, api_keys, processor, pbar))
            pos += 1

    # Run all tasks
    await asyncio.gather(*all_tasks)

    # Cleanup
    for p in pbars:
        p.refresh()
        p.close()

    print("\n" * (pos + 1))
    print("=" * 60)
    print("✅ ALL TASKS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
