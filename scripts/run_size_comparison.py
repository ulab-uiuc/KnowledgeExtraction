import asyncio
import json
import os
import pickle
import importlib
from tqdm import tqdm
from agents.call_agent import GenAgent
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor

# --- Size Experiment Configuration (Scaling Law) ---
EVOLUTION_MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct"
]

OUTPUT_BASE_DIR = "results/llama31_scaling_law"
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
    primary_folder = model_name.split("/")[-1]
    folder_candidates = [primary_folder, model_name.replace("/", "_")]
    
    output_dir = None
    max_turns_found = -1
    for candidate in folder_candidates:
        test_dir = os.path.join(OUTPUT_BASE_DIR, domain_id, candidate)
        if os.path.exists(test_dir):
            test_traj = os.path.join(test_dir, f"{COMPARISON_CONFIG['id']}.trajectory.json")
            if os.path.exists(test_traj):
                try:
                    with open(test_traj, "r") as f:
                        data = json.load(f)
                        current_turns = len(data) if isinstance(data, list) else 0
                        if current_turns > max_turns_found:
                            max_turns_found = current_turns
                            output_dir = test_dir
                except: pass
    
    if not output_dir:
        output_dir = os.path.join(OUTPUT_BASE_DIR, domain_id, primary_folder)
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
                if last_meta.get("is_completed") and start_turn >= MAX_TURNS:
                    pbar.set_description(f"DONE | {base_desc}")
                    pbar.refresh()
                    return
        except: pass

    gen_agent = GenAgent(api_key=api_keys, model=model_name)
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
            all_points = last_snapshot.get("points", [])
            if all_points:
                with open(raw_file, "w") as f: json.dump(all_points, f, indent=2, ensure_ascii=False)
                current_vectors = {p: processor.embed_cache[p] for p in all_points if p in processor.embed_cache}
                with open(emb_file, 'wb') as f_emb: pickle.dump(current_vectors, f_emb)
            with open(token_file, "w") as f: json.dump({"total_tokens": gen_agent.total_tokens, "turn": turn_num}, f, indent=2)

        pipeline.set_callback(save_callback)
        
        # We NO LONGER redirect stdout to avoid I/O closed file errors in async
        try:
            await pipeline.run(domain_query, resume_trajectory=resume_traj, max_turns=MAX_TURNS, silent=True)
            pbar.n = MAX_TURNS
            pbar.set_description(f"DONE | {base_desc}")
            pbar.refresh()
        except Exception as e:
            import traceback
            pbar.set_description(f"ERR  | {base_desc}")
            with open("r1_evolution_errors.log", "a") as err_f:
                err_f.write(f"[{model_name} | {domain_id}] Error: {str(e)}\n{traceback.format_exc()}\n")

    except Exception as e:
        import traceback
        pbar.set_description(f"OUT_ERR | {base_desc}")
        with open("r1_evolution_errors.log", "a") as err_f:
            err_f.write(f"[{model_name} | {domain_id}] Outer Error: {str(e)}\n{traceback.format_exc()}\n")

async def main():
    if not os.path.exists("api.json"): return
    with open("api.json") as f: api_data = json.load(f)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    api_keys = api_data["api_keys"]
    client_pool = MultiKeyClientPool(api_keys=api_keys)
    embed_config = api_data.get("embed_config", {})
    processor = KnowledgeProcessor(client_pool=client_pool, judge_model="deepseek-ai/deepseek-v3.1", embed_model=embed_config.get("model"))

    print(f"\n🚀 Starting Llama 3.1 Scaling Law Experiment (9 lines parallel)")
    all_tasks = []
    pbars = []
    pos = 0
    for domain_id, domain_query in TARGET_DOMAINS.items():
        domain_label = domain_id.replace("Machine_Learning_Systems", "ML-Sys").replace("Probabilistic_Methods", "Prob-Meth").replace("Deep_Learning", "Deep-L")
        for model_name in EVOLUTION_MODELS:
            size_label = "8B" if "8b" in model_name else "70B" if "70b" in model_name else "405B"
            desc = f"L31-{size_label}|{domain_label}".ljust(20)
            pbar = tqdm(total=MAX_TURNS, desc=desc, position=pos, leave=True)
            pbars.append(pbar)
            all_tasks.append(run_evolution_task(domain_id, domain_query, model_name, api_keys, processor, pbar))
            pos += 1
    await asyncio.gather(*all_tasks)
    for p in pbars:
        p.refresh()
        p.close()
    print("\n" * (pos + 1))
    print("--- ALL TASKS COMPLETED ---")

if __name__ == "__main__":
    asyncio.run(main())
