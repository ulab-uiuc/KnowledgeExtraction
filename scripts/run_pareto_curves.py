import asyncio
import json
import os
import importlib
import inspect
import pickle
from agents.clientpool import MultiKeyClientPool
from agents.call_agent import GenAgent
from core.processor import KnowledgeProcessor
from pipelines.base import BasePipeline

TARGET_DOMAINS = {
    "Transformer_Architectures": "Transformer Architectures",
    "Policy_Gradient_Methods": "Policy Gradient Methods",
    "Algorithmic_Fairness": "Algorithmic Fairness"
}

MODEL_NAME = "meta/llama-3.1-405b-instruct"
OUTPUT_BASE_DIR = "results/pareto_curves_405b"
MAX_TURNS = 30  # Increased from 15

PARETO_CONFIGS = [
    {"id": "P2_Sequential", "module": "pipelines.p2_sequential", "class": "SequentialPipeline", "params": {}},
    {"id": "P3_Reflection", "module": "pipelines.p3_reflection", "class": "ReflectionPipeline", "params": {}},
    {"id": "P4_Taxonomy_L2W2", "module": "pipelines.p4_taxonomy_explorer", "class": "RecursiveTaxonomyExplorer", "params": {"l1_width": 2, "l2_width": 2}},
    {"id": "P4_Taxonomy_L3W3", "module": "pipelines.p4_taxonomy_explorer", "class": "RecursiveTaxonomyExplorer", "params": {"l1_width": 3, "l2_width": 3}},
    {"id": "P4_Taxonomy_L5W5", "module": "pipelines.p4_taxonomy_explorer", "class": "RecursiveTaxonomyExplorer", "params": {"l1_width": 5, "l2_width": 5}},
    {"id": "P5_MultiProfile_N3", "module": "pipelines.p5_debate", "class": "MultiProfileDebatePipeline", "params": {"num_profiles": 3}},
    {"id": "P5_MultiProfile_N10", "module": "pipelines.p5_debate", "class": "MultiProfileDebatePipeline", "params": {"num_profiles": 10}},
    {"id": "P5_MultiProfile_N20", "module": "pipelines.p5_debate", "class": "MultiProfileDebatePipeline", "params": {"num_profiles": 20}},
]

async def run_trajectory_experiment(domain_id, domain_query, api_data):
    api_keys = api_data["api_keys"]
    embed_config = api_data.get("embed_config", {})
    output_dir = os.path.join(OUTPUT_BASE_DIR, domain_id)
    os.makedirs(output_dir, exist_ok=True)

    client_pool = MultiKeyClientPool(api_keys=api_keys)
    gen_agent = GenAgent(api_key=api_keys, model=MODEL_NAME)
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        embed_client_pool=MultiKeyClientPool(api_keys=api_keys, base_url=embed_config.get("base_url")),
        embed_model=embed_config.get("model"),
        threshold=0.92,
        candidate_threshold=0.7,
        judge_model="deepseek-ai/deepseek-v3.1"
    )

    for config in PARETO_CONFIGS:
        exp_id = config["id"]
        traj_file = os.path.join(output_dir, f"{exp_id}.trajectory.json")
        raw_file = os.path.join(output_dir, f"{exp_id}_raw.json")
        emb_file = os.path.join(output_dir, f"{exp_id}.emb.pkl")

        resume_traj = None
        if os.path.exists(traj_file):
            try:
                # REPAIR LOGIC: If traj exists but PKL is missing, re-generate embeddings
                if not os.path.exists(emb_file):
                    print(f"  [Repair] {exp_id}: PKL missing. Re-embedding points...")
                    with open(traj_file, "r") as f:
                        temp_traj = json.load(f)
                    if temp_traj:
                        all_pts = temp_traj[-1]["points"]
                        if all_pts:
                            vectors = await processor.get_embeddings(all_pts)
                            with open(emb_file, 'wb') as f_emb:
                                pickle.dump({p: v for p, v in zip(all_pts, vectors)}, f_emb)

                with open(traj_file, "r") as f:
                    resume_traj = json.load(f)
                
                # ENHANCED SKIP LOGIC:
                last_meta = resume_traj[-1]["meta"]
                last_turn = last_meta.get("turn", 0)
                
                # If already completed naturally (saturated before old limit) or reached new limit
                if last_meta.get("is_completed"):
                    if last_turn < 15: # Saturated naturally
                        print(f"  [Skip] {exp_id}: Already completed naturally at Turn {last_turn}.")
                        continue
                    elif last_turn >= MAX_TURNS: # Reached new limit
                        print(f"  [Skip] {exp_id}: Already reached MAX_TURNS ({MAX_TURNS}).")
                        continue
                    else:
                        # Reached old limit (15), allow resume to 30
                        print(f"  [Resume] {exp_id}: Previously cut off at Turn {last_turn}. Resuming to {MAX_TURNS}...")
                        last_meta["is_completed"] = False # Reset flag for continuation
                
                if isinstance(last_turn, int) and last_turn >= MAX_TURNS:
                    print(f"  [Skip] {exp_id}: Already at or beyond MAX_TURNS.")
                    continue
                
                # Fallback for old files without is_completed flag
                if isinstance(last_meta["turn"], int) and last_meta["turn"] > 5:
                    if last_meta["novel_points_count"] < 5: # Slightly more lenient for old files
                        print(f"  [Skip] {exp_id}: Likely saturated (legacy).")
                        continue
            except Exception:
                pass

        print(f"\n[Trajectory Run] {exp_id}...")
        
        try:
            module = importlib.import_module(config["module"])
            pipeline_class = getattr(module, config["class"])
            pipeline = pipeline_class(gen_agent, processor, model=MODEL_NAME, **config["params"])
            
            async def save_callback(current_trajectory):
                # Save full trajectory snapshot
                with open(traj_file, "w") as f:
                    json.dump(current_trajectory, f, indent=2, ensure_ascii=False)
                
                # Update individual raw and emb files for compatibility
                last_snapshot = current_trajectory[-1]
                all_points = last_snapshot["points"]
                with open(raw_file, "w") as f:
                    json.dump(all_points, f, indent=2, ensure_ascii=False)
                
                # Robust embedding saving: save what we have, don't gate with 'if current_vectors'
                current_vectors = {p: processor.embed_cache[p] for p in all_points if p in processor.embed_cache}
                with open(emb_file, 'wb') as f_emb:
                    pickle.dump(current_vectors, f_emb)

            pipeline.set_callback(save_callback)
            await pipeline.run(domain_query, resume_trajectory=resume_traj, max_turns=MAX_TURNS)
            print(f"    -> Done recording trajectory for {exp_id}.")

        except Exception as e:
            print(f"  [Error] {exp_id} failed: {e}")

async def main():
    if not os.path.exists("api.json"): return
    with open("api.json") as f: api_data = json.load(f)
    for domain_id, domain_query in TARGET_DOMAINS.items():
        print(f"\n{'!'*60}\nDOMAIN: {domain_query}\n{'!'*60}")
        await run_trajectory_experiment(domain_id, domain_query, api_data)

if __name__ == "__main__":
    asyncio.run(main())
