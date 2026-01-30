import asyncio
import json
import os
import pickle
import numpy as np
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge

# --- Configuration ---
TRAJ_DIR = "results/pareto_curves_405b"
# Anchor for normalization (Still using the final state of L2W2 as 1.0)
BASELINE_ID = "P4_Taxonomy_L2W2"
DOMAINS = ["Transformer_Architectures", "Policy_Gradient_Methods", "Algorithmic_Fairness"]
AUDIT_CACHE_PATH = "results/trajectory_audit_cache.json"
JUDGE_MODEL = "deepseek-ai/deepseek-v3.1"

def is_garbage(text):
    text = text.strip()
    if len(text) < 15: return True
    if text.endswith(':') or text.endswith(','): return True
    if "include:" in text.lower() or "following are" in text.lower(): return True
    return False

def load_cache():
    if os.path.exists(AUDIT_CACHE_PATH):
        try:
            with open(AUDIT_CACHE_PATH, "r") as f: return json.load(f)
        except Exception: return {}
    return {}

def save_cache(cache):
    with open(AUDIT_CACHE_PATH, "w") as f: json.dump(cache, f, indent=2)

async def get_valid_unique_count_for_points(points, domain_query, processor, judge, cache, exp_id, turn):
    """
    Evaluates a specific snapshot of points. Uses cache to avoid re-auditing same text.
    """
    clean_points = [p for p in points if not is_garbage(p)]
    if not clean_points: return 0

    # 1. Deduplicate the current snapshot
    embeddings = await processor.get_embeddings(clean_points)
    norm_embs = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embeddings])
    
    unique_points = []
    unique_indices = []
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    for i in range(len(clean_points)):
        if not unique_indices or np.max(sim_matrix[i, unique_indices]) < 0.92:
            unique_indices.append(i)
            unique_points.append(clean_points[i])

    # 2. Audit unique points (with cache)
    to_audit = [p for p in unique_points if p not in cache]
    if to_audit:
        print(f"      [Audit] {exp_id} T{turn}: New points to audit: {len(to_audit)}")
        results = await judge.check_batch(domain_query, to_audit)
        for p, r in zip(to_audit, results):
            cache[p] = r
        save_cache(cache)

    valid_count = sum(1 for p in unique_points if cache.get(p, False))
    return valid_count

async def main():
    if not os.path.exists("api.json"): return
    with open("api.json") as f: api_data = json.load(f)
    client_pool = MultiKeyClientPool(api_keys=api_data["api_keys"])
    embed_config = api_data.get("embed_config", {})
    processor = KnowledgeProcessor(
        client_pool=client_pool, 
        embed_client_pool=MultiKeyClientPool(api_keys=api_data["api_keys"], base_url=embed_config.get("base_url")),
        embed_model=embed_config.get("model"),
        judge_model=JUDGE_MODEL
    )
    judge = DomainJudge(client_pool=client_pool, model=JUDGE_MODEL, concurrency=30)
    audit_cache = load_cache()

    plot_data = {}

    for domain in DOMAINS:
        domain_query = domain.replace("_", " ")
        print(f"\n>>> Processing Trajectories for Domain: {domain}")
        
        # 1. First, find the baseline final count for this domain
        baseline_file = os.path.join(TRAJ_DIR, domain, f"{BASELINE_ID}.trajectory.json")
        baseline_emb = os.path.join(TRAJ_DIR, domain, f"{BASELINE_ID}.emb.pkl")
        if not os.path.exists(baseline_file): 
            print(f"!!! Baseline trajectory missing for {domain}")
            continue
        
        processor.load_embeddings(baseline_emb)
        with open(baseline_file, "r") as f:
            baseline_traj = json.load(f)
        
        # FIND THE ABSOLUTE LAST TURN IN THE FILE
        actual_last_snapshot = baseline_traj[-1]
        last_turn_val = actual_last_snapshot['meta']['turn']
        print(f"    [Anchor Check] Baseline file has {len(baseline_traj)} snapshots. Last turn: {last_turn_val}")
        
        final_points = actual_last_snapshot["points"]
        print(f"    [Anchor] Calculating final valid count for baseline (Turn {last_turn_val})...")
        baseline_final_valid = await get_valid_unique_count_for_points(
            final_points, domain_query, processor, judge, audit_cache, BASELINE_ID, "Final"
        )
        print(f"    [Anchor] Baseline final valid points: {baseline_final_valid}")

        # 2. Evaluate each trajectory
        domain_curves = {}
        traj_files = [f for f in os.listdir(os.path.join(TRAJ_DIR, domain)) if f.endswith(".trajectory.json")]
        
        for tf in sorted(traj_files):
            exp_id = tf.replace(".trajectory.json", "")
            print(f"    [Curve] Calculating trajectory for {exp_id}...")
            
            with open(os.path.join(TRAJ_DIR, domain, tf), "r") as f:
                trajectory = json.load(f)
            
            emb_path = os.path.join(TRAJ_DIR, domain, f"{exp_id}.emb.pkl")
            processor.load_embeddings(emb_path)
            
            curve_points = []
            current_max_valid = 0
            
            for step in trajectory:
                turn = step["meta"]["turn"]
                tokens = step["meta"]["cumulative_tokens"]
                
                # Get valid count for this snapshot
                valid_count = await get_valid_unique_count_for_points(
                    step["points"], domain_query, processor, judge, audit_cache, exp_id, turn
                )
                
                # Enforce Monotonicity (knowledge discovered is never lost)
                current_max_valid = max(current_max_valid, valid_count)
                
                ratio = current_max_valid / baseline_final_valid if baseline_final_valid > 0 else 0
                
                curve_points.append({
                    "turn": turn,
                    "cumulative_tokens": tokens,
                    "valid_points": current_max_valid,
                    "yield_ratio": round(ratio, 4)
                })
                print(f"      Turn {turn}: Tokens={tokens}, Ratio={ratio:.4f}")
            
            domain_curves[exp_id] = curve_points

        plot_data[domain] = domain_curves

    # Save final curve data
    with open("results/all_domains_pareto_data.json", "w") as f:
        json.dump(plot_data, f, indent=2)
    print("\n[*] Trajectory evaluation complete. Data saved to results/all_domains_pareto_data.json")

if __name__ == "__main__":
    asyncio.run(main())
