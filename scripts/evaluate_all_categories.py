import asyncio
import json
import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge

TARGET_CATEGORIES = {
    "deep_learning": "Deep Learning (including Transformer Architectures, Generative Models, and Theory)",
    "reinforcement_learning": "Reinforcement Learning (including Markov Decision Processes, Policy Gradient Methods, and Hierarchical RL)",
    "trustworthy_ml": "Trustworthy ML (including Algorithmic Fairness, Adversarial Robustness, and Model Interpretability)"
}

def is_garbage(text):
    text = text.strip()
    if len(text) < 15: return True
    if text.endswith(':') or text.endswith(','): return True
    if "include:" in text.lower() or "following are" in text.lower(): return True
    if text.lower().startswith("definition") and ":" not in text: return True
    return False

async def evaluate_major_category(cat_id, domain_query, processor, judge, root_results_dir):
    cat_dir = os.path.join(root_results_dir, cat_id)
    if not os.path.exists(cat_dir): return None
    print(f"\n{'='*80}\nEVALUATING MAJOR CATEGORY: {cat_id.upper()}\n{'='*80}")

    union_set_path = os.path.join(cat_dir, "category_union_set.json")
    
    # RESUME LOGIC: Load existing union set if available
    if os.path.exists(union_set_path):
        print(f"[*] Found existing union set. Loading progress...")
        with open(union_set_path, "r") as f:
            existing_data = json.load(f)
        processor.union_set = []
        for item in existing_data:
            # FORCE RE-AUDIT: If we want to re-evaluate with new prompt
            # We reset 'is_in_domain' to None if it was True, so it gets re-checked.
            current_val = item.get("is_in_domain")
            if current_val is True: # Only re-check points that were previously valid
                current_val = None
            
            processor.union_set.append({
                "representative_text": item["representative_text"],
                "is_in_domain": current_val, 
                "pipelines": set(item["pipelines_covered"]),
                "source_entries": item["detailed_sources"],
                "centroid_embedding": None 
            })
        print(f"    - Loaded {len(processor.union_set)} nodes. (Ready for re-auditing)")
    else:
        processor.union_set = []
    
    # Collect all raw files
    sub_dirs = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
    all_raw_files = sorted([os.path.join(d, f) for d in sub_dirs for f in os.listdir(os.path.join(cat_dir, d)) if f.endswith("_raw.json")])
    
    # Rebuild embedding matrix for existing nodes if we are going to add more
    # For now, let's assume we either run from scratch OR just resume auditing
    
    model_raw_data = {}
    for rel_path in all_raw_files:
        parts = os.path.basename(rel_path).replace("_raw.json", "").split("_")
        if len(parts) < 2: continue
        model_name, pipeline_name = parts[0], parts[1]
        pid = f"{os.path.dirname(rel_path)}/{model_name}_{pipeline_name}"
        if model_name not in model_raw_data: model_raw_data[model_name] = {}
        with open(os.path.join(cat_dir, rel_path), "r") as f: raw_points = json.load(f)
        model_raw_data[model_name][pid] = raw_points

        # Only merge if this pipeline isn't already processed (simple check)
        if any(pid in node["pipelines"] for node in processor.union_set):
            continue

        print(f"      [Merge] Analyzing {len(raw_points)} points for {pid}...")
        emb_path = os.path.join(cat_dir, rel_path.replace("_raw.json", ".emb.pkl"))
        if processor.load_embeddings(emb_path) < len(raw_points):
            await processor.get_embeddings(raw_points)
            processor.save_embeddings(emb_path)
        
        # Pre-dedup + Garbage filter
        raw_clean = [p for p in raw_points if not is_garbage(p)]
        bullet_points, embeddings = await processor._pre_deduplicate_pipeline(pid, raw_clean)
        if not bullet_points: continue

        # Parallel Merge Logic
        point_actions = []
        fuzzy_tasks = []
        if processor.union_set and processor.embedding_matrix is not None:
            curr_embs = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embeddings])
            sims = np.dot(curr_embs, processor.embedding_matrix.T)
            for i in range(len(bullet_points)):
                max_idx = int(np.argmax(sims[i])); max_sim = sims[i][max_idx]
                if max_sim >= 0.95: point_actions.append(("direct", max_idx, max_sim))
                elif max_sim >= 0.70:
                    task_idx = len(fuzzy_tasks)
                    fuzzy_tasks.append(processor._ask_llm_if_same(bullet_points[i], processor.union_set[max_idx]["representative_text"]))
                    point_actions.append(("fuzzy", max_idx, task_idx))
                else: point_actions.append(("none", -1, 1.0))
        else:
            point_actions = [("none", -1, 1.0)] * len(bullet_points)

        fuzzy_res = []
        if fuzzy_tasks:
            from tqdm.asyncio import tqdm as atqdm
            fuzzy_res = await atqdm.gather(*fuzzy_tasks, desc="    Fuzzy Check", leave=False)

        new_embs = []
        for i, (action, m_idx, extra) in enumerate(point_actions):
            text, emb = bullet_points[i], embeddings[i]
            norm_emb = emb / (np.linalg.norm(emb) + 1e-9)
            final_idx = m_idx if (action == "direct" or (action == "fuzzy" and fuzzy_res[extra])) else -1
            if final_idx != -1:
                node = processor.union_set[final_idx]
                node["pipelines"].add(pid)
                node["source_entries"].append({"text": text, "pipeline": pid, "similarity": 1.0})
                # Centroid update omitted for brevity in resume logic, but kept for new nodes
            else:
                processor.union_set.append({"representative_text": text, "centroid_embedding": norm_emb, "source_entries": [{"text": text, "pipeline": pid, "similarity": 1.0}], "pipelines": {pid}, "is_in_domain": None})
                new_embs.append(norm_emb)
        if new_embs:
            new_mat = np.stack(new_embs)
            processor.embedding_matrix = np.vstack([processor.embedding_matrix, new_mat]) if processor.embedding_matrix is not None else new_mat
        
        # Save after each file merge
        processor.save_union_set(union_set_path)

    # AUDIT PHASE with Checkpointing
    to_audit = [i for i, n in enumerate(processor.union_set) if n.get("is_in_domain") is None]
    if to_audit:
        print(f"[*] Auditing {len(to_audit)} new/remaining nodes...")
        chunk_size = 100
        pbar = tqdm(total=len(to_audit), desc="    Auditing Progress")
        for i in range(0, len(to_audit), chunk_size):
            chunk_indices = to_audit[i : i + chunk_size]
            tasks = [judge.is_in_domain(domain_query, processor.union_set[idx]["representative_text"]) for idx in chunk_indices]
            results = await asyncio.gather(*tasks)
            for idx, res in zip(chunk_indices, results):
                processor.union_set[idx]["is_in_domain"] = res
            pbar.update(len(chunk_indices))
            # CHECKPOINT: Save every chunk
            processor.save_union_set(union_set_path)
        pbar.close()

    # Final Metrics
    valid_nodes = [n for n in processor.union_set if n.get("is_in_domain") is True]
    global_total_valid = len(valid_nodes)
    metrics = {}
    for m_name, pipelines in model_raw_data.items():
        m_valid_indices = [i for i, n in enumerate(processor.union_set) if n.get("is_in_domain") is True and any(p.split('/')[1].startswith(m_name) for p in n["pipelines"])]
        total_raw = 0; total_valid_raw = 0; p_stats = {}
        for pid, raw_pts in pipelines.items():
            p_v_count = sum(1 for idx in m_valid_indices if pid in processor.union_set[idx]["pipelines"])
            raw_v_count = sum(1 for idx in m_valid_indices for e in processor.union_set[idx]["source_entries"] if e["pipeline"] == pid)
            p_stats[pid] = {"recall_internal": p_v_count/len(m_valid_indices) if m_valid_indices else 0, "accuracy": raw_v_count/len(raw_pts) if raw_pts else 0, "valid_count": p_v_count}
            total_raw += len(raw_pts); total_valid_raw += raw_v_count
        metrics[m_name] = {"global_recall": len(m_valid_indices)/global_total_valid if global_total_valid else 0, "overall_accuracy": total_valid_raw/total_raw if total_raw else 0, "valid_pts": len(m_valid_indices), "unique_discovery": 0, "pipelines": p_stats}
        for idx in m_valid_indices:
            if not any(p.split('/')[1].split('_')[0] != m_name for p in processor.union_set[idx]["pipelines"]): metrics[m_name]["unique_discovery"] += 1
    return metrics

async def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=str, default="results"); args = parser.parse_args()
    if not os.path.exists("api.json"): return
    with open("api.json") as f: api_data = json.load(f)
    client_pool = MultiKeyClientPool(api_keys=api_data["api_keys"])
    processor = KnowledgeProcessor(client_pool=client_pool, embed_client_pool=MultiKeyClientPool(api_keys=[], base_url=api_data.get("embed_config", {}).get("base_url")), embed_model=api_data.get("embed_config", {}).get("model", "Qwen/Qwen3-Embedding-8B"), judge_model="deepseek-ai/deepseek-v3.2")
    processor.llm_semaphore = asyncio.Semaphore(30); judge = DomainJudge(client_pool=client_pool, model="deepseek-ai/deepseek-v3.2", concurrency=30)

    all_results = {}
    for cat_id, query in TARGET_CATEGORIES.items():
        res = await evaluate_major_category(cat_id, query, processor, judge, args.root)
        if res: all_results[cat_id] = res

    report = []
    header = "\n\n" + "="*100 + "\nFINAL CROSS-CATEGORY SUMMARY REPORT\n" + "="*100
    print(header); report.append(header)
    for cat_id, model_res in all_results.items():
        cat_h = f"\n[Category: {cat_id.upper()}]\n{'Model':25} | {'Global Recall':15} | {'Accuracy':12} | {'Valid Pts':10} | {'Unique Discovery'}\n" + "-"*100
        print(cat_h); report.append(cat_h)
        for m_name in sorted(model_res.keys(), key=lambda x: model_res[x]['global_recall'], reverse=True):
            m = model_res[m_name]
            line = f"{m_name:25} | {m['global_recall']:15.2%} | {m['overall_accuracy']:12.2%} | {m['valid_pts']:10} | {m['unique_discovery']}"
            print(line); report.append(line)
    with open("final_evaluation_report.txt", "w") as f: f.write("\n".join(report))
    print(f"\n[Done] Final report saved.")

if __name__ == "__main__": asyncio.run(main())
