#!/usr/bin/env python3
import asyncio
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from agents.clientpool import MultiKeyClientPool
from core.processor import KnowledgeProcessor
from core.judge import DomainJudge

# --- Configuration ---
RESULTS_DIR = "results/model_comparison_l5w5"
DOMAINS = {
    "Deep_Learning": "Deep Learning",
    "Machine_Learning_Systems": "Machine Learning Systems",
    "Probabilistic_Methods": "Probabilistic Methods"
}
DEFAULT_MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct"
]
JUDGE_MODEL = "deepseek-ai/deepseek-v3.1"
FUZZY_CACHE_PATH = "results/eval_fuzzy_cache.json"
AUDIT_CACHE_PATH = "results/eval_audit_cache.json"

STRICT_MATCH = 0.92
FUZZY_LOW = 0.70

def numpy_to_python(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, set): return list(obj)
    return str(obj)

def load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except: return {}
    return {}

def save_cache(cache, path):
    with open(path, "w") as f: json.dump(cache, f, indent=2, ensure_ascii=False)

def get_pair_hash(t1, t2):
    import hashlib
    pair = sorted([t1, t2])
    return hashlib.md5(f"{pair[0]}|{pair[1]}".encode()).hexdigest()

def check_fuzzy_cache(cache, h):
    return cache.get(h)

async def process_model_to_unique(model, domain_id, processor, judge, fuzzy_cache, audit_cache, pbar_pos, lock):
    # Match the folder naming from run_size_comparison.py
    folder_name = model.split("/")[-1]  # "meta/llama-3.1-8b-instruct" -> "llama-3.1-8b-instruct"
    m_dir = os.path.join(RESULTS_DIR, domain_id, folder_name)

    if not os.path.exists(m_dir):
        print(f"      [Warning] Directory not found: {m_dir}")
        return []

    raw_file = os.path.join(m_dir, "P4_Taxonomy_L5W5_raw.json")
    if not os.path.exists(raw_file):
        print(f"      [Warning] Raw file not found in {m_dir}: P4_Taxonomy_L5W5_raw.json")
        return []

    print(f"      📄 Loading raw file... ({raw_file})")
    with open(raw_file, "r") as f: points = json.load(f)
    if not points: return []
    print(f"      📊 Loaded {len(points)} raw points")

    # 1. Strict Deduplication
    print(f"      🔍 Stage 1: Getting embeddings...")
    embs = await processor.get_embeddings(points)
    print(f"      🔍 Stage 1: Computing strict dedup (≥0.92)...")
    norms = np.array(embs)
    norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9)
    
    remaining_indices = []
    seen_mask = [False] * len(points)
    for i in range(len(points)):
        if seen_mask[i]: continue
        remaining_indices.append(i)
        sims = np.dot(norms[i], norms.T)
        for j in range(i+1, len(points)):
            if sims[j] >= STRICT_MATCH: seen_mask[j] = True

    print(f"      ✓ Stage 1: {len(remaining_indices)} points after strict dedup")

    # 2. Fuzzy Deduplication
    print(f"      🔍 Stage 2: Fuzzy dedup (0.70-0.92)...")
    print(f"      🔍 Stage 2: Checking cache for {len(remaining_indices)} candidates...")
    unique_indices = []
    fuzzy_tasks = []
    for i, idx in enumerate(remaining_indices):
        if i == 0:
            unique_indices.append(idx)
            continue
        ancestor_indices = remaining_indices[:i]
        sims = np.dot(norms[idx], norms[ancestor_indices].T)
        max_sub_idx = np.argmax(sims)
        max_ancestor_idx = ancestor_indices[max_sub_idx]

        if sims[max_sub_idx] >= FUZZY_LOW:
            h = get_pair_hash(points[idx], points[max_ancestor_idx])
            # Read without lock (dict read is thread-safe under GIL)
            res = check_fuzzy_cache(fuzzy_cache, h)
            if res is True: continue
            elif res is False: unique_indices.append(idx)
            else: fuzzy_tasks.append((idx, max_ancestor_idx, h))
        else:
            unique_indices.append(idx)

    print(f"      ✓ Cache check done: {len(fuzzy_tasks)} need LLM, {len(unique_indices)} resolved")

    if fuzzy_tasks:
        print(f"      🔍 Stage 2: {len(fuzzy_tasks)} pairs need LLM judge...")
        pbar = tqdm(total=len(fuzzy_tasks), desc=f"      Fuzzy [{model[-8:]}]", position=pbar_pos, leave=False)

        # Batch concurrent execution for speed
        async def judge_and_update(idx, anc_idx, h):
            is_dup = await processor._ask_llm_if_same(points[idx], points[anc_idx])
            async with lock:
                fuzzy_cache[h] = is_dup
            pbar.update(1)
            return idx, is_dup

        judge_tasks = [judge_and_update(idx, anc_idx, h) for idx, anc_idx, h in fuzzy_tasks]
        results = await asyncio.gather(*judge_tasks)

        for idx, is_dup in results:
            if not is_dup: unique_indices.append(idx)

        pbar.close()

    print(f"      ✓ Stage 2: {len(unique_indices)} points after fuzzy dedup")

    # 3. Domain Audit
    print(f"      🔍 Stage 3: Domain audit...")
    final_indices = []
    audit_needed = []
    for idx in unique_indices:
        # Read without lock (dict read is thread-safe under GIL)
        res = audit_cache.get(points[idx])
        if res is True: final_indices.append(idx)
        elif res is False: continue
        else: audit_needed.append(idx)
        
    if audit_needed:
        print(f"      🔍 Stage 3: {len(audit_needed)} points need domain audit...")
        pbar = tqdm(total=len(audit_needed), desc=f"      Audit [{model[-8:]}]", position=pbar_pos, leave=False)
        domain_query = DOMAINS[domain_id]
        results = await judge.check_batch(domain_query, [points[i] for i in audit_needed])
        for idx, is_valid in zip(audit_needed, results):
            async with lock:
                audit_cache[points[idx]] = is_valid
            if is_valid: final_indices.append(idx)
            pbar.update(1)
        pbar.close()

    print(f"      ✓ Stage 3: {len(final_indices)} valid points after audit")

    # Return (raw_count, unique_count, valid_points_list)
    return len(points), len(unique_indices), [points[i] for i in final_indices]

async def build_union_with_dedup(all_points, processor, fuzzy_cache, lock, pbar_pos):
    """Build union set with two-stage deduplication (embedding + LLM judge)"""
    if not all_points:
        return []

    # Stage 1: Strict Deduplication (≥0.92)
    embs = await processor.get_embeddings(all_points)
    norms = np.array(embs)
    norms = norms / (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-9)

    remaining_indices = []
    seen_mask = [False] * len(all_points)
    for i in range(len(all_points)):
        if seen_mask[i]: continue
        remaining_indices.append(i)
        sims = np.dot(norms[i], norms.T)
        for j in range(i+1, len(all_points)):
            if sims[j] >= STRICT_MATCH:
                seen_mask[j] = True

    # Stage 2: Fuzzy Deduplication (0.70-0.92)
    union_indices = []
    fuzzy_tasks = []
    for i, idx in enumerate(remaining_indices):
        if i == 0:
            union_indices.append(idx)
            continue
        ancestor_indices = remaining_indices[:i]
        sims = np.dot(norms[idx], norms[ancestor_indices].T)
        max_sub_idx = np.argmax(sims)
        max_ancestor_idx = ancestor_indices[max_sub_idx]

        if sims[max_sub_idx] >= FUZZY_LOW:
            h = get_pair_hash(all_points[idx], all_points[max_ancestor_idx])
            # Read without lock (dict read is thread-safe under GIL)
            res = check_fuzzy_cache(fuzzy_cache, h)
            if res is True:
                continue
            elif res is False:
                union_indices.append(idx)
            else:
                fuzzy_tasks.append((idx, max_ancestor_idx, h))
        else:
            union_indices.append(idx)

    if fuzzy_tasks:
        pbar = tqdm(total=len(fuzzy_tasks), desc=f"      Union Fuzzy", position=pbar_pos, leave=False)

        # Batch concurrent execution for speed
        async def judge_and_update(idx, anc_idx, h):
            is_dup = await processor._ask_llm_if_same(all_points[idx], all_points[anc_idx])
            async with lock:
                fuzzy_cache[h] = is_dup
            pbar.update(1)
            return idx, is_dup

        judge_tasks = [judge_and_update(idx, anc_idx, h) for idx, anc_idx, h in fuzzy_tasks]
        results = await asyncio.gather(*judge_tasks)

        for idx, is_dup in results:
            if not is_dup:
                union_indices.append(idx)

        pbar.close()

    return [all_points[i] for i in union_indices]

async def compute_unique_discovery(model_points, other_models_points, processor, lock):
    """Compute unique discovery using semantic matching (not string matching)"""
    if not model_points or not other_models_points:
        return len(model_points)

    # Get embeddings for model points
    model_embs = await processor.get_embeddings(model_points)
    model_norms = np.array(model_embs)
    model_norms = model_norms / (np.linalg.norm(model_norms, axis=1, keepdims=True) + 1e-9)

    # Get embeddings for other models' points
    other_embs = await processor.get_embeddings(other_models_points)
    other_norms = np.array(other_embs)
    other_norms = other_norms / (np.linalg.norm(other_norms, axis=1, keepdims=True) + 1e-9)

    # For each model point, check if it's similar to any other model's point
    unique_count = 0
    for i in range(len(model_points)):
        sims = np.dot(model_norms[i], other_norms.T)
        max_sim = np.max(sims) if len(sims) > 0 else 0.0
        # Consider unique if similarity < 0.92 (not semantically matched by others)
        if max_sim < STRICT_MATCH:
            unique_count += 1

    return unique_count

async def process_domain_task(domain_id, domain_query, models, processor, judge, fuzzy_cache, audit_cache, start_pos, lock):
    print(f"\n{'='*80}")
    print(f"📚 Processing Domain: {domain_id}")
    print(f"{'='*80}")

    domain_results = {}

    # Process each model sequentially (within domain, models share embedding cache)
    for i, model in enumerate(models):
        model_name = model.split('/')[-1]
        print(f"\n   🤖 [{i+1}/{len(models)}] Processing {model_name}...", flush=True)
        raw_count, unique_count, valid_points = await process_model_to_unique(model, domain_id, processor, judge, fuzzy_cache, audit_cache, 0, lock)
        domain_results[model] = {
            "raw": raw_count,
            "unique": unique_count,
            "valid_points": valid_points
        }
        print(f"      ✓ {model_name}: Raw={raw_count}, Unique={unique_count}, Valid={len(valid_points)}", flush=True)

    # Build Union with two-stage deduplication
    all_valid_points = []
    for m in models:
        all_valid_points.extend(domain_results[m]["valid_points"])

    if not all_valid_points:
        print(f"   ⚠️  No valid points found for {domain_id}")
        return domain_id, {"models": {}, "union_size": 0}

    print(f"\n   🔄 Building Union set ({len(all_valid_points)} total points)...", flush=True)
    union_texts = await build_union_with_dedup(all_valid_points, processor, fuzzy_cache, lock, 0)
    print(f"      ✓ Union: {len(union_texts)} unique points")

    report = {"models": {}, "union_size": len(union_texts)}

    # Compute metrics for each model
    print(f"\n   📊 Computing metrics...")
    for model in models:
        model_name = model.split('/')[-1]
        model_data = domain_results[model]
        m_pts = model_data["valid_points"]

        raw_count = model_data["raw"]
        unique_count = model_data["unique"]
        valid_count = len(m_pts)
        accuracy = valid_count / unique_count if unique_count > 0 else 0

        # Compute unique discovery (semantic matching)
        other_models_points = []
        for other_model in models:
            if other_model != model:
                other_models_points.extend(domain_results[other_model]["valid_points"])

        unique_disc = await compute_unique_discovery(m_pts, other_models_points, processor, lock)
        recall = valid_count / len(union_texts) if union_texts else 0

        report["models"][model] = {
            "raw": raw_count,
            "unique": unique_count,
            "valid": valid_count,
            "accuracy": accuracy,
            "recall": recall,
            "unique_discovery": unique_disc
        }

        print(f"      {model_name}: Recall={recall:.2%}, Accuracy={accuracy:.2%}, Unique Discovery={unique_disc}")

    print(f"   ✅ Domain {domain_id} completed!\n")
    return domain_id, report

async def main_evaluation():
    import sys
    print("\n" + "="*80, flush=True)
    print("🚀 Llama 3.1 Scaling Law Evaluation", flush=True)
    print("="*80, flush=True)
    print(f"📂 Results Directory: {RESULTS_DIR}", flush=True)
    print(f"📊 Domains: {list(DOMAINS.keys())}", flush=True)
    print(f"🤖 Models: {[m.split('/')[-1] for m in DEFAULT_MODELS]}", flush=True)
    print(f"🔍 Judge Model: {JUDGE_MODEL}", flush=True)
    print("="*80 + "\n", flush=True)

    print("📖 Loading api.json...", flush=True)
    with open("api.json") as f: api_data = json.load(f)
    print("   ✓ api.json loaded\n", flush=True)

    print("📥 Loading caches...", flush=True)
    fuzzy_cache = load_cache(FUZZY_CACHE_PATH)
    print(f"   ✓ Fuzzy cache loaded: {len(fuzzy_cache)} entries", flush=True)
    audit_cache = load_cache(AUDIT_CACHE_PATH)
    print(f"   ✓ Audit cache loaded: {len(audit_cache)} entries\n", flush=True)

    lock = asyncio.Lock()

    print("🔧 Initializing processors...", flush=True)
    client_pool = MultiKeyClientPool(api_keys=api_data["api_keys"])
    print("   ✓ Client pool created", flush=True)
    embed_config = api_data.get("embed_config", {})

    # FIX: Correctly initialize embed_client_pool with local base_url
    print("   🔧 Creating KnowledgeProcessor...", flush=True)
    processor = KnowledgeProcessor(
        client_pool=client_pool,
        judge_model=JUDGE_MODEL,
        embed_model=embed_config.get("model"),
        embed_client_pool=MultiKeyClientPool(api_keys=api_data["api_keys"], base_url=embed_config.get("base_url"))
    )
    print("   ✓ KnowledgeProcessor created", flush=True)
    print("   🔧 Creating DomainJudge...", flush=True)
    judge = DomainJudge(client_pool=client_pool, model=JUDGE_MODEL)
    print("✓ Processors initialized\n", flush=True)

    print("🔄 Starting sequential domain processing (faster than parallel due to API limits)...", flush=True)
    results = []
    for d_id, d_query in DOMAINS.items():
        print(f"\n   • Processing domain: {d_id}", flush=True)
        result = await process_domain_task(d_id, d_query, DEFAULT_MODELS, processor, judge, fuzzy_cache, audit_cache, 0, lock)
        results.append(result)
        # Save cache after each domain to avoid losing progress
        save_cache(fuzzy_cache, FUZZY_CACHE_PATH)
        save_cache(audit_cache, AUDIT_CACHE_PATH)
        print(f"   💾 Cache saved after {d_id}", flush=True)

    print("\n💾 Saving caches...")
    save_cache(fuzzy_cache, FUZZY_CACHE_PATH)
    save_cache(audit_cache, AUDIT_CACHE_PATH)
    print(f"   Fuzzy cache: {len(fuzzy_cache)} entries")
    print(f"   Audit cache: {len(audit_cache)} entries")

    # Save final report (JSON)
    report_path = os.path.join(RESULTS_DIR, "scaling_law_report.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(dict(results), f, indent=2, ensure_ascii=False)

    # Save summary table (TXT)
    summary_path = os.path.join(RESULTS_DIR, "scaling_law_summary.txt")
    with open(summary_path, "w") as f:
        f.write("="*120 + "\n")
        f.write("Llama 3.1 Scaling Law Evaluation Summary\n")
        f.write("="*120 + "\n\n")
        f.write(f"{'Domain':<25} {'Model':<30} {'Recall':>8} {'Accuracy':>10} {'Valid':>8} {'Raw':>8} {'Unique':>8} {'Discovery':>10}\n")
        f.write("-"*120 + "\n")

        for domain_id, domain_report in dict(results).items():
            domain_name = domain_id.replace('_', ' ')
            for model, stats in domain_report["models"].items():
                model_name = model.split('/')[-1]
                f.write(f"{domain_name:<25} {model_name:<30} "
                       f"{stats['recall']:>7.1%} "
                       f"{stats['accuracy']:>9.1%} "
                       f"{stats['valid']:>8} "
                       f"{stats['raw']:>8} "
                       f"{stats['unique']:>8} "
                       f"{stats['unique_discovery']:>10}\n")
            f.write(f"{'':25} Union: {domain_report['union_size']}\n")
            f.write("\n")

    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(json.dumps(dict(results), indent=2))
    print("="*80 + "\n")
    print(f"📄 JSON Report saved to: {report_path}")
    print(f"📄 Summary Table saved to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(main_evaluation())
