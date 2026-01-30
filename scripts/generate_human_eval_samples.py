#!/usr/bin/env python3
"""
Generate human evaluation samples for LLM-as-Judge validation.

This script samples from the extracted knowledge points to create annotation files for:
1. Deduplication Judge: Are two knowledge points semantically the same?
2. Domain Audit Judge: Is this a valid domain knowledge point?

Output: CSV files ready for human annotation.
"""

import json
import random
import csv
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
SAMPLE_SIZE_DEDUP = 150  # Number of pairs for dedup evaluation
SAMPLE_SIZE_AUDIT = 150  # Number of points for audit evaluation
RANDOM_SEED = 42

# Similarity thresholds (match the evaluation pipeline)
STRICT_THRESHOLD = 0.92
FUZZY_LOW = 0.70

# Data sources - use cross_series as it has the most recent data
DATA_DIR = Path("results/cross_series_7b")
AUDIT_CACHE = Path("results/cross_series_audit_cache.json")
OUTPUT_DIR = Path("human_eval_samples")

def load_trajectories():
    """Load all trajectory data and embeddings."""
    all_points = []
    all_embeddings = []

    for domain_dir in DATA_DIR.iterdir():
        if not domain_dir.is_dir():
            continue
        domain = domain_dir.name

        for model_dir in domain_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            # Load raw points
            raw_file = model_dir / "P4_Taxonomy_L5W5_raw.json"
            emb_file = model_dir / "P4_Taxonomy_L5W5.emb.pkl"

            if raw_file.exists():
                with open(raw_file) as f:
                    points = json.load(f)

                # Load embeddings if available
                embeddings_data = None
                if emb_file.exists():
                    with open(emb_file, "rb") as f:
                        embeddings_data = pickle.load(f)
                    # Debug: print type on first load
                    if len(all_points) == 0:
                        print(f"  Embedding type: {type(embeddings_data)}")
                        if isinstance(embeddings_data, dict):
                            keys = list(embeddings_data.keys())[:3]
                            print(f"  Sample keys: {keys}")

                for i, pt in enumerate(points):
                    emb = None
                    if embeddings_data is not None:
                        if isinstance(embeddings_data, dict):
                            # Dictionary format: key is the text
                            emb = embeddings_data.get(pt)
                        elif isinstance(embeddings_data, (list, np.ndarray)) and i < len(embeddings_data):
                            emb = embeddings_data[i]

                    all_points.append({
                        "text": pt,
                        "domain": domain,
                        "model": model,
                        "embedding": emb
                    })
                    if emb is not None:
                        all_embeddings.append(emb)

    # Build index mapping for points with embeddings
    emb_indices = [i for i, pt in enumerate(all_points) if pt["embedding"] is not None]
    print(f"  Points with embeddings: {len(emb_indices)}/{len(all_points)}")

    return all_points, np.array(all_embeddings) if all_embeddings else None

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def sample_dedup_pairs_from_cache(all_points, fuzzy_cache, n_samples):
    """
    Sample pairs for deduplication evaluation from LLM-judged cache.
    Strategy: Find pairs in fuzzy zone (0.70-0.92) that have LLM judgments,
    balanced between YES (same) and NO (different) judgments.
    """
    valid_indices = [i for i, pt in enumerate(all_points) if pt["embedding"] is not None]
    print(f"Points with valid embeddings: {len(valid_indices)}")

    # Group points by domain for efficiency
    domain_points = defaultdict(list)
    for idx in valid_indices:
        domain_points[all_points[idx]["domain"]].append(idx)

    judged_yes = []  # LLM said "same"
    judged_no = []   # LLM said "different"

    print("Finding fuzzy-zone pairs with cached LLM judgments...")

    for domain, indices in domain_points.items():
        n_pts = len(indices)
        if n_pts < 2:
            continue

        print(f"  Processing {domain} ({n_pts} points)...")

        # Build embedding matrix for this domain
        embeddings = np.array([all_points[idx]["embedding"] for idx in indices])
        # Normalize for efficient cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normed = embeddings / norms

        # Find pairs in fuzzy zone using batched similarity computation
        batch_size = 500
        for start in range(0, n_pts, batch_size):
            end = min(start + batch_size, n_pts)
            # Compute similarity of batch against all points
            batch_sims = np.dot(embeddings_normed[start:end], embeddings_normed.T)

            for bi, i in enumerate(range(start, end)):
                for j in range(i + 1, n_pts):
                    sim = batch_sims[bi, j]
                    if FUZZY_LOW <= sim < STRICT_THRESHOLD:
                        # Check if this pair was judged
                        t1 = all_points[indices[i]]["text"]
                        t2 = all_points[indices[j]]["text"]
                        pair_hash = get_pair_hash(t1, t2)

                        if pair_hash in fuzzy_cache:
                            entry = (indices[i], indices[j], float(sim), fuzzy_cache[pair_hash])
                            if fuzzy_cache[pair_hash]:
                                judged_yes.append(entry)
                            else:
                                judged_no.append(entry)

            # Early exit if we have enough
            if len(judged_yes) >= n_samples * 2 and len(judged_no) >= n_samples * 2:
                break

        print(f"    Found {len(judged_yes)} YES + {len(judged_no)} NO so far")
        if len(judged_yes) >= n_samples * 2 and len(judged_no) >= n_samples * 2:
            break

    print(f"Total found: {len(judged_yes)} YES (same) + {len(judged_no)} NO (different) judged pairs")

    # Balance: 50% YES, 50% NO
    random.shuffle(judged_yes)
    random.shuffle(judged_no)
    n_each = n_samples // 2

    selected = []
    selected += judged_yes[:n_each]
    selected += judged_no[:n_each]
    random.shuffle(selected)

    # Return as (idx1, idx2, sim) tuples, with judgment stored separately
    return [(e[0], e[1], e[2]) for e in selected], {get_pair_hash(all_points[e[0]]["text"], all_points[e[1]]["text"]): e[3] for e in selected}

def sample_audit_points(all_points, audit_cache, n_samples):
    """
    Sample points for domain audit evaluation.
    Strategy: Balance between points marked valid and invalid by LLM.
    """
    valid_points = []
    invalid_points = []
    unknown_points = []

    for pt in all_points:
        text = pt["text"]
        if text in audit_cache:
            if audit_cache[text]:
                valid_points.append(pt)
            else:
                invalid_points.append(pt)
        else:
            unknown_points.append(pt)

    print(f"Audit cache: {len(valid_points)} valid, {len(invalid_points)} invalid, {len(unknown_points)} unknown")

    # Sample balanced: 50% valid, 50% invalid (from LLM's perspective)
    random.shuffle(valid_points)
    random.shuffle(invalid_points)

    n_each = n_samples // 2
    selected = valid_points[:n_each] + invalid_points[:n_each]
    random.shuffle(selected)

    return selected[:n_samples]

def get_pair_hash(t1, t2):
    """Compute hash for a pair of texts (same as evaluation code)."""
    import hashlib
    pair = sorted([t1, t2])
    return hashlib.md5(f"{pair[0]}|{pair[1]}".encode()).hexdigest()

def load_fuzzy_cache():
    """Load all fuzzy caches."""
    cache = {}
    cache_files = [
        "results/cross_series_fuzzy_cache.json",
        "results/eval_fuzzy_cache.json",
        "results/evolution_fuzzy_cache.json",
    ]
    for path in cache_files:
        if Path(path).exists():
            with open(path) as f:
                cache.update(json.load(f))
    return cache

def clean_text_for_excel(text):
    """Clean text to prevent Excel formula interpretation."""
    # Remove leading bullet points (- or *)
    text = text.strip()
    while text.startswith('-') or text.startswith('*'):
        text = text[1:].strip()
    return text

def write_dedup_csv(pairs, all_points, fuzzy_cache, output_path):
    """Write deduplication annotation CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "domain",
            "point_a", "point_b",
            "embedding_similarity",
            "llm_judgment",
            "human_judgment (YES=same, NO=different)",
            "confidence (1-5)",
            "notes"
        ])

        for idx, (i, j, sim) in enumerate(pairs):
            text_a_raw = all_points[i]["text"]
            text_b_raw = all_points[j]["text"]
            text_a = clean_text_for_excel(text_a_raw)
            text_b = clean_text_for_excel(text_b_raw)

            # Look up LLM judgment from cache (use raw text for hash)
            pair_hash = get_pair_hash(text_a_raw, text_b_raw)
            llm_result = fuzzy_cache.get(pair_hash)
            if llm_result is not None:
                llm_judgment = "YES (same)" if llm_result else "NO (different)"
            else:
                llm_judgment = "not_judged"

            writer.writerow([
                idx + 1,
                all_points[i]["domain"],
                text_a,
                text_b,
                f"{sim:.3f}",
                llm_judgment,
                "",  # human judgment
                "",  # confidence
                ""   # notes
            ])

    print(f"Wrote {len(pairs)} dedup pairs to {output_path}")

def write_audit_csv(points, audit_cache, output_path):
    """Write domain audit annotation CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "domain",
            "knowledge_point",
            "llm_judgment",
            "human_judgment (YES=valid, NO=invalid)",
            "confidence (1-5)",
            "invalid_reason (if NO: meta/generic/fragment/incorrect/other)",
            "notes"
        ])

        for idx, pt in enumerate(points):
            llm_result = audit_cache.get(pt["text"], "unknown")
            if isinstance(llm_result, bool):
                llm_result = "valid" if llm_result else "invalid"

            writer.writerow([
                idx + 1,
                pt["domain"],
                clean_text_for_excel(pt["text"]),
                llm_result,
                "",  # human judgment
                "",  # confidence
                "",  # invalid reason
                ""   # notes
            ])

    print(f"Wrote {len(points)} audit points to {output_path}")

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading trajectories...")
    all_points, embeddings = load_trajectories()
    print(f"Loaded {len(all_points)} points total")

    # Load audit cache
    audit_cache = {}
    if AUDIT_CACHE.exists():
        with open(AUDIT_CACHE) as f:
            audit_cache = json.load(f)
        print(f"Loaded audit cache with {len(audit_cache)} entries")

    # Load fuzzy cache for LLM judgments
    fuzzy_cache = load_fuzzy_cache()
    print(f"Loaded fuzzy cache with {len(fuzzy_cache)} entries")

    # Sample dedup pairs from LLM-judged cache (balanced YES/NO)
    print("\n=== Sampling Deduplication Pairs (from LLM-judged cache) ===")
    dedup_pairs, pair_judgments = sample_dedup_pairs_from_cache(all_points, fuzzy_cache, SAMPLE_SIZE_DEDUP)
    # Merge pair_judgments into fuzzy_cache for lookup
    fuzzy_cache.update(pair_judgments)
    write_dedup_csv(dedup_pairs, all_points, fuzzy_cache, OUTPUT_DIR / "dedup_annotation.csv")

    # Sample audit points
    print("\n=== Sampling Audit Points ===")
    audit_points = sample_audit_points(all_points, audit_cache, SAMPLE_SIZE_AUDIT)
    write_audit_csv(audit_points, audit_cache, OUTPUT_DIR / "audit_annotation.csv")

    print(f"\n=== Done ===")
    print(f"Output files in: {OUTPUT_DIR}/")
    print(f"  - dedup_annotation.csv: {SAMPLE_SIZE_DEDUP} pairs for deduplication judgment")
    print(f"  - audit_annotation.csv: {SAMPLE_SIZE_AUDIT} points for domain audit")
    print(f"\nAnnotation instructions:")
    print(f"  Dedup: Judge if two points convey the SAME knowledge (YES) or DIFFERENT (NO)")
    print(f"  Audit: Judge if the point is VALID domain knowledge (YES) or INVALID (NO)")
    print(f"         Invalid reasons: meta-statement, generic fluff, fragment, factually incorrect")

if __name__ == "__main__":
    main()
