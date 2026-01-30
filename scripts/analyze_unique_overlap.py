#!/usr/bin/env python3
"""
Analyze overlap between unique discoveries of different model scales.
Sample random unique discoveries and find their closest matches in other models.
"""
import json
import os
import pickle
import random
import math
from collections import defaultdict

# Configuration
REPORT_PATH = "results/model_comparison_l5w5/scaling_law_report.json"
RESULTS_DIR = "results/model_comparison_l5w5"
OUTPUT_PATH = "results/model_comparison_l5w5/unique_overlap_analysis.txt"

MODELS = [
    "meta/llama-3.1-8b-instruct",
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-405b-instruct"
]

MODEL_LABELS = {
    "meta/llama-3.1-8b-instruct": "8B",
    "meta/llama-3.1-70b-instruct": "70B",
    "meta/llama-3.1-405b-instruct": "405B"
}

DOMAINS = [
    "Deep_Learning",
    "Machine_Learning_Systems",
    "Probabilistic_Methods"
]

SAMPLES_PER_MODEL = 10  # Sample 10 unique discoveries per model per domain


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def load_embeddings(domain, model):
    """Load embeddings for a specific domain and model."""
    # Extract model name after the slash (e.g., "meta/llama-3.1-8b-instruct" -> "llama-3.1-8b-instruct")
    folder_name = model.split("/")[-1]
    emb_path = os.path.join(RESULTS_DIR, domain, folder_name, "P4_Taxonomy_L5W5.emb.pkl")

    if not os.path.exists(emb_path):
        print(f"   ⚠️  Embeddings not found: {emb_path}")
        return {}

    with open(emb_path, "rb") as f:
        return pickle.load(f)


def load_valid_points(domain, model):
    """Load valid points for a specific domain and model."""
    # Extract model name after the slash
    folder_name = model.split("/")[-1]
    raw_path = os.path.join(RESULTS_DIR, domain, folder_name, "P4_Taxonomy_L5W5_raw.json")

    if not os.path.exists(raw_path):
        print(f"   ⚠️  Raw file not found: {raw_path}")
        return []

    with open(raw_path, "r") as f:
        return json.load(f)


def main():
    print("\n" + "=" * 80)
    print("🔍 Analyzing Unique Discovery Overlap Across Model Scales")
    print("=" * 80)

    # Load report to get unique discoveries
    print(f"\n📥 Loading report from {REPORT_PATH}")
    with open(REPORT_PATH, "r") as f:
        report = json.load(f)

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("UNIQUE DISCOVERY OVERLAP ANALYSIS")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("For each model's unique discoveries, we sample randomly and find")
    output_lines.append("the closest match in other models' valid points.")
    output_lines.append("")
    output_lines.append("Similarity thresholds:")
    output_lines.append("  - ≥0.92: Strict match (should have been deduplicated)")
    output_lines.append("  - 0.70-0.92: Fuzzy zone (LLM judge should have checked)")
    output_lines.append("  - <0.70: Likely different concepts")
    output_lines.append("")

    # Process each domain
    for domain in DOMAINS:
        print(f"\n{'=' * 80}")
        print(f"📊 Processing: {domain}")
        print(f"{'=' * 80}")

        output_lines.append("")
        output_lines.append("=" * 80)
        output_lines.append(f"DOMAIN: {domain}")
        output_lines.append("=" * 80)
        output_lines.append("")

        domain_report = report[domain]

        # Load embeddings and valid points for all models
        embeddings = {}
        valid_points = {}

        for model in MODELS:
            print(f"   Loading {MODEL_LABELS[model]}...")
            embeddings[model] = load_embeddings(domain, model)
            valid_points[model] = load_valid_points(domain, model)

        # For each model, sample unique discoveries
        for source_model in MODELS:
            source_label = MODEL_LABELS[source_model]
            model_data = domain_report["models"][source_model]

            print(f"\n   🎯 Analyzing {source_label} unique discoveries...")

            output_lines.append("")
            output_lines.append("-" * 80)
            output_lines.append(f"SOURCE MODEL: {source_label}")
            output_lines.append(f"Total unique discoveries: {model_data['unique_discovery']}")
            output_lines.append("-" * 80)
            output_lines.append("")

            # Get valid points for source model
            source_valid = [p for p in valid_points[source_model] if p in embeddings[source_model]]

            if len(source_valid) == 0:
                print(f"      ⚠️  No valid points with embeddings found")
                continue

            # Sample random points
            sample_size = min(SAMPLES_PER_MODEL, len(source_valid))
            sampled_points = random.sample(source_valid, sample_size)

            for idx, point in enumerate(sampled_points, 1):
                output_lines.append(f"[{source_label} Sample #{idx}]")
                output_lines.append(f"{point}")
                output_lines.append("")

                source_emb = embeddings[source_model][point]

                # Find closest match in each other model
                for target_model in MODELS:
                    if target_model == source_model:
                        continue

                    target_label = MODEL_LABELS[target_model]
                    target_valid = [p for p in valid_points[target_model] if p in embeddings[target_model]]

                    if len(target_valid) == 0:
                        continue

                    # Compute similarities
                    max_sim = -1
                    closest_point = None

                    for target_point in target_valid:
                        target_emb = embeddings[target_model][target_point]
                        sim = cosine_similarity(source_emb, target_emb)

                        if sim > max_sim:
                            max_sim = sim
                            closest_point = target_point

                    # Format similarity with color indicators
                    if max_sim >= 0.92:
                        sim_label = f"{max_sim:.4f} [STRICT MATCH - Should be deduplicated!]"
                    elif max_sim >= 0.70:
                        sim_label = f"{max_sim:.4f} [FUZZY ZONE - Check LLM judge]"
                    else:
                        sim_label = f"{max_sim:.4f} [Likely different]"

                    output_lines.append(f"  Closest in {target_label}: (sim={sim_label})")
                    output_lines.append(f"  {closest_point}")
                    output_lines.append("")

                output_lines.append("")

    # Save output
    print(f"\n💾 Saving analysis to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    print(f"\n📄 Output saved to: {OUTPUT_PATH}")
    print("\nPlease review the output and for each sample, judge whether:")
    print("  - SAME: Describes the same concept/knowledge")
    print("  - DIFFERENT: Truly different knowledge")
    print("  - PARTIAL: Overlapping but with different focus/detail level")
    print()


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
