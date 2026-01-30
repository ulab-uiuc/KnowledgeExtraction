#!/usr/bin/env python3
"""
Plot knowledge growth curves for cross-series experiment.
Shows how Recall and Accuracy evolve with cumulative token cost for each model.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- ICML-compatible font settings (9-10pt for labels/ticks) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 10,
    'mathtext.fontset': 'stix',
})

# --- Configuration ---
RESULTS_DIR = "results/cross_series_7b"
REPORT_PATH = os.path.join(RESULTS_DIR, "cross_series_report.json")
AUDIT_CACHE_PATH = "results/cross_series_audit_cache.json"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
GROWTH_DATA_PATH = os.path.join(RESULTS_DIR, "growth_data.json")
COMBINED_OUTPUT = "paper/img/series_curve.pdf"

DOMAINS = {
    "Deep_Learning": "Deep Learning",
    "Machine_Learning_Systems": "Machine Learning Systems",
    "Probabilistic_Methods": "Probabilistic Methods"
}

MODELS = [
    "meta/llama-3.1-8b-instruct",
    "qwen/qwen2.5-7b-instruct",
    "deepseek-ai/deepseek-r1-distill-qwen-7b",
]

MODEL_DISPLAY_NAMES = {
    "meta/llama-3.1-8b-instruct": "Llama 3.1 8B",
    "qwen/qwen2.5-7b-instruct": "Qwen 2.5 7B",
    "deepseek-ai/deepseek-r1-distill-qwen-7b": "DeepSeek R1 Distill 7B",
}

MODEL_COLORS = {
    "meta/llama-3.1-8b-instruct": "#1f77b4",  # Blue
    "qwen/qwen2.5-7b-instruct": "#ff7f0e",    # Orange
    "deepseek-ai/deepseek-r1-distill-qwen-7b": "#2ca02c",  # Green
}


def load_json(path):
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_growth_curve(trajectory, audit_cache, union_size):
    """
    Compute per-turn metrics by backtracking from final audit.

    Args:
        trajectory: List of turn snapshots
        audit_cache: Dict mapping point text to validity (True/False)
        union_size: Size of the union set for recall calculation

    Returns:
        List of dicts with keys: turn, tokens, valid, total, recall, accuracy
    """
    growth_curve = []

    # Skip turn 0 (initialization), process turns 1 onwards
    for snapshot in trajectory[1:]:
        turn = snapshot["meta"]["turn"]
        points_until_t = snapshot["points"]
        tokens_t = snapshot["meta"]["cumulative_tokens"]

        # Count valid points by checking audit cache
        valid_points = [p for p in points_until_t if audit_cache.get(p, False)]

        total_count = len(points_until_t)
        valid_count = len(valid_points)

        # Calculate metrics
        recall = valid_count / union_size if union_size > 0 else 0
        accuracy = valid_count / total_count if total_count > 0 else 0

        growth_curve.append({
            "turn": turn,
            "tokens": tokens_t,
            "valid": valid_count,
            "total": total_count,
            "recall": recall,
            "accuracy": accuracy
        })

    return growth_curve


def plot_metric_growth(domain_id, domain_data, metric_name, ylabel, filename):
    """
    Plot a single metric (recall or accuracy) vs tokens.

    Args:
        domain_id: Domain identifier
        domain_data: Dict mapping model -> growth_curve
        metric_name: "recall" or "accuracy"
        ylabel: Y-axis label
        filename: Output filename
    """
    plt.figure(figsize=(3.5, 2.5))

    for model in MODELS:
        if model not in domain_data:
            continue

        curve = domain_data[model]
        if not curve:
            continue

        tokens = [d["tokens"] for d in curve]
        values = [d[metric_name] for d in curve]

        plt.plot(
            tokens,
            values,
            marker='o',
            linewidth=1.5,
            markersize=4,
            label=MODEL_DISPLAY_NAMES[model],
            color=MODEL_COLORS[model]
        )

    plt.xlabel("Cumulative Tokens")
    plt.ylabel(ylabel)
    plt.title(f"{DOMAINS[domain_id]}")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: {filename}")


def plot_combined_grid(all_growth_data, output_path):
    """
    Plot combined 2x3 grid: top row = Accuracy, bottom row = Recall.
    Columns are ordered by domain.

    Args:
        all_growth_data: Dict mapping domain_id -> {model -> growth_curve}
        output_path: Output file path
    """
    domain_order = ["Deep_Learning", "Machine_Learning_Systems", "Probabilistic_Methods"]

    # Double column ICML figure size (same as qwen plot)
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 3.0))

    metrics = [("accuracy", "Accuracy"), ("recall", "Recall")]

    for row_idx, (metric_name, metric_label) in enumerate(metrics):
        for col_idx, domain_id in enumerate(domain_order):
            ax = axes[row_idx, col_idx]

            if domain_id not in all_growth_data:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue

            domain_data = all_growth_data[domain_id]

            for model in MODELS:
                if model not in domain_data:
                    continue

                curve = domain_data[model]
                if not curve:
                    continue

                tokens = [d["tokens"] for d in curve]
                values = [d[metric_name] for d in curve]

                ax.plot(
                    tokens,
                    values,
                    marker='o',
                    linewidth=1.8,
                    markersize=5,
                    markevery=max(1, len(tokens) // 5),  # Show ~5 markers
                    label=MODEL_DISPLAY_NAMES[model],
                    color=MODEL_COLORS[model]
                )

            ax.grid(True, alpha=0.3)

            # Column titles (domain names) only on top row
            if row_idx == 0:
                ax.set_title(DOMAINS[domain_id], fontweight='bold')

            # Row labels (metric names) only on left column
            if col_idx == 0:
                ax.set_ylabel(metric_label)

            # X-axis labels only on bottom row
            if row_idx == 1:
                ax.set_xlabel("Tokens")
            else:
                ax.set_xticklabels([])

            # Let matplotlib auto-scale y-axis for best visualization
            # (no fixed limits - shows relative changes better)

            # 统一纵坐标小数位数（2位）
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # Add single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(MODELS), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.14)  # 减小子图间隔
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved combined grid: {output_path}")


def main():
    print("\n" + "=" * 80)
    print("📈 Plotting Cross-Series Knowledge Growth Curves")
    print("=" * 80)

    # Load report and audit cache
    print("\n📥 Loading data...")
    if not os.path.exists(REPORT_PATH):
        print(f"   ❌ Error: Report not found at {REPORT_PATH}")
        print("   Please run evaluate_cross_series.py first!")
        return

    if not os.path.exists(AUDIT_CACHE_PATH):
        print(f"   ❌ Error: Audit cache not found at {AUDIT_CACHE_PATH}")
        print("   Please run evaluate_cross_series.py first!")
        return

    report = load_json(REPORT_PATH)
    audit_cache = load_json(AUDIT_CACHE_PATH)
    print(f"   ✓ Report loaded ({len(report)} domains)")
    print(f"   ✓ Audit cache loaded ({len(audit_cache)} entries)")

    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Process each domain
    all_growth_data = {}

    for domain_id in DOMAINS.keys():
        if domain_id not in report:
            print(f"\n⚠️  Skipping {domain_id}: not found in report")
            continue

        print(f"\n{'='*80}")
        print(f"📊 Processing: {DOMAINS[domain_id]}")
        print(f"{'='*80}")

        domain_report = report[domain_id]
        union_size = domain_report["union_size"]
        print(f"   Union size: {union_size}")

        domain_growth_data = {}

        for model in MODELS:
            model_name = model.split('/')[-1]
            print(f"\n   🤖 Processing {model_name}...")

            # Load trajectory
            traj_path = os.path.join(
                RESULTS_DIR,
                domain_id,
                model_name,
                "P4_Taxonomy_L5W5.trajectory.json"
            )

            if not os.path.exists(traj_path):
                print(f"      ⚠️  Trajectory not found: {traj_path}")
                continue

            trajectory = load_json(traj_path)
            print(f"      ✓ Loaded trajectory ({len(trajectory)} turns)")

            # Compute growth curve
            growth_curve = compute_growth_curve(trajectory, audit_cache, union_size)
            domain_growth_data[model] = growth_curve

            if growth_curve:
                final = growth_curve[-1]
                print(f"      ✓ Final metrics (Turn {final['turn']}): "
                      f"Valid={final['valid']}, "
                      f"Recall={final['recall']:.1%}, "
                      f"Accuracy={final['accuracy']:.1%}, "
                      f"Tokens={final['tokens']:,}")

        # Plot recall growth
        recall_filename = os.path.join(PLOTS_DIR, f"{domain_id}_recall_growth.png")
        plot_metric_growth(
            domain_id,
            domain_growth_data,
            "recall",
            "Recall (Coverage of Union)",
            recall_filename
        )

        # Plot accuracy growth
        accuracy_filename = os.path.join(PLOTS_DIR, f"{domain_id}_accuracy_growth.png")
        plot_metric_growth(
            domain_id,
            domain_growth_data,
            "accuracy",
            "Accuracy (Valid / Total)",
            accuracy_filename
        )

        all_growth_data[domain_id] = domain_growth_data

    # Save raw growth data
    print(f"\n{'='*80}")
    print("💾 Saving growth data...")
    with open(GROWTH_DATA_PATH, "w") as f:
        json.dump(all_growth_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved: {GROWTH_DATA_PATH}")

    # Generate combined 2x3 grid figure for paper
    print(f"\n{'='*80}")
    print("📊 Generating combined 2x3 grid figure...")
    plot_combined_grid(all_growth_data, COMBINED_OUTPUT)

    print("\n" + "=" * 80)
    print("✅ All plots generated successfully!")
    print("=" * 80)
    print(f"\n📂 Output directory: {PLOTS_DIR}")
    print(f"📄 Growth data: {GROWTH_DATA_PATH}")
    print(f"📄 Combined figure: {COMBINED_OUTPUT}")
    print()


if __name__ == "__main__":
    main()
