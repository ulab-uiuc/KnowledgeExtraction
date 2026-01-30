import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- ICML-compatible font settings (接近打印尺寸，字体9-10pt) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'mathtext.fontset': 'stix',
})

# --- Configuration ---
JSON_PATH = "results/all_domains_pareto_data_old.json"
OUTPUT_PATH = "paper/img/aggregated_pareto_curve.pdf"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Common X-axis for interpolation (Log space)
X_GRID = np.logspace(3, 6.5, 100)

PIPELINE_STYLING = {
    "P2_Sequential": {"color": "#1f77b4", "marker": "o", "label": "Sequential"},
    "P3_Reflection": {"color": "#ff7f0e", "marker": "s", "label": "Reflection"},
    "P4_Taxonomy_L2W2": {"color": "#2ca02c", "marker": "^", "label": "Taxonomy (L2W2)"},
    "P4_Taxonomy_L3W3": {"color": "#d62728", "marker": "^", "label": "Taxonomy (L3W3)"},
    "P4_Taxonomy_L5W5": {"color": "#9467bd", "marker": "^", "label": "Taxonomy (L5W5)"},
    "P5_MultiProfile_N3": {"color": "#8c564b", "marker": "D", "label": "Multi-Profile (N3)"},
    "P5_MultiProfile_N10": {"color": "#e377c2", "marker": "D", "label": "Multi-Profile (N10)"},
    "P5_MultiProfile_N20": {"color": "#7f7f7f", "marker": "D", "label": "Multi-Profile (N20)"},
}

# Legend order: Row1: Sequential, Reflection; Row2: Taxonomy variants; Row3: Multi-Profile variants
LEGEND_ORDER = [
    "P2_Sequential", "P3_Reflection", None,  # Row 1 (with placeholder for alignment)
    "P4_Taxonomy_L2W2", "P4_Taxonomy_L3W3", "P4_Taxonomy_L5W5",  # Row 2
    "P5_MultiProfile_N3", "P5_MultiProfile_N10", "P5_MultiProfile_N20",  # Row 3
]

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    all_pipelines = set()
    for domain in data:
        all_pipelines.update(data[domain].keys())

    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    sorted_pipelines = sorted(list(all_pipelines))

    for p_name in sorted_pipelines:
        domain_interp_results = []
        max_x_per_domain = []

        for domain in data:
            if p_name not in data[domain]:
                continue

            steps = data[domain][p_name]
            # Filter steps with 0 tokens and ensure cumulative increase for interpolation
            x_raw = [s["cumulative_tokens"] for s in steps if s["cumulative_tokens"] > 0]
            y_raw = [s["yield_ratio"] for s in steps if s["cumulative_tokens"] > 0]

            if not x_raw:
                continue

            # Ensure raw Y is monotonic before interpolation
            y_raw = np.maximum.accumulate(y_raw)

            # Record the max x for this domain
            max_x_per_domain.append(max(x_raw))

            # Linear interpolation in log-x space using common X_GRID
            y_interp = np.interp(X_GRID, x_raw, y_raw)
            domain_interp_results.append(y_interp)

        if not domain_interp_results:
            continue

        # Find the cutoff point: minimum of all domains' max x
        x_cutoff = min(max_x_per_domain)

        # Create mask for valid x range (only plot where we have actual data)
        valid_mask = X_GRID <= x_cutoff

        # Calculate statistics
        y_mean = np.mean(domain_interp_results, axis=0)
        y_std = np.std(domain_interp_results, axis=0)

        # Determine styling
        style = PIPELINE_STYLING.get(p_name, {"color": "gray", "marker": "o", "label": p_name})

        # Plot Mean Line (only valid range)
        plt.plot(
            X_GRID[valid_mask], y_mean[valid_mask],
            label=style["label"],
            color=style["color"],
            linewidth=1.8,
            alpha=0.9,
            zorder=10
        )

        # Plot Standard Deviation Shade (very narrow, only 0.25*std)
        plt.fill_between(
            X_GRID[valid_mask],
            np.maximum(0, y_mean[valid_mask] - 0.25 * y_std[valid_mask]),
            y_mean[valid_mask] + 0.25 * y_std[valid_mask],
            color=style["color"],
            alpha=0.12,
            zorder=1
        )

        # Add occasional markers (4-5 per curve)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            n_markers = min(5, max(4, len(valid_indices) // 20))
            marker_positions = np.linspace(0, len(valid_indices)-1, n_markers, dtype=int)
            marker_indices = valid_indices[marker_positions]
            plt.scatter(
                X_GRID[marker_indices],
                y_mean[marker_indices],
                color=style["color"],
                marker=style["marker"],
                s=16,  # markersize ~4
                zorder=11
            )

    plt.xscale('log')
    plt.xlabel('Cumulative Tokens (Log Scale)')
    plt.ylabel('Yield Ratio')
    plt.grid(True, which="both", ls="-", alpha=0.15)

    # Set y-axis limit
    plt.ylim(0, 6.5)

    plt.text(0.02, 0.97, "Baseline: Taxonomy (L2W2)",
             transform=plt.gca().transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))

    # Custom legend ordering: 3 rows x 3 columns
    # Row 1: Sequential, Reflection, (empty)
    # Row 2: Taxonomy variants
    # Row 3: Multi-Profile variants
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    ordered_handles = []
    ordered_labels = []
    for key in LEGEND_ORDER:
        if key is None:
            # Create invisible placeholder for alignment
            ordered_handles.append(plt.Line2D([], [], linestyle='none'))
            ordered_labels.append('')
        elif key in PIPELINE_STYLING:
            style = PIPELINE_STYLING[key]
            if style["label"] in label_to_handle:
                ordered_handles.append(label_to_handle[style["label"]])
                ordered_labels.append(style["label"])

    leg = plt.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     frameon=True, ncol=3, columnspacing=0.8, handlelength=1.2, fontsize=7)
    # 加粗图例中的线条
    for line in leg.get_lines():
        line.set_linewidth(2.0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)  # Room for x-label + legend
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Aggregated plot saved to {OUTPUT_PATH}")
    plt.close()

if __name__ == "__main__":
    main()
