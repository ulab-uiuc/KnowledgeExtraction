import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
JSON_PATH = "results/all_domains_pareto_data.json"
OUTPUT_DIR = "results/plots/all_domains"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PIPELINE_STYLING = {
    "P2_Sequential": {"color": "#1f77b4", "marker": "o", "label": "P2: Sequential"},
    "P3_Reflection": {"color": "#ff7f0e", "marker": "s", "label": "P3: Reflection"},
    "P4_Taxonomy_L2W2": {"color": "#2ca02c", "marker": "^", "label": "P4: Taxonomy (L2W2)"},
    "P4_Taxonomy_L3W3": {"color": "#d62728", "marker": "^", "label": "P4: Taxonomy (L3W3)"},
    "P4_Taxonomy_L5W5": {"color": "#9467bd", "marker": "^", "label": "P4: Taxonomy (L5W5)"},
    "P5_MultiProfile_N3": {"color": "#8c564b", "marker": "D", "label": "P5: Multi-Profile (N3)"},
    "P5_MultiProfile_N10": {"color": "#e377c2", "marker": "D", "label": "P5: Multi-Profile (N10)"},
    "P5_MultiProfile_N20": {"color": "#7f7f7f", "marker": "D", "label": "P5: Multi-Profile (N20)"},
}

DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
DEFAULT_MARKERS = ['h', 'H', '8', 'd', '|', '_']

def plot_domain(domain_name, pipelines_data):
    plt.figure(figsize=(12, 8))
    
    p_names = sorted(pipelines_data.keys())
    
    for i, p_name in enumerate(p_names):
        data = pipelines_data[p_name]
        
        # Extract X and Y, skipping points with 0 tokens (incompatible with log scale)
        tokens = []
        ratios = []
        for step in data:
            t = step["cumulative_tokens"]
            r = step["yield_ratio"]
            if t > 0:
                tokens.append(t)
                ratios.append(r)
        
        if not tokens:
            continue

        # Force Monotonic Increase for Y-axis (Cumulative Knowledge Discovery)
        # This handles data noise where some turns might report fewer unique points
        ratios = np.maximum.accumulate(ratios)
        
        # Determine styling
        style = PIPELINE_STYLING.get(p_name, {
            "color": DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
            "marker": "o",
            "label": p_name
        })
        
        # Plot
        plt.plot(
            tokens, 
            ratios, 
            label=style["label"], 
            color=style["color"], 
            marker=style["marker"], 
            markersize=6, 
            linewidth=1.5,
            alpha=0.9
        )

    plt.xscale('log')
    plt.xlabel('Cumulative Tokens (Log Scale)', fontsize=12)
    plt.ylabel('Yield Ratio (Relative to Baseline)', fontsize=12)
    plt.title(f'Knowledge Extraction Trajectories: {domain_name.replace("_", " ")}', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"{domain_name}_pareto.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    for domain_name, pipelines_data in data.items():
        plot_domain(domain_name, pipelines_data)

if __name__ == "__main__":
    main()
