# Simulate multi-round LLM knowledge outputs and run the unified coverage + infinite-sample extrapolation toolkit
#
# Simulation design:
# - Ground-truth knowledge size S_true with Zipf-long-tail popularity (alpha>1)
# - T rounds; each round samples K_r items w/o replacement ~ Poisson(mean_items) bounded
# - Paraphrase/noise injected to test normalization & dedup
# - Outputs: rounds (list[list[str]]), tables, and two plots
#

# https://chatgpt.com/share/69042014-e22c-8009-a468-2837140581be


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import json

# -------------------------------
# 1) Generate synthetic "true" knowledge universe
# -------------------------------
rng = np.random.default_rng(42)
random.seed(42)

S_true = 300              # ground-truth number of atomic facts
alpha = 1.15              # Zipf exponent (heavier tail -> more singletons)
T = 12                    # number of rounds
mean_items = 38           # expected items per round

# Create base facts with mild subtopic tags
subtopics = ["definition", "theory", "training", "evaluation", "applications", "failure"]
subtag = rng.choice(subtopics, size=S_true, replace=True)
base_facts = [f"fact {i:03d} — {subtag[i]}" for i in range(S_true)]

# Popularity ~ Zipf; normalize
ranks = np.arange(1, S_true+1)
zipf_weights = 1.0 / (ranks ** alpha)
zipf_weights /= zipf_weights.sum()

# -------------------------------
# 2) Build rounds by sampling without replacement using popularity weights
# -------------------------------
def paraphrase(s: str) -> str:
    # simple paraphrase/noise to test normalization (case, punctuation, stopword)
    variants = [
        s, s.upper(), s.capitalize(),
        s + ".", s + "!", " " + s + " ",
        s.replace(" — ", ": "),
        s.replace("fact", "knowledge"),
        s.replace(" — ", " - "),
    ]
    return random.choice(variants)

rounds = []
round_sizes = rng.poisson(mean_items, size=T).clip(8, 70)  # keep reasonable bounds
for t in range(T):
    k = int(round_sizes[t])
    k = min(k, S_true)
    # sample unique indices by weighted choice
    idx = rng.choice(np.arange(S_true), size=k, replace=False, p=zipf_weights)
    items = [paraphrase(base_facts[i]) for i in idx]
    # add a few duplicates within-round to test frequency stats
    if k > 10:
        dups = rng.choice(items, size=int(0.05*k), replace=True)
        items.extend(list(dups))
    rounds.append(items)

# Save the simulated rounds as JSON for reuse
with open("/home/siqizhu4/llmknowledge/simulated_rounds.json", "w") as f:
    json.dump(rounds, f, ensure_ascii=False, indent=2)

# -------------------------------
# 3) Run the unified toolkit
# -------------------------------
# Import helper functions from the saved module
import sys
sys.path.append("/home/siqizhu4/llmknowledge")
from llm_coverage_unified import build_occurrence, accumulation_curve, mm_asymptote
from llm_coverage_unified import chao1_lower_bound, good_turing_unseen_mass
from llm_coverage_unified import fit_pade_rational, fit_powerlaw_residual

# Build occurrence and frequencies
items, occ, freq = build_occurrence(rounds)
S_obs = len(items)
f1 = int((freq == 1).sum())
f2 = int((freq == 2).sum())
n = int(freq.sum())
R = occ.shape[0]

# Classic estimators
S_chao1 = chao1_lower_bound(S_obs, f1, f2)
cov_chao1 = S_obs / S_chao1 if S_chao1 > 0 else float("nan")
gt_unseen = good_turing_unseen_mass(f1, n)
cov_gt = max(0.0, min(1.0, 1.0 - gt_unseen))

# Accumulation & M–M
x, y = accumulation_curve(occ)
A_hat, B_hat = mm_asymptote(x, y)
cov_mm = S_obs / A_hat if A_hat and not np.isnan(A_hat) else float("nan")

# Infinite-sample extrapolation
pade2 = fit_pade_rational(x, y, degree=2)
pade3 = fit_pade_rational(x, y, degree=3)
plaw  = fit_powerlaw_residual(x, y, A_grid=600, scale=15.0)

# Coverage from the three infinite estimates
cov_pade2 = S_obs / pade2["S_inf"] if np.isfinite(pade2["S_inf"]) and pade2["S_inf"]>0 else np.nan
cov_pade3 = S_obs / pade3["S_inf"] if np.isfinite(pade3["S_inf"]) and pade3["S_inf"]>0 else np.nan
cov_plaw  = S_obs / plaw ["S_inf"]  if np.isfinite(plaw ["S_inf"])  and plaw ["S_inf"] >0 else np.nan

# -------------------------------
# 4) Present results
# -------------------------------
overview = pd.DataFrame([{
    "TRUE S (hidden)": S_true,
    "Rounds (R)": R,
    "Observed uniques (S_obs)": S_obs,
    "Singletons (f1)": f1,
    "Doubletons (f2)": f2,
    "Total tokens (n)": n
}])

estimators = pd.DataFrame([{
    "Chao1 S_hat": S_chao1, "Coverage (Chao1)": cov_chao1,
    "Good–Turing unseen mass": gt_unseen, "Coverage (Good–Turing)": cov_gt,
    "M–M Asymptote A_hat": A_hat, "Coverage (M–M)": cov_mm,
    "Padé(d=2) S_inf": pade2["S_inf"], "Coverage (Padé d=2)": cov_pade2,
    "Padé(d=3) S_inf": pade3["S_inf"], "Coverage (Padé d=3)": cov_pade3,
    "Power-law S_inf": plaw["S_inf"], "Coverage (Power-law)": cov_plaw
}])

# Plot 1: Accumulation with M–M
plt.figure()
plt.plot(x, y, marker='o')
if np.isfinite(A_hat):
    plt.axhline(A_hat)
plt.title("Simulated Knowledge Accumulation (with M–M Asymptote)")
plt.xlabel("Rounds")
plt.ylabel("Cumulative unique items")
plt.show()

# Plot 2: Extrapolation
k_plot = np.arange(1, max(80, int(2.0 * len(x))) + 1, dtype=float)
plt.figure()
plt.plot(x, y, marker='o', linestyle='none')
for name, model in [
    (f"Padé d=2 (S∞≈{pade2['S_inf']:.1f})", pade2),
    (f"Padé d=3 (S∞≈{pade3['S_inf']:.1f})", pade3),
    (f"Power-law (S∞≈{plaw['S_inf']:.1f})", plaw),
]:
    yhat = np.array([model["predict"](k) for k in k_plot])
    plt.plot(k_plot, yhat, label=name)
for model in [pade2, pade3, plaw]:
    if np.isfinite(model["S_inf"]):
        plt.axhline(model["S_inf"])
plt.title("Extrapolation to Infinite Sampling — Simulated")
plt.xlabel("Rounds (k)")
plt.ylabel("Cumulative unique items S(k)")
plt.legend()
plt.show()

# Show tables to the user
from IPython.display import display

print("=== Simulated — Overview (includes hidden TRUE S) ===")
display(overview)

print("=== Simulated — Estimators & Infinite-sample Coverage ===")
display(estimators)

# Save the simulated rounds for download
print("Saved simulated rounds to /home/siqizhu4/llmknowledge/simulated_rounds.json")
