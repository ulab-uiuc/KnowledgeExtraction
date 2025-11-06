# LLM Knowledge Coverage Estimation — minimal, runnable toolkit
#
# This notebook-style script provides:
# - Parsing & normalization of multi-round LLM outputs into "knowledge units"
# - Frequency statistics (S_obs, f1, f2, n)
# - Estimators: Chao1 (abundance), Good–Turing unseen mass, pairwise Chapman,
#               simple Schnabel (multi-round capture-recapture)
# - Accumulation curve + asymptote fit via Michaelis–Menten linearization
# - Slice-by-subtopic (optional)
#
# Input format:
#   rounds = [
#     ["fact 1", "fact 2", ...],   # round 1
#     ["fact 2", "fact 3", ...],   # round 2
#     ...
#   ]
#
# You can replace the demo 'rounds' with your own lists.
#
# Notes:
# - Charts use matplotlib (no seaborn), one chart per figure, no custom colors.
# - Internet is disabled in this environment.
# - For a larger pipeline, plug in your own normalization/deduplication rules.
#
from collections import Counter, defaultdict
import re
import numpy as np
import math
import itertools
import pandas as pd
import matplotlib.pyplot as plt

try:
    from caas_jupyter_tools import display_dataframe_to_user
    HAVE_CAS_DISPLAY = True
except Exception:
    HAVE_CAS_DISPLAY = False

# -------------------------------
# Normalization & canonicalization
# -------------------------------
def normalize_item(s: str) -> str:
    """
    Simple normalization: lowercase, strip, collapse whitespace, strip trailing punctuation.
    You may plug in lemmatization, entity linking, or entailment-based dedup here.
    """
    if s is None:
        return ""
    t = s.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[;,\.\s]+$", "", t)
    return t

def build_occurrence(rounds):
    """
    Build:
      - canon_items: ordered list of unique canonical items
      - occ: (R x S) boolean matrix of presence per round (incidence)
      - freq: per-item total frequency across all rounds (counting duplicates within rounds)
    """
    R = len(rounds)
    # canonicalized per round (preserving duplicates for frequency counting within a round)
    rounds_norm = [[normalize_item(x) for x in r if normalize_item(x)] for r in rounds]
    # set-based per round for incidence matrix
    rounds_set = [sorted(set(r)) for r in rounds_norm]
    all_items = sorted(set(itertools.chain.from_iterable(rounds_set)))
    idx = {it:i for i,it in enumerate(all_items)}
    S = len(all_items)

    occ = np.zeros((R, S), dtype=int)
    for r_id, rset in enumerate(rounds_set):
        for it in rset:
            occ[r_id, idx[it]] = 1

    # frequency across all rounds counting duplicates within each round
    freq_counter = Counter()
    for r in rounds_norm:
        freq_counter.update(r)
    freq = np.array([freq_counter[it] for it in all_items], dtype=int)

    return all_items, occ, freq

# -------------------------------
# Core statistics
# -------------------------------
def core_stats(occ, freq):
    """
    Returns dictionary:
      S_obs: observed unique items
      f1, f2: singletons and doubletons (by total frequency)
      n: total token count (sum of frequencies)
      R: rounds
    """
    S_obs = freq.size
    f1 = int((freq == 1).sum())
    f2 = int((freq == 2).sum())
    n = int(freq.sum())
    R = int(occ.shape[0]) if occ is not None else None
    return {"S_obs": S_obs, "f1": f1, "f2": f2, "n": n, "R": R}

# -------------------------------
# Estimators
# -------------------------------
def chao1_lower_bound(S_obs, f1, f2):
    """
    Abundance-based Chao1 lower-bound estimator of total richness.
    If f2==0, use bias-corrected variant with denominator 2 (add small epsilon).
    """
    eps = 1e-9
    denom = 2 * f2 if f2 > 0 else 2.0
    S_hat = S_obs + (f1 * (f1 - 1)) / (2.0 * (f2 + eps)) if f2 > 0 else S_obs + (f1 * (f1 - 1)) / (2.0)
    return max(S_hat, S_obs)

def good_turing_unseen_mass(f1, n):
    """Good–Turing estimate of unseen probability mass."""
    if n <= 0:
        return 1.0
    return f1 / max(n, 1)

def coverage_from_mass(unseen_mass):
    """Coverage is 1 - unseen_mass."""
    return max(0.0, min(1.0, 1.0 - unseen_mass))

def pairwise_chapman(occ):
    """
    For each pair of rounds (i, j), compute Chapman estimator and return their mean.
    Chapman (two-sample capture-recapture, closed population):
      S_hat = ((n1+1)*(n2+1)/(m+1)) - 1
      where n1, n2 are counts in each sample, m is overlap.
    Returns dict with per-pair estimates and their mean.
    """
    R, S = occ.shape
    estimates = []
    pairs = []
    for i in range(R):
        for j in range(i+1, R):
            n1 = int(occ[i].sum())
            n2 = int(occ[j].sum())
            m = int((occ[i] & occ[j]).sum())
            if m == 0:
                # skip degenerate pairs (no overlap) to avoid infinite estimate
                continue
            S_hat = ((n1 + 1.0) * (n2 + 1.0) / (m + 1.0)) - 1.0
            estimates.append(S_hat)
            pairs.append((i, j, n1, n2, m, S_hat))
    out = {
        "pair_estimates": pairs,
        "mean_pairwise_estimate": float(np.mean(estimates)) if estimates else float('nan')
    }
    return out

def schnabel_estimator(occ):
    """
    Simple Schnabel estimator for multiple-sample capture-recapture.
    Formula (one common form):
      N_hat = (sum_t C_t * M_t) / (sum_t R_t)
      where:
        C_t = catch in sample t (unique count in round t)
        M_t = number already marked before sample t (cumulative unique before t)
        R_t = recaptures in sample t (items in sample t that were already marked)
    """
    R, S = occ.shape
    marked_set = set()
    num = 0.0
    den = 0.0
    for t in range(R):
        sample_items = set(np.where(occ[t] == 1)[0])
        C_t = len(sample_items)
        M_t = len(marked_set)
        R_t = len(sample_items & marked_set)
        if R_t > 0:
            num += C_t * M_t
            den += R_t
        # update marked
        marked_set |= sample_items
    if den == 0:
        return float('nan')
    return num / den

# -------------------------------
# Accumulation curve & asymptote
# -------------------------------
def accumulation_curve(occ):
    """
    Observed unique vs. number of rounds (1..R). Assumes rounds are in chronological order.
    Returns arrays: x (1..R), y (cumulative observed uniques).
    """
    R, S = occ.shape
    cum = []
    seen = set()
    y = []
    for t in range(R):
        seen |= set(np.where(occ[t] == 1)[0])
        y.append(len(seen))
    x = np.arange(1, R+1)
    return x, np.array(y, dtype=float)

def michaelis_menten_asymptote(x, y):
    """
    Fit a Michaelis–Menten curve: y = (A*x)/(B + x), with A>0, B>0
    Use linearization: 1/y = (B/A)*(1/x) + 1/A
    Exclude any x where y==0.
    Returns (A_hat, B_hat); asymptote is A_hat.
    """
    # Filter zeros
    mask = (y > 0) & (x > 0)
    if mask.sum() < 2:
        return float('nan'), float('nan')
    X = 1.0 / x[mask]
    Y = 1.0 / y[mask]
    # Linear fit Y = alpha * X + beta
    # alpha = B/A, beta = 1/A
    A = np.vstack([X, np.ones_like(X)]).T
    alpha, beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    if beta <= 0:
        return float('nan'), float('nan')
    A_hat = 1.0 / beta
    B_hat = alpha / beta
    if A_hat < max(y):  # Asymptote should be >= current observed
        A_hat = max(y)
    return float(A_hat), float(B_hat)

# -------------------------------
# Driver that produces a compact report
# -------------------------------
def estimate_coverage(rounds, subtopics=None):
    """
    rounds: list[list[str]], each inner list is one sampling round's knowledge items.
    subtopics: optional parallel structure len(rounds), each a list of subtopic labels aligned to items.
               If provided, we derive subtopic slices.
    Returns a dictionary of results and produces plots/tables.
    """
    items, occ, freq = build_occurrence(rounds)
    stats = core_stats(occ, freq)
    S_obs, f1, f2, n, R = stats["S_obs"], stats["f1"], stats["f2"], stats["n"], stats["R"]

    # Estimators
    S_chao1 = chao1_lower_bound(S_obs, f1, f2)
    gt_unseen = good_turing_unseen_mass(f1, n)
    gt_cov = coverage_from_mass(gt_unseen)
    chap = pairwise_chapman(occ)
    schnabel = schnabel_estimator(occ)

    # Coverage from specific S_hat
    cov_chao1 = S_obs / S_chao1 if S_chao1 and not math.isnan(S_chao1) else float('nan')
    cov_chap_mean = S_obs / chap["mean_pairwise_estimate"] if chap["mean_pairwise_estimate"] and not math.isnan(chap["mean_pairwise_estimate"]) else float('nan')
    cov_schnabel = S_obs / schnabel if schnabel and not math.isnan(schnabel) else float('nan')

    # Accumulation & asymptote
    x, y = accumulation_curve(occ)
    A_hat, B_hat = michaelis_menten_asymptote(x, y)
    cov_asym = S_obs / A_hat if A_hat and not math.isnan(A_hat) else float('nan')

    # --- Tables ---
    overview = pd.DataFrame([{
        "Rounds (R)": R,
        "Observed uniques (S_obs)": S_obs,
        "Singletons (f1)": f1,
        "Doubletons (f2)": f2,
        "Total tokens (n)": n
    }])

    estimators = pd.DataFrame([{
        "Chao1 S_hat": S_chao1,
        "Coverage (Chao1)": cov_chao1,
        "Good–Turing unseen mass": gt_unseen,
        "Coverage (Good–Turing)": gt_cov,
        "Chapman mean S_hat": chap["mean_pairwise_estimate"],
        "Coverage (Chapman mean)": cov_chap_mean,
        "Schnabel S_hat": schnabel,
        "Coverage (Schnabel)": cov_schnabel,
        "Asymptote A_hat (M–M)": A_hat,
        "Coverage (Asymptote)": cov_asym
    }])

    pair_rows = []
    for (i, j, n1, n2, m, S_hat) in chap["pair_estimates"]:
        pair_rows.append({
            "Pair (i,j)": f"({i+1},{j+1})",
            "n1": n1, "n2": n2, "overlap m": m, "Chapman S_hat": S_hat
        })
    pairs_df = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(columns=["Pair (i,j)","n1","n2","overlap m","Chapman S_hat"])

    # Display tables nicely
    if HAVE_CAS_DISPLAY:
        display_dataframe_to_user("Overview (LLM Knowledge Coverage)", overview)
        display_dataframe_to_user("Estimators", estimators)
        if len(pairs_df) > 0:
            display_dataframe_to_user("Pairwise Chapman Estimates", pairs_df)

    # --- Plot accumulation curve ---
    plt.figure()
    plt.plot(x, y, marker='o')
    if not math.isnan(A_hat):
        # draw asymptote as horizontal line
        plt.axhline(A_hat)
    plt.title("Knowledge Accumulation Curve")
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative unique items")
    plt.show()

    results = {
        "items": items,
        "occurrence_matrix": occ,
        "frequencies": freq,
        "overview": overview,
        "estimators": estimators,
        "pairwise_chapman": pairs_df,
        "accumulation_x": x,
        "accumulation_y": y,
        "asymptote_A_hat": A_hat,
        "asymptote_B_hat": B_hat
    }
    return results

# -------------------------------
# Demo with toy data
# -------------------------------
demo_rounds = [
    [
        "Definition of RAG",
        "RAG = Retrieval-Augmented Generation",
        "Key components: retriever, generator",
        "Retriever uses dense indexes",
        "Applications: QA, code search",
        "Failure: retrieval drift",
    ],
    [
        "RAG = Retrieval-Augmented Generation",
        "Key components: retriever, generator",
        "Dense retriever vs sparse retriever",
        "Failure: retrieval drift",
        "Mitigation: query rewriting",
        "Evaluation: faithfulness",
    ],
    [
        "RAG pipelines: indexing, retrieval, generation",
        "Failure: hallucination",
        "Evaluation: faithfulness",
        "Mitigation: query rewriting",
        "Applications: QA, code search",
        "Cold-start indexing strategies",
    ],
    [
        "Negative mining improves retriever",
        "Failure: hallucination",
        "RAG = Retrieval-Augmented Generation",
        "Graph RAG is a variant",
        "Applications: QA, code search",
        "Evaluation: human preference",
    ],
    [
        "Retriever uses dense indexes",
        "Graph RAG is a variant",
        "Query expansion helps recall",
        "Cold-start indexing strategies",
        "Evaluation: human preference",
        "Failure: retrieval drift",
    ]
]

res = estimate_coverage(demo_rounds)

# Print a compact textual summary
print("==== Overview ====")
print(res["overview"].to_string(index=False))
print("\n==== Estimators ====")
print(res["estimators"].to_string(index=False))
print("\n==== Pairwise Chapman (first 5 rows) ====")
if len(res["pairwise_chapman"]) > 0:
    print(res["pairwise_chapman"].head().to_string(index=False))
else:
    print("(no overlapping pairs; estimator undefined)")


