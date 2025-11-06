
import numpy as np, re, math
from collections import Counter

def _normalize_item(s: str) -> str:
    t = s.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[;,\.\s]+$", "", t)
    return t

def build_occurrence(rounds):
    rounds_norm = [[_normalize_item(x) for x in r if _normalize_item(x)] for r in rounds]
    rounds_set = [sorted(set(r)) for r in rounds_norm]
    all_items = sorted(set().union(*map(set, rounds_set)))
    idx = {it:i for i,it in enumerate(all_items)}
    R, S = len(rounds_set), len(all_items)
    import numpy as np
    occ = np.zeros((R, S), dtype=int)
    for r_id, rset in enumerate(rounds_set):
        for it in rset:
            occ[r_id, idx[it]] = 1
    freq_counter = Counter()
    for r in rounds_norm:
        freq_counter.update(r)
    freq = np.array([freq_counter[it] for it in all_items], dtype=int)
    return all_items, occ, freq

def chao1_lower_bound(S_obs, f1, f2):
    eps = 1e-9
    if f2 > 0:
        return max(S_obs, S_obs + (f1*(f1-1))/(2.0*(f2+eps)))
    else:
        return max(S_obs, S_obs + (f1*(f1-1))/2.0)

def good_turing_unseen_mass(f1, n):
    return f1 / max(n, 1)

def accumulation_curve(occ):
    R, S = occ.shape
    seen=set(); y=[]
    import numpy as np
    for t in range(R):
        seen |= set(np.where(occ[t]==1)[0])
        y.append(len(seen))
    x = np.arange(1, R+1, dtype=float)
    return x, np.array(y, dtype=float)

def mm_asymptote(x, y):
    import numpy as np
    mask = (y > 0) & (x > 0)
    if mask.sum() < 2:
        return float('nan'), float('nan')
    X = 1.0/x[mask]; Y = 1.0/y[mask]
    A = np.vstack([X, np.ones_like(X)]).T
    alpha, beta = np.linalg.lstsq(A, Y, rcond=None)[0]
    if beta <= 0:
        return float('nan'), float('nan')
    A_hat = 1.0/beta
    B_hat = alpha/beta
    if A_hat < max(y):
        A_hat = float(max(y))
    return float(A_hat), float(B_hat)

def fit_pade_rational(x, y, degree=2):
    import numpy as np
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); d = degree
    Phi_left = np.vstack([x**p for p in range(d+1)]).T
    Phi_right = np.vstack([-(y*(x**p)) for p in range(1, d+1)]).T
    Phi = np.concatenate([Phi_left, Phi_right], axis=1)
    coeffs, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    a = coeffs[:d+1]; b = np.concatenate([[1.0], coeffs[d+1:]])
    def predict(k):
        num = sum(a[p]*(k**p) for p in range(d+1))
        den = sum(b[p]*(k**p) for p in range(d+1))
        return num/den if den!=0 else np.nan
    S_inf = a[-1]/b[-1] if abs(b[-1])>1e-12 else predict(max(x)*1e6)
    return {"a": a, "b": b, "S_inf": float(S_inf), "predict": predict}

def fit_powerlaw_residual(x, y, A_grid=500, scale=12.0):
    import numpy as np
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    y_max = float(np.max(y)); eps = 1e-6
    candidates = np.linspace(y_max+1e-3, y_max*scale, A_grid)
    best = {"A": np.nan, "c": np.nan, "beta": np.nan, "sse": np.inf}
    Xlog = np.log(x + eps)
    for A in candidates:
        R = A - y
        if np.any(R <= 0): 
            continue
        Ylog = np.log(R)
        M = np.vstack([np.ones_like(Xlog), -Xlog]).T
        params, *_ = np.linalg.lstsq(M, Ylog, rcond=None)
        logc, beta = params[0], params[1]
        sse = float(np.sum((Ylog - (M@params))**2))
        if sse < best["sse"]:
            best = {"A": float(A), "c": float(np.exp(logc)), "beta": float(beta), "sse": sse}
    def predict(k):
        return best["A"] - best["c"] * (k**(-best["beta"]))
    return {"S_inf": best["A"], "c": best["c"], "beta": best["beta"], "predict": predict}
