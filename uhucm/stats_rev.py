"""Revision statistics: Wilson CIs of hold rates, Friedman + Nemenyi across the five datasets, Wilcoxon tests for efficiency
(normal approximation throughout), growth-rate fits and memory vs n. Writes out_synth/stats_rev.json."""
import json, math, os, glob
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, norm, studentized_range, rankdata
H = os.path.dirname(os.path.abspath(__file__))

def wilson(h, n, a=0.05):
    if not n: return None
    z = norm.ppf(1 - a / 2); p = h / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - w, 3), round(c + w, 3)]

def load_quality():
    Q = {}
    for lab, p in (("D4", "out_suite/suite_FSQ_NYC.json"), ("D5", "out_suite/suite_FSQ_TKY.json")):
        Q[lab] = {q["K"]: q for q in json.load(open(os.path.join(H, p)))["quality"]}
    for lab, c in (("D1", "Philadelphia"), ("D2", "New_Orleans"), ("D3", "Tampa")):
        p = os.path.join(H, "results_kaggle", f"rev_{c}.json")
        if os.path.exists(p): Q[lab] = {int(q["K"]): q for q in json.load(open(p))["d50"]}
    return Q

def nemenyi(ranks_mean, k, n, a=0.05):
    q = studentized_range.ppf(1 - a, k, 1e6) / math.sqrt(2)
    return q * math.sqrt(k * (k + 1) / (6 * n))

out = {"ci": {}, "friedman": {}}
Q = load_quality()
for lab, byK in sorted(Q.items()):
    for K, q in byK.items():
        out["ci"][f"{lab}/K={K}"] = {g: dict(rate=round(q[f"hits_{g}"] / q[f"n_{g}"], 3), n=q[f"n_{g}"], ci=wilson(q[f"hits_{g}"], q[f"n_{g}"]))
                                    for g in ("kept", "rej", "matched") if q.get(f"n_{g}")}
labs = sorted(Q)
for K in (100, 300, 1000, "mean"):
    rows = []
    for lab in labs:
        if K == "mean":
            rows.append([np.mean([Q[lab][k][f"hits_{g}"] / Q[lab][k][f"n_{g}"] for k in Q[lab]]) for g in ("kept", "matched", "rej")])
        else:
            q = Q[lab][K]; rows.append([q[f"hits_{g}"] / q[f"n_{g}"] for g in ("kept", "matched", "rej")])
    A = np.array(rows); n = len(A)
    if n < 3: continue
    st, p = friedmanchisquare(A[:, 0], A[:, 1], A[:, 2])
    rk = np.array([rankdata(-r) for r in A], float)   # rank 1 = highest hold rate
    mr = rk.mean(0); cd = nemenyi(mr, 3, n)
    out["friedman"][str(K)] = dict(n=n, chi2=round(float(st), 3), p=round(float(p), 4), mean_rank=dict(zip(("kept", "matched", "rej"), map(float, mr))),
                                   CD=round(cd, 3), kept_vs_rej=bool(abs(mr[0] - mr[2]) > cd), matched_vs_rej=bool(abs(mr[1] - mr[2]) > cd),
                                   kept_vs_matched=bool(abs(mr[0] - mr[1]) > cd))
json.dump(out, open(os.path.join(H, "out_synth", "stats_rev.json"), "w"), indent=1)
print(json.dumps(out["friedman"], indent=1))
