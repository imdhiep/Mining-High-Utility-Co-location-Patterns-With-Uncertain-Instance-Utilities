"""Significance tests on existing results: Wilcoxon signed-rank tests for efficiency comparisons across
(dataset, threshold) pairs, Holm correction of the E7 permutation tests, and s0 overlap fractions."""
import json, numpy as np
from scipy.stats import wilcoxon
import results_db
db, syn, d8 = results_db.load()
EFF = ["Phi", "New", "Tam", "FSQ_NYC", "FSQ_TKY"]; LAB = {"Phi": "D1", "New": "D2", "Tam": "D3", "FSQ_NYC": "D4", "FSQ_TKY": "D5", "Las": "D6a", "Pho": "D6b", "Tor": "D6c"}
out = {}
def pairs(a, b, key):
    x, y = [], []
    for k in EFF:
        for K, v in sorted(db[k]["eff_mu"].items()):
            x.append(float(v[a][key])); y.append(float(v[b][key]))
    x, y = np.array(x), np.array(y)
    st = wilcoxon(x, y, zero_method="zsplit", method="approx")
    return dict(n=len(x), median_ratio=round(float(np.median(y / x)), 3), wins=int((x < y).sum()), ties=int((x == y).sum()), p=float(st.pvalue))
for a, b in (("exact", "noprune"), ("exact", "mean"), ("exact", "chord")):
    for key in ("evaluated", "time"):
        out[f"{a}_vs_{b}_{key}"] = pairs(a, b, key)
abl = {}
for v in ("mean", "chord", "noFUB", "noPCB", "noSB", "noNP", "noprune"):
    for key in ("evaluated", "time"):
        x = np.array([float(db[k]["ablation"]["full"][key]) for k in EFF]); y = np.array([float(db[k]["ablation"][v][key]) for k in EFF])
        p = float(wilcoxon(x, y, zero_method="zsplit", method="approx").pvalue) if np.any(x != y) else 1.0
        abl[f"{v}_{key}"] = dict(ratio_min=round(float((y / x).min()), 2), ratio_max=round(float((y / x).max()), 2), p=p)
out["ablation"] = abl
# Holm on E7
def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
for name, keys in (("all", list(LAB)), ("noD6", ["Phi", "New", "Tam", "FSQ_NYC", "FSQ_TKY"])):
    lab, ps = [], []
    for k in keys:
        for K in (100, 300, 1000):
            q = db[k]["quality"][K]
            if q.get("perm_p") is not None: lab.append(f"{LAB[k]}/K={K}"); ps.append(q["perm_p"])
    adj = holm(np.array(ps))
    out[f"E7_holm_{name}"] = dict(m=len(ps), raw_sig=[l for l, p in zip(lab, ps) if p < 0.05], holm_sig=[l for l, p in zip(lab, adj) if p < 0.05],
                                 min_adj=round(float(adj.min()), 4))
# kept vs matched hold rate (paired over dataset-threshold), Wilcoxon
for name, keys in (("all", list(LAB)), ("noD6", ["Phi", "New", "Tam", "FSQ_NYC", "FSQ_TKY"])):
    a = np.array([db[k]["quality"][K]["hold_matched"] for k in keys for K in (100, 300, 1000)], float)
    b = np.array([db[k]["quality"][K]["hold_rej"] for k in keys for K in (100, 300, 1000)], float)
    c = np.array([db[k]["quality"][K]["hold_kept"] for k in keys for K in (100, 300, 1000)], float)
    out[f"E7_hold_{name}"] = dict(n=len(a), matched_minus_rej_median=round(float(np.median(a - b)), 3),
                                  p_matched_vs_rej=float(wilcoxon(a, b, zero_method="zsplit").pvalue),
                                  kept_minus_rej_median=round(float(np.median(c - b)), 3), p_kept_vs_rej=float(wilcoxon(c, b, zero_method="zsplit").pvalue))
# s0 overlap: fraction of default R_1.645 (K=300) also reported with s0 = city median
s0 = {}
for k in ("Phi", "New", "Tam"):
    A = db[k]["quality"][300]["R1.645"]; v = db[k]["sensitivity"]["s0_median"]; B, J = v["R"], v["J"]
    inter = J * (A + B) / (1 + J); s0[LAB[k]] = dict(default=A, variant=B, J=J, overlap_of_default=round(inter / A, 3))
out["s0"] = s0
json.dump(out, open("out_synth/stats_tests.json", "w"), indent=1)
print(json.dumps(out, indent=1))
