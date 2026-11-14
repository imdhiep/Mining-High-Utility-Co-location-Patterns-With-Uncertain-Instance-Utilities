"""Wilcoxon signed-rank test of the per-dataset paired differences in hold rate (score-matched kept minus rejected) at
each K, Holm-corrected over the three thresholds, from out_synth/stats_rev3.json (FAIR=1 stats_rev3.py).
-> out_synth/stats_wilcoxon_fair.json"""
import json, os
import numpy as np
from scipy.stats import wilcoxon
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "out_synth", "stats_rev3.json")))
res, ps = {}, []
for K in (100, 300, 1000):
    x = [(ds, d["per"][f"{ds}/K={K}"]["paired"]["diff"]) for ds in d["datasets"]
         if f"{ds}/K={K}" in d["per"] and d["per"][f"{ds}/K={K}"].get("paired") and d["per"][f"{ds}/K={K}"]["paired"].get("n_pairs")]
    v = np.array([b for _, b in x])
    p = float(wilcoxon(v, zero_method="zsplit").pvalue)
    res[K] = dict(n=len(v), datasets=[a for a, _ in x], pos=int((v > 0).sum()), zero=int((v == 0).sum()), neg=int((v < 0).sum()),
                  median=round(float(np.median(v)), 3), p=p); ps.append(p)
order = np.argsort(ps); run = 0.0
for r, i in enumerate(order):
    run = max(run, min(1.0, (3 - r) * ps[i])); res[(100, 300, 1000)[i]]["p_holm"] = round(run, 4)
json.dump(res, open(os.path.join(H, "out_synth", "stats_wilcoxon_fair.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
