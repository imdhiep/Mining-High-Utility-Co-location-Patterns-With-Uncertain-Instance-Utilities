"""Semi-synthetic calibration table, lower-bound rule only (z = 1.645): FDP of point, parametric (z = 2.45, own target),
UHUCPM lower bound and the oracle lower bound (true prior and center), and efficiency = true patterns found by the lower
bound / true patterns found by the oracle lower bound (summed over seeds and K, seeds 0-9). Writes out_ceb/semisyn_table2.tex/json."""
import json, os, numpy as np
from make_semisyn_table import read, LAY, TR, H
rows, J = [], {}
for lab, c in LAY:
    first = True
    for t, tl in TR:
        rd = lambda name: {**read(f"{H}/out_semisyn/{name}", c), **read(f"{H}/out_semisyn_s3_9/{name}", c)}   # seeds 0-2 (out_semisyn) and 3-9 (out_semisyn_s3_9)
        old = rd(f"own_{c}_{t}_gamma.txt"); new = rd(f"log_{c}_{t}_np_cboot.txt"); orc = rd(f"log_{c}_{t}_oracle.txt")
        if not new: continue
        m = lambda R, k, i: float(np.mean([v[k][i] for v in R.values()])) if R else float("nan")
        # true discoveries = n * (1 - FDP)
        td = lambda v: v["lb1.645"][0] * (1 - v["lb1.645"][1])
        ks = [k for k in new if k in orc]
        eff = [sum(td(new[k]) for k in ks) / sum(td(orc[k]) for k in ks)] if ks and sum(td(orc[k]) for k in ks) > 0 else []
        sd = lambda R, k, i: float(np.std([v[k][i] for v in R.values()], ddof=1)) if len(R) > 1 else float("nan")
        r = dict(point=m(new, "point", 1), param=m(old, "lb2.45", 1), lb=m(new, "lb1.645", 1), lb_sd=sd(new, "lb1.645", 1), orc=m(orc, "lb1.645", 1),
                 lb_det=m(new, "lb1.645", 2), orc_det=m(orc, "lb1.645", 2), eff=float(np.mean(eff)) if eff else float("nan"),
                 n=(len(old), len(new), len(orc)), neff=len(ks), td_new=sum(td(new[k]) for k in ks), td_orc=sum(td(orc[k]) for k in ks))
        J[f"{lab}|{t}"] = r
        f = lambda x: "--" if x != x else f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.2f}"
        name = {"D4": "D4", "D5": "D5", "D6-LV": "Las Vegas"}[lab]
        rows.append(f"{name if first else ''} & {tl} & {f(r['point'])} & {f(r['param'])} & {f(r['lb'])}$\\pm${f(r['lb_sd'])} & {f(r['lb_det'])} & {f(r['orc'])} & {f(r['eff'])} \\\\")
        first = False
json.dump(J, open(f"{H}/out_ceb/semisyn_table2.json", "w"), indent=1)
open(f"{H}/out_ceb/semisyn_table2.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
for k, v in J.items(): print(k, v["n"], v["neff"], round(v["lb_det"], 3), round(v["orc_det"], 3))
