"""Semi-synthetic table (layouts D4, D5, Las Vegas; three truth families; K = 100, 300, 1000; 3 seeds).
Old: Gamma-Poisson model of the earlier version (median center, plug-in normalizer), z = 2.45, scored against its own
median-centered truth.  New: compound EB with total variance and expected normalizer (moment-centered truth); Point =
all of R0; LB = z 1.645; BFDR = optimal set at delta 0.05.  Oracle: posterior under the true prior and center, BFDR 0.05.
Eff = true discoveries of BFDR / those of the oracle, averaged over seeds and K.  Writes out_ceb/semisyn_table.tex/json."""
import json, glob, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__))


def read(fn, city):
    R = {}
    if not os.path.exists(fn): return R
    for l in open(fn):
        a = l.split(' ', 4)
        if len(a) < 5 or a[0] != city: continue
        try: R[(a[2], a[3])] = json.loads(a[4])
        except Exception: pass
    return R


LAY = [("D4", "NYC"), ("D5", "TKY"), ("D6-LV", "LasVegas")]
TR = [("lognormal", "Log-normal"), ("gamma", "Gamma"), ("mixture", "Mixture")]
rows, J = [], {}
m = lambda R, k, i: np.mean([v[k][i] for v in R.values()]) if R else float("nan")
for lab, c in LAY:
    for t, tl in TR:
        old = read(f"{H}/out_semisyn/own_{c}_{t}_gamma.txt", c); new = read(f"{H}/out_semisyn/log_{c}_{t}_np_cboot.txt", c)
        orc = read(f"{H}/out_semisyn/log_{c}_{t}_oracle.txt", c)
        eff = [new[k]["bfdr0.05"][2] / orc[k]["bfdr0.05"][2] for k in new if k in orc and orc[k]["bfdr0.05"][2] > 0]
        r = dict(old_fdp=m(old, "lb2.45", 1), old_det=m(old, "lb2.45", 2), point=m(new, "point", 1), lb_fdp=m(new, "lb1.645", 1),
                 lb_det=m(new, "lb1.645", 2), bf_fdp=m(new, "bfdr0.05", 1), bf_det=m(new, "bfdr0.05", 2),
                 or_fdp=m(orc, "bfdr0.05", 1), or_det=m(orc, "bfdr0.05", 2), eff=float(np.mean(eff)) if eff else float("nan"),
                 n=(len(old), len(new), len(orc)))
        if not new: continue
        J[f"{lab}|{t}"] = r
        f = lambda x: "--" if x != x else f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.2f}"
        rows.append(f"{lab if not any(k.startswith(lab) for k in list(J)[:-1]) else ''} & {tl} & {f(r['old_fdp'])} & {f(r['point'])} & {f(r['lb_fdp'])}/{f(r['lb_det'])} & "
                    f"{f(r['bf_fdp'])}/{f(r['bf_det'])} & {f(r['or_fdp'])}/{f(r['or_det'])} & {f(r['eff'])} \\\\")
os.makedirs(f"{H}/out_ceb", exist_ok=True)
json.dump(J, open(f"{H}/out_ceb/semisyn_table.json", "w"), indent=1)
open(f"{H}/out_ceb/semisyn_table.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
for k, v in J.items(): print(k, v["n"])
