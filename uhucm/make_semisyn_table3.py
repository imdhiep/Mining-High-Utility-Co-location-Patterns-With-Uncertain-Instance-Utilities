"""Semi-synthetic calibration table from the runs of fair/jobs.py (fair_runs/semi_*/stdout.txt).

For every (layout, truth, K) the first 10 seeds whose true scores contain at least K patterns are used; validity depends
on the truth only, so every method and ablation is scored on the same draws (fair/check.py rule R3). The threshold gamma_K is the K-th largest TRUE score,
so all methods are scored against the same truth set. Columns (lower-bound rule, z = 1.645 for every method):
Point FDP, parametric Gamma-Poisson FDP, UHUCPM FDP (mean and SD over runs), UHUCPM detection, oracle FDP, and
efficiency = true patterns found by UHUCPM / true patterns found by the oracle (ratio of sums over the common runs).
Also summarises the ablations and the spatial-field run (New York and Tokyo, log-normal truth).
Writes out_ceb/semisyn_table2.tex (rows used by fill_v2.py) and out_ceb/semisyn_m7.json."""
import json, os, numpy as np

H = os.path.dirname(os.path.abspath(__file__))
D = f"{H}/fair_runs"


def read(job, city, n=10):
    """records of a job, restricted to the first n valid seeds of every K"""
    fn = f"{D}/{job}/stdout.txt"; R = {}
    if not os.path.exists(fn): return R
    for l in open(fn):
        a = l.split(' ', 4)
        if len(a) < 5 or a[0] != city: continue
        try:
            v = json.loads(a[4])
        except Exception:
            continue
        if v: R[(int(a[2]), int(a[3]))] = v
    keep = {(s, K) for K in {k for _, k in R} for s in sorted(s for s, k in R if k == K)[:n]}
    return {k: v for k, v in R.items() if k in keep}


LAY = [("D4", "NYC"), ("D5", "TKY")]
TR = [("lognormal", "Log-normal"), ("gamma", "Gamma"), ("mixture", "Mixture")]
td = lambda v, key="lb1.645": v[key][0] * (1 - v[key][1])           # true discoveries = n (1 - FDP)
mean = lambda R, k, i: float(np.mean([v[k][i] for v in R.values()])) if R else float("nan")
sd = lambda R, k, i: float(np.std([v[k][i] for v in R.values()], ddof=1)) if len(R) > 1 else float("nan")


def summ(R):
    return dict(n=len(R), fdp=mean(R, "lb1.645", 1), fdp_sd=sd(R, "lb1.645", 1), det=mean(R, "lb1.645", 2),
                found=float(sum(v["lb1.645"][0] for v in R.values())), true=float(sum(td(v) for v in R.values())))


rows, J = [], {"main": {}, "abl": {}}
f = lambda x: "--" if x != x else f"{x:.3f}".lstrip("0") if abs(x) < 1 else f"{x:.2f}"
for lab, c in LAY:
    first = True
    for t, tl in TR:
        new, orc, par = (read(f"semi_{c}_{t}_{m}", c) for m in ("np", "oracle", "gamma"))
        ks = sorted(set(new) & set(orc) & set(par))                 # identical (seed, K) runs for every method
        if not ks: continue
        new, orc, par = ({k: R[k] for k in ks} for R in (new, orc, par))
        r = dict(runs=len(ks), point=mean(new, "point", 1), param=mean(par, "lb1.645", 1), param_det=mean(par, "lb1.645", 2),
                 lb=mean(new, "lb1.645", 1), lb_sd=sd(new, "lb1.645", 1), lb_det=mean(new, "lb1.645", 2),
                 orc=mean(orc, "lb1.645", 1), orc_det=mean(orc, "lb1.645", 2),
                 eff=sum(td(v) for v in new.values()) / max(1e-9, sum(td(v) for v in orc.values())))
        J["main"][f"{lab}|{t}"] = r
        name = {"D4": "D4", "D5": "D5"}[lab]
        rows.append(f"{name if first else ''} & {tl} & {f(r['point'])} & {f(r['param'])} & {f(r['lb'])}$\\pm${f(r['lb_sd'])} & "
                    f"{f(r['lb_det'])} & {f(r['orc'])} & {f(r['eff'])} \\\\")
        first = False
    base = read(f"semi_{c}_lognormal_np", c)
    for tag, job in [("median_center", "abl_median"), ("plugin_Z", "abl_plugZ"), ("no_VT", "abl_noVT"),
                     ("venue_boot", "abl_venueboot"), ("spatial_field", "spf")]:
        R = read(f"semi_{c}_lognormal_{job}", c)
        ks = sorted(set(R) & set(base)) if tag != "spatial_field" else sorted(R)
        if not ks: continue
        J["abl"][f"{lab}|{tag}"] = dict(variant=summ({k: R[k] for k in ks}), default=summ({k: base[k] for k in ks if k in base}))
        if tag == "spatial_field" and all("lb_band" in R[k] for k in ks):
            J["abl"][f"{lab}|{tag}"]["banded"] = dict(fdp=mean({k: R[k] for k in ks}, "lb_band", 1), det=mean({k: R[k] for k in ks}, "lb_band", 2))
os.makedirs(f"{H}/out_ceb", exist_ok=True)
json.dump(J, open(f"{H}/out_ceb/semisyn_m7.json", "w"), indent=1)
json.dump(J["main"], open(f"{H}/out_ceb/semisyn_table2.json", "w"), indent=1)
open(f"{H}/out_ceb/semisyn_table2.tex", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
for k, v in J["abl"].items():
    a, b = v["variant"], v["default"]
    print(f"{k:28s} runs={a['n']:2d}  FDP {b['fdp']:.3f} -> {a['fdp']:.3f}   det {b['det']:.3f} -> {a['det']:.3f}   true found {b['true']:.0f} -> {a['true']:.0f}")
