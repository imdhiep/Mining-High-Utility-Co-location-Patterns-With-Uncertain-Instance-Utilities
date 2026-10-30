"""LaTeX rows for the second revision (Tables V, VII, VIII, IX), generated only from result files.
Writes out_synth/rev3_tables.json."""
import json, os, math
import numpy as np
from stats_rev2 import load, wilson, boot_diff, cluster_boot
H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(H, "out_synth", "stats_rev2.json")))
TM = json.load(open(os.path.join(H, "out_synth", "stats_timing.json"))) if os.path.exists(os.path.join(H, "out_synth", "stats_timing.json")) else {}
D = load(); T = {}
f2 = lambda x: f"{x:.2f}"
ci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
ORDER = ["D1", "D2", "D3", "D4", "D5", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8"]

# Table IX: hold rates at K = 300 per dataset, paired difference matched - rejected
rows = []
for lab in ORDER:
    r = S["per"].get(f"{lab}/K=300")
    if not r: continue
    cell = lambda g: f"{r[g]['rate']:.2f} {ci(r[g]['ci'])}"
    p = r["paired"]
    rows.append(f"{lab} & {cell('kept')} & {cell('matched')} & {cell('rej')} & {p['diff']:+.2f} {ci(p['ci'])} \\\\")
pl = S["pooled"]["300"]
rows.append("\\midrule")
rows.append(f"All & \\multicolumn{{3}}{{l}}{{Friedman $p={S['friedman']['300']['p']:.1e}$; ranks {S['friedman']['300']['mean_rank']['kept']:.2f}/{S['friedman']['300']['mean_rank']['matched']:.2f}/{S['friedman']['300']['mean_rank']['rej']:.2f}}} & {pl['diff']:+.2f} {ci(pl['ci_cluster'])} \\\\")
T["e9"] = "\n".join(rows)

# Table VII: filters pooled over datasets at K = 300
rows = []
nice = {"minpi": "Min.\\ participation", "pi": "Participation index", "ev": "Min.\\ evidence"}
for f in ("minpi", "pi", "ev"):
    F = S["filters"][f]; R = F["rows"]
    Js = [r["J"] for r in R.values() if r["J"] is not None]
    nf = sum(r["n_only_f"] for r in R.values()); nz = sum(r["n_only_z"] for r in R.values())
    hf = sum(round(r["hold_only_f"] * r["n_only_f"]) for r in R.values() if r["n_only_f"]); hz = sum(round(r["hold_only_z"] * r["n_only_z"]) for r in R.values() if r["n_only_z"])
    p = F["pooled"]
    rows.append(f"{nice[f]} & {min(Js):.2f}--{max(Js):.2f} & {hf/nf:.2f} ({nf}) & {hz/nz:.2f} ({nz}) & {p['diff']:+.2f} {ci(p['ci_cluster'])} ({p['n_pairs']}) \\\\")
T["filters"] = "\n".join(rows)

# Table V: E0 / E3 at the calibrated z (1.645 on Yelp, 2.45 on Foursquare)
rows = []
for lab in ("D1", "D2", "D3", "D4", "D5"):
    q = D[lab]["q"]
    c = [f"{q[K]['Rz']}/{q[K]['R436']}" for K in (100, 300, 1000)]
    x = q[300]
    if "size_kept" in x:
        extra = f"{x['size_kept']:.1f}/{x['size_rej']:.1f} & {x['atoms_kept']:.0f}/{x['atoms_rej']:.0f}"
    else:
        extra = None
    rows.append((lab, c, extra, x["FE_miss"]))
T["e0_fsq"] = [dict(lab=l, cells=c, extra=e, fe=fe) for l, c, e, fe in rows]

# Table VIII: ablation nodes and 10-run mean (SD) times at K = 300 (binary-search implementation, d = 50 m, k = 20)
P2 = os.path.join(H, "out_synth", "stats_timing2.json")
if os.path.exists(P2):
    T2 = json.load(open(P2))
    labs = [l for l in ("D1", "D2", "D3", "D4", "D5") if f"{l}|50|20" in T2 and "nosearch" in T2[f"{l}|50|20"]["rows"]["300"]["table"]]
    g = lambda l, v: T2[f"{l}|50|20"]["rows"]["300"]["table"][v]
    def node(v): return " & ".join(f"{g(l, v)[2]:,}" for l in labs)
    def tim(v): return " & ".join(f"{g(l, v)[0]:.2f}$\\pm${g(l, v)[1]:.2f}" for l in labs)
    names = [("exact", "UHUCM"), ("chord", "Chord bound only"), ("mean", "Mean bound only"), ("nosearch", "Linear suffix scan"),
             ("noSB", "Without SB"), ("noprune", "NoPrune")]
    rows = [f"{n} & {node(v)} \\\\" for v, n in names if v != "nosearch"] + ["\\midrule"] + [f"{n} & {tim(v)} \\\\" for v, n in names]
    T["ablation"] = "\n".join(rows); T["ablation_labs"] = labs
json.dump(T, open(os.path.join(H, "out_synth", "rev3_tables.json"), "w"), indent=1)
print(T["e9"]); print(T["filters"]); print(T.get("ablation", ""))
