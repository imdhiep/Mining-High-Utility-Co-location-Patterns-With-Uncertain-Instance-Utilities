"""Table bodies for the revised Section VI (D1-D5 + D6 + EviGen). Prints a JSON dict name -> LaTeX rows."""
import json, os, math
import numpy as np
import results_db
H = os.path.dirname(os.path.abspath(__file__))
db, syn, d8 = results_db.load()
ORD = [("Phi", "D1", "Philadelphia"), ("New", "D2", "New_Orleans"), ("Tam", "D3", "Tampa"), ("FSQ_NYC", "D4", None), ("FSQ_TKY", "D5", None)]
CITY = {"Phi": "Philadelphia", "New": "New Orleans", "Tam": "Tampa", "FSQ_NYC": "New York", "FSQ_TKY": "Tokyo"}
REV = {}
for k, lab, c in ORD:
    if c and os.path.exists(os.path.join(H, "results_kaggle", f"rev_{c}.json")):
        REV[k] = json.load(open(os.path.join(H, "results_kaggle", f"rev_{c}.json")))
        for q in REV[k]["d50"]:
            db[k]["quality"].setdefault(int(q["K"]), {}).update({x: q[x] for x in q if x.startswith(("hits_", "n_", "filters"))})
    elif not c:
        REV[k] = None
f2 = lambda x, d=2: "--" if x is None else f"{x:.{d}f}"
num = lambda x: f"{x:,}"
def wilson(h, n):
    from scipy.stats import norm
    z = norm.ppf(0.975); p = h / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - w, c + w
T = {}
T["stats"] = "\n".join(f"{lab} & {CITY[k]} & {num(db[k]['stats']['n'])} & {num(db[k]['stats']['cliques'])} & {db[k]['stats']['omega']} & {db[k]['stats']['ev_median']:.0f} & {db[k]['stats']['insig_1645']:.2f} & {db[k]['stats']['neg']:.2f} \\\\" for k, lab, _ in ORD)
rows = []
for k, lab, _ in ORD:
    c = []
    for K in (100, 300, 1000):
        q = db[k]["quality"][K]; c.append(f"{q['R1.645']}/{q['R4.36']}")
    q = db[k]["quality"][300]
    rows.append(f"{lab} & " + " & ".join(c) + f" & {q['size_kept']:.1f}/{q['size_rej']:.1f} & {q['atoms_kept']:.0f}/{q['atoms_rej']:.0f} & {q['FE_miss']:.2f} \\\\")
T["e0"] = "\n".join(rows)
# E9 predictive validity with Wilson CIs
rows = []
for K in (100, 300, 1000):
    for k, lab, _ in ORD:
        q = db[k]["quality"][K]
        if "hits_kept" not in q: rows.append(f"{lab} & {K} & \\multicolumn{{4}}{{c}}{{pending}} \\\\"); continue
        cell = lambda g: f"{q['hits_'+g]/q['n_'+g]:.2f} [{wilson(q['hits_'+g], q['n_'+g])[0]:.2f}, {wilson(q['hits_'+g], q['n_'+g])[1]:.2f}]"
        p = q["perm_p"]; ps = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        rows.append(f"{lab} & {K} & {cell('kept')} & {cell('matched')} & {cell('rej')} & {ps} \\\\")
    if K != 1000: rows.append("\\midrule")
T["e9"] = "\n".join(rows)
# filters at K = 300
rows = []
for k, lab, _ in ORD:
    q = db[k]["quality"][300]; F = q.get("filters")
    if not F: rows.append(f"{lab} & \\multicolumn{{6}}{{c}}{{pending}} \\\\"); continue
    cells = [f"{F[x]['J']:.2f} & {f2(F[x]['hold_only_f'])}/{f2(F[x]['hold_only_z'])}" for x in ("minpi", "pi", "ev")]
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
T["filters"] = "\n".join(rows)
# calibration
e2 = syn["e2"]["results"]; e2b = json.load(open(os.path.join(H, "out_synth", "e2b.json")))["results"]
fd = lambda x: f"{x['FDP']:.3f} ({x['FDP_sd']:.3f})"
rows = []
for kn, nice in (("correct", "Correct"), ("estimated", "Estimated"), ("t3", "$t_3$ noise"), ("prior+0.3", "Prior $+0.3\\tau$"), ("prior-0.3", "Prior $-0.3\\tau$"),
                 ("sigma x0.7", "$\\sigma_e\\times0.7$"), ("sigma x1.3", "$\\sigma_e\\times1.3$"), ("spatial", "Spatial 0.5")):
    x = e2[kn]; rows.append(f"{nice} & {x['z=0 (shrunk)']['FDP']:.3f} & {fd(x['normal d=0.05'])} & {x['subG d=0.05']['FDP']:.3f} & {x['Cantelli d=0.05']['FDP']:.3f} & {x['z=0 (shrunk)']['power']:.2f}/{x['normal d=0.05']['power']:.2f} \\\\")
rows.append("\\midrule")
for kn, nice in (("activity (Def. 4)", "Activity (Def.~\\ref{def:relative})"), ("spatial 0.1", "Spatial 0.1"), ("n=1000, k=10", "$n$=1000, $k$=10"), ("truth 10%", "Truth 10\\%")):
    x = e2b[kn]; rows.append(f"{nice} & {x['z=0 (shrunk)']['FDP']:.3f} & {fd(x['normal d=0.05'])} & {x['subG d=0.05']['FDP']:.3f} & {x['Cantelli d=0.05']['FDP']:.3f} & {x['z=0 (shrunk)']['power']:.2f}/{x['normal d=0.05']['power']:.2f} \\\\")
T["e2"] = "\n".join(rows)
# sensitivity
yv = [("kappa_x0.25", "$\\kappa_r\\times 0.25$"), ("kappa_x4", "$\\kappa_r\\times 4$"), ("sigma_x0.7", "$\\sigma_s\\times 0.7$"), ("sigma_x1.3", "$\\sigma_s\\times 1.3$"), ("g_const", "$w\\equiv 1$"), ("s0_median", "$s_0=$ median")]
fv = [("alpha_x0.5", "$\\alpha_f\\times0.5$"), ("alpha_x2", "$\\alpha_f\\times2$"), ("quasi", "Quasi-Poisson")]
rows = []
for i, (v, nice) in enumerate(yv):
    left = f"{nice} & " + " & ".join(f"{db[k]['sensitivity'][v]['J']:.2f}" for k in ("Phi", "New", "Tam"))
    right = (f"{fv[i][1]} & " + " & ".join(f"{db[k]['sensitivity'][fv[i][0]]['J']:.2f}" for k in ("FSQ_NYC", "FSQ_TKY"))) if i < len(fv) else " & & "
    rows.append(left + " & " + right + " \\\\")
T["sens"] = "\n".join(rows)
# rho
rows = []
for k, lab, _ in ORD:
    dpd = db[k]["dependence"]; rr = dpd["rows"]; rh = dpd["rho_hat"]
    cell = lambda r: (lambda x: f"{x['R']} ({num(x['evaluated'])})")(rr[min(rr, key=lambda t: abs(t - r))])
    rows.append(f"{lab} & {rh:.3f} & {cell(0.0)} & {cell(rh)} & {cell(0.3)} & {cell(0.6)} \\\\")
T["rho"] = "\n".join(rows)
# distance (E6) D1-D5
rows = []
for k, lab, c in ORD:
    for dd in (30, 50, 70):
        if dd == 50: q = db[k]["quality"][300]; C = db[k]["stats"]["cliques"]
        else:
            d = syn["dsens"].get((k, dd))
            if d is None and REV.get(k) and f"d{dd}" in REV[k]:
                q = [x for x in REV[k][f"d{dd}"] if int(x["K"]) == 300][0]; C = q.get("cliques")
            elif d is None: continue
            else: q = d["quality"][300]; C = d["stats"]["cliques"]
        hk = q.get("hold_kept"); hr = q.get("hold_rej"); hm = q.get("hold_matched")
        rows.append(f"{lab} & {dd} & {num(C) if C else '--'} & {q['R1.645']} & {q['R4.36']} & {f2(hk)} & {f2(hm)} & {f2(hr)} & {f2(q['FE_miss'])} \\\\")
T["dsens"] = "\n".join(rows)
# ablation
ds = [k for k, _, _ in ORD]
rows = [f"{nice} & " + " & ".join(num(int(db[k]["ablation"][v]["evaluated"])) for k in ds) + " \\\\" for v, nice in
        (("full", "UHUCM (exact bound)"), ("chord", "Chord bound only"), ("mean", "Mean bound"), ("noSB", "Without SB"), ("noprune", "NoPrune"))]
rows.append("\\midrule")
rows += [f"{nice} & " + " & ".join(f"{float(db[k]['ablation'][v]['time']):.2f}" for k in ds) + " \\\\" for v, nice in
         (("full", "UHUCM time (s)"), ("mean", "Mean bound time"), ("noprune", "NoPrune time"))]
T["ablation"] = "\n".join(rows)
# D6
pos = sorted([r for r in d8 if "orig" in r and r.get("uhucm") and all(v > 0 for v in r["util"].values())], key=lambda r: (r["dataset"], r["minutil"]))
T["d6pos"] = "\n".join(f"{'CA' if 'California' in r['dataset'] else 'LV'} & {r['minutil']} & {r['uhucm']['n']} & {r['orig']['n']} & yes & {r['uhucm']['seconds']:.2f} & {r['orig']['seconds']:.2f} \\\\" for r in pos)
by = {}
for r in [r for r in d8 if "orig" in r and any(v < 0 for v in r["util"].values())]:
    ds_ = "CA" if "California" in r["dataset"] else "LV"; kk = sum(v < 0 for v in r["util"].values())
    b = by.setdefault((ds_, kk), dict(cfg=0, ex=0, miss=0, tot=0, eq=0)); b["cfg"] += 1
    if not r.get("uhucm"): b["ex"] += 1; continue
    b["miss"] += len(r.get("orig_missed", [])); b["tot"] += r["uhucm"]["n"]; b["eq"] += bool(r.get("nos4_equal_uhucm"))
T["d6neg"] = "\n".join(f"{d} & {k} & {b['cfg']} & {b['ex']} & {b['miss']}/{b['tot']} & {b['eq']}/{b['cfg']-b['ex']} \\\\" for (d, k), b in sorted(by.items()))
# E5
rows = []
for k in ("FSQ_NYC", "FSQ_TKY"):
    E = syn["ext"].get(k)
    if not E: continue
    for K in ("100", "300"):
        r = E["fl"].get(K) or E["fl"].get(int(K))
        if not r: continue
        rows.append(f"{'D4' if k=='FSQ_NYC' else 'D5'} & {K} & {r['J_fl_R0']:.2f} & {r['fl_inst_hucp']:.2f} & {r['hold_fl']:.2f} & {r['hold_R0']:.2f} & {r['hold_Rz']:.2f} \\\\")
T["e5"] = "\n".join(rows)
json.dump(T, open(os.path.join(H, "out_synth", "rev_tables.json"), "w"), indent=1)
for k, v in T.items(): print("==", k); print(v)
