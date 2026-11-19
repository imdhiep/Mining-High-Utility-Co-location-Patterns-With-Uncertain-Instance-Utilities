"""Tables, figures and numbers of Section VI, generated only from the result summaries (no hand-entered numbers).

Inputs (fair_runs/kaggle_alexduongvn/summaries unless stated):
  redo_yelp_fc867c3.json   accuracy and diagnostics on Philadelphia, New Orleans, Tampa (smoothed prior)
  redo_fsq_fc867c3.json    accuracy on New York, diagnostics, hold rates on New York and Tokyo
  phl_law_c.json           accuracy under the three laws on Philadelphia (compact copy), optional
  nyc_law.json, nyc_k40.json  local reruns on New York (law job and k = 40), optional
  hold_all.json, yelp holdcurve (inside diag_local_and_hold.json) for the Yelp cities (rating model, unchanged)
  bench_FSQ_fc867c3_mean.json and bench_Yelp_<tag>.json   running times
Outputs: FIG_OUT (figures), paper_out/tables.tex (LaTeX tables, one block per marker) and paper_out/nums.json.
Usage: python paper_exp.py [YELP_BENCH_FILE]"""
import os, sys, json
import numpy as np
import figstyle as fs
from figstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "fair_runs", "kaggle_alexduongvn", "summaries")
OUT = os.environ.get("FIG_OUT", os.path.join(HERE, "paper_out", "figures")); os.makedirs(OUT, exist_ok=True)
POUT = os.path.join(HERE, "paper_out"); os.makedirs(POUT, exist_ok=True)
W1, W2 = 3.45, 7.1
NAME = {"Philadelphia": "Philadelphia", "New_Orleans": "New Orleans", "Tampa": "Tampa", "NYC": "New York", "TKY": "Tokyo",
        "Tucson": "Tucson", "Indianapolis": "Indianapolis", "Nashville": "Nashville", "Reno": "Reno", "Saint_Louis": "St. Louis",
        "Edmonton": "Edmonton", "Santa_Barbara": "Santa Barbara", "Boise": "Boise"}
METH = ["uhucpm", "uhucpm_q05", "uhucpm_q10", "cq", "pso_topk"]
MLAB = {"uhucpm": "UHUCPM", "uhucpm_q05": r"UHUCPM-S$_{0.05}$", "uhucpm_q10": r"UHUCPM-S$_{0.10}$", "cq": "CQ-HUCPM",
        "pso_topk": "PSO-MUCP"}
COL = {"uhucpm": fs.BLUE, "uhucpm_q05": fs.LBLUE, "uhucpm_q10": fs.GREEN, "cq": fs.ORANGE, "pso_topk": fs.GRAY}
nums = {}
J = lambda f: json.load(open(os.path.join(S, f)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# ------------------------------------------------------------------ accuracy results: value -> method -> [FDP, Det], n
yelp, fsq = J("redo_yelp_fc867c3.json"), J("redo_fsq_fc867c3.json")
ACC = {}
for job, rows in yelp["acc"].items():
    L, V = job[4:].rsplit("_", 1); ACC[(L, V)] = {v: r for v, r in rows["rows"].items()}
for job, rows in fsq["acc"].items():
    L, V = job[4:].rsplit("_", 1); ACC[(L, V)] = {v: r for v, r in rows.items() if not r.get("incomplete")}


def from_runs(path, vary):
    """mean FDP and detection per value from a full acc_*.json (local rerun) or a compact *_c.json"""
    d = json.load(open(path)); g = {}
    for key, r in d["runs"].items():
        if isinstance(r, list):                                     # compact copy: [nT, [n, fdp, det] x methods]
            rec = {m: dict(n=x[0], fdp=x[1], det=x[2]) for m, x in zip(d["methods"], r[1:])}
        else:
            if not all(m in r for m in METH): continue
            rec = r
        p = key.split("|"); g.setdefault("|".join(p[:-1]), []).append(rec)
    return {v: dict(n=len(rs), **{m: [float(np.mean([x[m]["fdp"] for x in rs])), float(np.mean([x[m]["det"] for x in rs]))] for m in METH})
            for v, rs in g.items()}


for L, V, f in (("Philadelphia", "law", "phl_law_c.json"), ("NYC", "law", "nyc_law.json"), ("NYC", "k40", "nyc_k40.json")):
    p = os.path.join(S, f)
    if os.path.exists(p):
        if V == "k40": ACC[("NYC", "k")].update(from_runs(p, "k"))
        else: ACC[(L, V)] = from_runs(p, V)
    else: print("missing", f)


def cell(L, V, v, m, i):
    r = ACC.get((L, V), {}).get(v); return None if r is None else r[m][i]


# ------------------------------------------------------------------ Table: accuracy under the three laws (main table)
def tab_law():
    out = [r"\begin{tabular}{@{}llr" + "rr" * len(METH) + "@{}}", r"\toprule",
           r" & & & " + " & ".join(r"\multicolumn{2}{c}{" + {"uhucpm_q05": r"S$_{0.05}$", "uhucpm_q10": r"S$_{0.10}$"}.get(m, MLAB[m]) + "}" for m in METH) + r" \\",
           " ".join(r"\cmidrule(lr){%d-%d}" % (4 + 2 * i, 5 + 2 * i) for i in range(len(METH))),
           r"City & Law & $K$ & " + " & ".join(["FDP & Det"] * len(METH)) + r" \\", r"\midrule"]
    for L in ("Philadelphia", "NYC"):
        for law in ("lognormal", "gamma", "mixture"):
            for K in (100, 300, 1000):
                r = ACC.get((L, "law"), {}).get(f"{law}|{K}")
                vals = " & ".join(("--" if r is None else f"{r[m][0]:.3f} & {r[m][1]:.2f}") for m in METH)
                out.append(f"{NAME[L] if law == 'lognormal' and K == 100 else ''} & {law.capitalize() if K == 100 else ''} & {K} & {vals} \\\\")
            if law != "mixture": out.append(r"\addlinespace[1pt]")
        if L == "Philadelphia": out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# ------------------------------------------------------------------ Figure: one parameter at a time
PARAMS = [("d", "New_Orleans"), ("k", "NYC"), ("frac", "NYC"), ("c", "Tampa"), ("sd", "NYC")]   # Fig. 3: one panel per parameter
PLAB = {"K": "threshold $K$", "d": "distance threshold $d$ (m)", "k": "number of features", "frac": "share of instances", "c": "evidence factor $c$", "sd": "spread of log-rates"}


def fig_params():
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(W2, 1.95))
    for i, (V, L) in enumerate(PARAMS):
        ax = axes[i]; rows = ACC.get((L, V), {})
        xs = sorted(rows, key=float); x = [float(v) for v in xs]
        for m in ("uhucpm", "uhucpm_q05", "cq"):
            ax.plot(x, [rows[v][m][1] for v in xs], color=COL[m], marker="o", ms=2.5, lw=1.0, ls="-", label=MLAB[m] + ", detection")
            ax.plot(x, [rows[v][m][0] for v in xs], color=COL[m], marker="x", ms=2.5, lw=1.0, ls="--", label=MLAB[m] + ", FDP")
        ax.set_ylim(0, 1); ax.set_xticks(x); ax.tick_params(labelsize=6); ax.grid(True, lw=0.4, alpha=0.5)
        ax.set_xlabel(PLAB[V][0].upper() + PLAB[V][1:], fontsize=6.5, labelpad=1)
        if i == 0: ax.set_ylabel("FDP and detection", fontsize=6.5, labelpad=1)
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.3, left=0.055, right=0.99, top=0.80, bottom=0.30)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=5.8, frameon=False, handlelength=1.8, columnspacing=0.9)
    for i, (ax, (V, L)) in enumerate(zip(axes, PARAMS)):
        bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, 0.005, f"({chr(97 + i)}) {NAME[L]}", ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig_params.pdf")


# ------------------------------------------------------------------ Figure: FDP against detection
def fig_tradeoff():
    cur = {"Philadelphia": yelp["curves"]["diag_Philadelphia"], "Tampa": yelp["curves"]["diag_Tampa"],
           "NYC": fsq["diag"]["NYC"], "TKY": fsq["diag"]["TKY"]}
    bf = {"Philadelphia": yelp["diag"]["Philadelphia"]["bfdr"], "Tampa": yelp["diag"]["Tampa"]["bfdr"],
          "NYC": fsq["diag"]["NYC"]["bfdr"], "TKY": fsq["diag"]["TKY"]["bfdr"]}
    orc = {}
    if os.path.exists(os.path.join(S, "diag_oracle.json")):              # runs with the oracle and minimum-count curves
        for L, v in J("diag_oracle.json")["layouts"].items():
            cur[L] = v["curves"]; bf[L] = v["curves"]["bfdr"]; orc[L] = v["curves"]
    else: print("missing diag_oracle.json")
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.9)); caps = []
    for ax, (L, c) in zip(axes, cur.items()):
        lb, pt = np.array(c["lb"]), np.array(c["point"])
        if L in orc:
            ob = np.array(orc[L]["oracle_bfdr"])
            ax.plot(ob[:, 2], ob[:, 1], color=fs.GRAY, marker="x", ms=2.5, lw=1.0, ls="--", label=r"Oracle posterior, $\delta$ varied")
        ax.plot(lb[:, 2], lb[:, 1], color=COL["uhucpm"], marker="o", ms=2.5, lw=1.0, label="Lower bound, $z$ varied")
        ax.plot(pt[:, 2], pt[:, 1], color=COL["cq"], marker="s", ms=2.5, lw=1.0, label=r"Point score, threshold varied")
        zi = [r[0] for r in c["lb"]].index(1.645)
        ax.plot([lb[zi, 2]], [lb[zi, 1]], marker="*", ms=7, color=COL["uhucpm"], ls="none", label="UHUCPM ($z^*$)")
        ax.plot([pt[0, 2]], [pt[0, 1]], marker="*", ms=7, color=COL["cq"], ls="none", label="CQ-HUCPM")
        b = np.array(bf[L]); qi = [r[0] for r in bf[L]]
        ax.plot([b[qi.index(0.05), 2], b[qi.index(0.1), 2]], [b[qi.index(0.05), 1], b[qi.index(0.1), 1]], marker="D", ms=3.5,
                color=COL["uhucpm_q05"], ls="none", label=r"UHUCPM-S$_{0.05}$, S$_{0.10}$")
        ax.set_xlabel("Detection", fontsize=7, labelpad=1); ax.set_ylabel("FDP", fontsize=7, labelpad=1); ax.tick_params(labelsize=6)
        caps.append(f"({chr(97 + len(caps))}) {NAME[L]}")
    h, l = [], []
    for ax in axes:                                                      # legend entries of all panels, once each
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.99, top=0.87, bottom=0.27)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=6, frameon=False, handlelength=1.8, columnspacing=1.0)
    for ax, cap in zip(axes, caps):
        bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, 0.005, cap, ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig_tradeoff.pdf")
    # numbers for the text: FDP of the lower-bound rule and of the point rule at matched detection (linear interpolation)
    for L, c in cur.items():
        lb, pt = np.array(c["lb"]), np.array(c["point"])
        dz = lb[[r[0] for r in c["lb"]].index(1.0), 2]                      # detection of the lower bound at z = 1
        o = np.argsort(pt[:, 2]); fpt = float(np.interp(dz, pt[o, 2], pt[o, 1]))
        nums[f"tradeoff_{L}"] = dict(det=round(float(dz), 3), fdp_lb=round(float(lb[[r[0] for r in c['lb']].index(1.0), 1]), 3), fdp_point=round(fpt, 3))
        if L in orc:
            q5 = lambda rows: [r for r in rows if r[0] == 0.05][0]
            mins = [r[1] for k in ("min2", "min5", "min10") for r in orc[L][k] if r[3] > 0]
            nums[f"oracle_{L}"] = dict(ours=q5(orc[L]["bfdr"])[1:3], oracle=q5(orc[L]["oracle_bfdr"])[1:3],
                                       min_fdp=[round(min(mins), 3), round(max(mins), 3)])


# ------------------------------------------------------------------ hold rates at matched size
def hold():
    hc_y = J("diag_local_and_hold.json")["holdcurve"]["yelp"]
    hc = {L: {K: {n: v for n, v in rows["h"].items()} for K, rows in d["K"].items()} for L, d in hc_y.items()}
    for L in ("NYC", "TKY"):
        hc[L] = {K: {n: v for n, v in rows.items()} for K, rows in fsq["holdcurve"][L].items() if K != "available"}
    order = ["Philadelphia", "New_Orleans", "Tampa", "NYC", "TKY", "Tucson", "Indianapolis", "Nashville", "Reno", "Saint_Louis",
             "Edmonton", "Santa_Barbara", "Boise"]
    out = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"City & UHUCPM & CQ-HUCPM & PSO-MUCP \\", r"\midrule"]
    better = tie = worse = bp = tp = wp = 0
    for L in order:
        r = hc[L]["300"]["100"]
        vals = [("--" if x is None else f"{x:.2f}") for x in r]
        best = max(x for x in r if x is not None)
        vals = [(r"\textbf{" + v + "}" if (x is not None and abs(x - best) < 1e-9) else v) for v, x in zip(vals, r)]
        out.append(f"{NAME[L]} & " + " & ".join(vals) + r" \\")
        if L == "TKY": out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    # counts over all cells with n <= 2K for the text
    for L in order:
        for K, rows in hc[L].items():
            for n, x in rows.items():
                if x[0] is None or int(n) > 2 * int(K): continue
                if x[1] is not None:
                    d = x[0] - x[1]; better += d > 0.005; worse += d < -0.005; tie += abs(d) <= 0.005
                if x[2] is not None:
                    d = x[0] - x[2]; bp += d > 0.005; wp += d < -0.005; tp += abs(d) <= 0.005
    nums["hold_vs_cq"] = dict(better=int(better), tie=int(tie), worse=int(worse))
    nums["hold_vs_pso"] = dict(better=int(bp), tie=int(tp), worse=int(wp))
    # own answer sets: UHUCPM against CQ-HUCPM (Yelp cities from hold_all, Foursquare from the rerun)
    ha = J("hold_all.json")["data"]; own = []
    for L in order:
        if L in ("NYC", "TKY"):
            for K, r in fsq["hold"][L].items(): own.append(r["uhucpm"][1] > r["cq"][1])
        else:
            for K, r in ha[L].items():
                if isinstance(r, list): own.append(r[1] > r[3])
    nums["hold_own_uhucpm_gt_cq"] = [int(sum(own)), len(own)]
    # figure: hold rate against n on Philadelphia and New York, K = 300
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.9)); caps = []
    for ax, L in zip(axes, ("Philadelphia", "NYC")):
        rows = hc[L]["300"]; ns = sorted((int(n) for n in rows if int(n) <= 500))
        for j, m in enumerate(("uhucpm", "cq", "pso_topk")):
            ys = [rows[str(n)][j] for n in ns]; xx = [n for n, y in zip(ns, ys) if y is not None]; yy = [y for y in ys if y is not None]
            ax.plot(xx, yy, color=COL[m], marker="o", ms=2.5, lw=1.0, label=MLAB[m])
        ax.set_xscale("log"); ax.set_xticks([25, 50, 100, 200, 500]); ax.set_xticklabels(["25", "50", "100", "200", "500"]); ax.minorticks_off()
        ax.set_ylim(0, 1.02); ax.set_xlabel("Number of top patterns $n$", fontsize=7, labelpad=1)
        ax.set_ylabel("Hold rate", fontsize=7, labelpad=1); ax.tick_params(labelsize=6); caps.append(f"({chr(97 + len(caps))}) {NAME[L]}")
        ax.grid(True, lw=0.4, alpha=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.45, left=0.13, right=0.98, top=0.87, bottom=0.27)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6, frameon=False, handlelength=1.8, columnspacing=1.0)
    for ax, cap in zip(axes, caps):
        bb = ax.get_position(); fig.text((bb.x0 + bb.x1) / 2, 0.005, cap, ha="center", va="bottom", fontsize=7.5)
    save(fig, "fig_hold.pdf")
    return "\n".join(out)


# ------------------------------------------------------------------ running time
def bench(yelp_file):
    """Fig. time: running time of UHUCPM, CQ-HUCPM and PSO-MUCP when one parameter changes (blank = did not finish)."""
    B = J("bench_FSQ_fc867c3_mean.json")["data"]
    for k, v in J("bench_syn.json")["data"].items(): B[k] = {K: dict(cmp=r["cmp"], **r["t"]) for K, r in v.items() if K != "n"}
    if yelp_file and os.path.exists(os.path.join(S, yelp_file)):
        for k, v in J(yelp_file)["data"].items(): B[k] = {K: dict(cmp=r["cmp"], **r["t"]) for K, r in v.items() if K != "n"}
    else: print("missing Yelp timing file", yelp_file)
    t = lambda key, K, p: (B.get(key, {}).get(str(K), {}) or {}).get(p)
    # one figure per varied parameter; the others stay at their defaults (stated in the caption)
    nm = dict(NAME, syn="Synthetic")
    figs = {
        "fig_time_K": ("Threshold $K$", [("New_Orleans", [(f"New_Orleans_k20_d50_f100", K, K) for K in (100, 300, 1000)]),
                                             ("Tampa", [(f"Tampa_k20_d50_f100", K, K) for K in (100, 300, 1000)]),
                                             ("NYC", [(f"NYC_k20_d50_f100", K, K) for K in (100, 300, 1000)])]),
        "fig_time_d": ("Distance threshold $d$ (m)", [("Tampa", [(f"Tampa_k20_d{d}_f100", 100, d) for d in (30, 50, 70)]),
                                                     ("NYC", [(f"NYC_k20_d{d}_f100", 100, d) for d in (30, 50, 70)])]),
        "fig_time_k": ("Number of features", [("Tampa", [(f"Tampa_k{k}_d50_f100", 100, k) for k in (12, 16, 20)]),
                                                 ("NYC", [(f"NYC_k{k}_d50_f100", 100, k) for k in (20, 40, 60, 80)])]),
        "fig_time_n": (None, [("New_Orleans", [(f"New_Orleans_k20_d50_f{f}", 100, f) for f in (25, 50, 75, 100)]),
                              ("TKY", [(f"TKY_k20_d50_f{f}", 100, f) for f in (25, 50, 75, 100)]),
                              ("syn", [(f"syn_n{n}", 100, n // 1000) for n in range(20000, 120001, 20000)])])}
    MK = {"uhucpm": ("o", fs.BLUE, "UHUCPM"), "cq": ("s", fs.ORANGE, "CQ-HUCPM"), "pso": ("^", fs.GREEN, "PSO-MUCP")}
    spec = []
    for name, (xl, panels) in figs.items():
        wide = len(panels) == 3
        fig, axes = plt.subplots(1, len(panels), figsize=(W2 if wide else W1, 1.75 if wide else 1.6))
        for i, (ax, (L, pts)) in enumerate(zip(axes, panels)):
            spec.append((L, pts, xl))
            ys = []
            for p, (mk, c, lab) in MK.items():
                xy = [(x, t(key, K, p)[0]) for key, K, x in pts if t(key, K, p)]
                ax.plot([u for u, _ in xy], [v for _, v in xy], marker=mk, ms=3, lw=1.1, color=c, label=lab); ys += [v for _, v in xy]
            ax.set_yscale("log"); lo, hi = min(ys), max(ys)
            R = np.log10(hi / lo); ax.set_ylim(lo / 10 ** 0.15, hi * 10 ** (0.3 * R + 0.35))        # head room for the legend
            if xl == "Threshold $K$": ax.set_xscale("log"); ax.set_xticks([100, 300, 1000]); ax.set_xticklabels(["100", "300", "1000"]); ax.minorticks_off()
            else: ax.set_xticks([x for _, _, x in pts])
            xlab = xl or ("Number of instances ($\\times$1000)" if L == "syn" else "Proportion of instances (%)")
            ax.set_xlabel(xlab, fontsize=6.5, labelpad=1); ax.tick_params(labelsize=6)
            ax.set_ylabel("Running time (s)", fontsize=6.5, labelpad=1)
            ax.grid(True, which="major", lw=0.4, alpha=0.5)
            ax.legend(loc="upper center", ncol=3 if wide else 2, fontsize=5, frameon=True, framealpha=0.9, handlelength=1.6, columnspacing=0.8, borderpad=0.3)
        fig.subplots_adjust(wspace=0.38 if wide else 0.55, left=0.07 if wide else 0.15, right=0.99, top=0.98, bottom=0.27 if wide else 0.29)
        for i, (ax, (L, _)) in enumerate(zip(axes, panels)):
            bb = ax.get_position()
            fig.text((bb.x0 + bb.x1) / 2, 0.005, f"({chr(97 + i)}) {nm[L]}", ha="center", va="bottom", fontsize=7.5)
        save(fig, name + ".pdf")
    used = {key for _, pts, _ in spec for key, _, _ in pts}
    keys = [(k, K) for k in used for K in B.get(k, {}) if K != "n" and isinstance(B[k][K], dict)]
    nums["time_settings"] = len(keys)
    nums["time_slower"] = sorted(f"{k}@{K}" for k, K in keys if any(t(k, K, p) and t(k, K, p)[0] <= t(k, K, "uhucpm")[0] for p in ("cq", "pso")))
    eq = [B[k][K]["cmp"]["uhucpm0_equals_cq"] for k, K in keys if "uhucpm0_equals_cq" in B[k][K].get("cmp", {})]
    nums["z0_equals_cq"] = [int(sum(eq)), len(eq)]
    n80 = B["NYC_k80_d50_f100"]["100"]; nums["scored_k80"] = round(100 * n80["uhucpm"][3] / n80["noprune"][3], 1)
    return ""


if __name__ == "__main__":
    yf = sys.argv[1] if len(sys.argv) > 1 else "bench_Yelp_intel.json"
    tabs = {"LAW": tab_law(), "HOLD": hold(), "TIME": bench(yf)}
    fig_params(); fig_tradeoff()
    with open(os.path.join(POUT, "tables.tex"), "w") as f:
        for k, v in tabs.items(): f.write(f"%BEGIN-{k}\n{v}\n%END-{k}\n")
    json.dump(nums, open(os.path.join(POUT, "nums.json"), "w"), indent=1); print(json.dumps(nums))
