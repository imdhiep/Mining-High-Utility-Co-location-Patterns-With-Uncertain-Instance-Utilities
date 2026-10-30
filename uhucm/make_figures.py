"""Figures for Section VI, generated only from results_db (no hand-entered numbers). Output: PDF files."""
import os, json, math, csv
import numpy as np
import figstyle as fs
from figstyle import plt
import results_db

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tkde_overleaf", "figures")
os.makedirs(OUT, exist_ok=True)
db, syn, d8 = results_db.load()
LAB = {"Phi": "D1", "New": "D2", "Tam": "D3", "FSQ_NYC": "D4", "FSQ_TKY": "D5", "Las": "D6a", "Pho": "D6b", "Tor": "D6c"}
ORDER = ["Phi", "New", "Tam", "FSQ_NYC", "FSQ_TKY"]
W1, W2 = 3.45, 7.1      # IEEE column / page width in inches


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# ---------------------------------------------------------------- Fig: answer-set change (E0)
def fig_e0():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.0))
    zs = [0.0, 1.645, 2.45, 4.36]
    for ax, K, lab in zip(axes, (100, 1000), ("(a) $K=100$", "(b) $K=1000$")):
        for i, nm in enumerate(ORDER):
            q = db[nm]["quality"][K]; R0 = q["R0"]
            ys = [1.0] + [q[f"R{z}"] / R0 for z in zs[1:]]
            ax.plot(zs, ys, marker=fs.MARKERS[i % 6], color=fs.PALETTE[i], linestyle="-" if i < 5 else "--", label=LAB[nm])
        ax.set_xticks(zs); ax.set_xticklabels(["0", "1.645", "2.45", "4.36"]); ax.set_ylim(0.3, 1.02)
        ax.set_xlabel("$z$"); ax.set_ylabel(r"$|\mathcal{R}_z|/|\mathcal{R}_0|$ (= Jaccard)")
    h, l = axes[0].get_legend_handles_labels(); fig.subplots_adjust(wspace=0.28)
    fs.finalize(fig, axes, ["(a) $K=100$", "(b) $K=1000$"], h, l, ncol=8, bottom=0.42, leg_y=0.17); save(fig, "fig_e0.pdf")


# ---------------------------------------------------------------- Fig: hold-out (E7)
def fig_e7():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1))
    for ax, K, lab in zip(axes, (300, 1000), ("(a) $K=300$", "(b) $K=1000$")):
        x = np.arange(len(ORDER)); w = 0.27
        kept = [db[n]["quality"][K]["hold_kept"] for n in ORDER]
        rej = [db[n]["quality"][K]["hold_rej"] for n in ORDER]
        mat = [db[n]["quality"][K]["hold_matched"] for n in ORDER]
        ax.bar(x - w, kept, w, color=fs.BLUE, label="Kept by UHUCM ($z$=1.645)")
        ax.bar(x, mat, w, color=fs.GRAY, label="Kept, matched on training score")
        ax.bar(x + w, rej, w, color=fs.ORANGE, label="Accepted by $z$=0, rejected by UHUCM")
        ax.set_xticks(x); ax.set_xticklabels([LAB[n] for n in ORDER]); ax.set_ylim(0, 1.0); ax.set_ylabel("Hold rate in test period")
    h, l = axes[0].get_legend_handles_labels(); fig.subplots_adjust(wspace=0.22)
    fs.finalize(fig, axes, ["(a) $K=300$", "(b) $K=1000$"], h, l, ncol=3, bottom=0.38, leg_y=0.17); save(fig, "fig_e7.pdf")


# ---------------------------------------------------------------- Fig: FE miss rate by |A(P)| (E3)
def fig_e3():
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    bins = ["1-10", "11-50", "51-200", ">200"]; x = np.arange(4); names = ["Phi", "New", "Tam", "FSQ_NYC", "Las", "Tor"]
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        b = db[nm]["quality"][1000]["FE_bins"]
        y = [(m / t if t else np.nan) for m, t in b]
        ax.bar(x - 0.4 + w * (i + 0.5), y, w, color=fs.PALETTE[i % 6], label=LAB[nm])
    ax.set_xticks(x); ax.set_xticklabels(bins); ax.set_xlabel(r"Number of atoms $|A(P)|$"); ax.set_ylabel("FE miss rate"); ax.set_ylim(0, 1.05)
    h, l = ax.get_legend_handles_labels(); fig.subplots_adjust(bottom=0.30)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.08), ncol=6); save(fig, "fig_e3.pdf")


# ---------------------------------------------------------------- Fig: calibration (E2)
def fig_e2():
    e2 = syn["e2"]["results"]; deltas = [0.01, 0.05, 0.1, 0.2]
    knobs = ["correct", "t3", "prior+0.3", "prior-0.3", "sigma x0.7", "sigma x1.3", "spatial"]
    nice = {"correct": "Correct model", "t3": "$t_3$ noise", "prior+0.3": "Prior +0.3", "prior-0.3": "Prior $-$0.3",
            "sigma x0.7": r"$\sigma_e\times0.7$", "sigma x1.3": r"$\sigma_e\times1.3$", "spatial": "Spatial 0.5"}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.0))
    for i, kn in enumerate(knobs):
        f = [e2[kn][f"normal d={d}"]["FDP"] for d in deltas]; p = [e2[kn][f"normal d={d}"]["power"] for d in deltas]
        axes[0].plot(deltas, f, marker=fs.MARKERS[i % 6], color=fs.PALETTE[i], label=nice[kn], linestyle="-" if i < 6 else "--")
        axes[1].plot(deltas, p, marker=fs.MARKERS[i % 6], color=fs.PALETTE[i], label=nice[kn], linestyle="-" if i < 6 else "--")
    axes[0].plot(deltas, deltas, color="black", linewidth=0.7, linestyle=":", label=r"FDP $=\delta$")
    axes[0].set_xlabel(r"$\delta$ (normal choice $z=\Phi^{-1}(1-\delta)$)"); axes[0].set_ylabel("Mean FDP")
    axes[1].set_xlabel(r"$\delta$"); axes[1].set_ylabel("Detection rate")
    for ax in axes: ax.set_xticks(deltas)
    h, l = axes[0].get_legend_handles_labels(); fig.subplots_adjust(wspace=0.25)
    fs.finalize(fig, axes, ["(a) False discovery proportion", "(b) Detection rate"], h, l, ncol=8, bottom=0.44, leg_y=0.19); save(fig, "fig_e2.pdf")


# ---------------------------------------------------------------- Fig: efficiency vs threshold (D1 and D5), log2 bars + table
def fig_eff_mu():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    for ax, nm, lab in zip(axes, ("Phi", "FSQ_TKY"), ("(a) D1 (Yelp, Philadelphia)", "(b) D5 (Foursquare, Tokyo)")):
        em = db[nm]["eff_mu"]; Ks = sorted(em)
        series = [(lbl, [float(em[K][k]["time"]) for K in Ks]) for k, lbl in (("exact", "UHUCM"), ("mean", "Mean bound"), ("noprune", "NoPrune"))]
        fs.bar_with_table(ax, [f"K={K}" for K in Ks], series, log2=True, ylabel="Run time (s)", fmt="{:.2f}")
        ax.set_title(lab, y=-0.62, fontsize=8)
    fig.subplots_adjust(wspace=0.45, bottom=0.30); save(fig, "fig_eff_mu.pdf")


def fig_eff_nodes():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    for ax, nm, lab in zip(axes, ("Phi", "FSQ_TKY"), ("(a) D1 (Yelp, Philadelphia)", "(b) D5 (Foursquare, Tokyo)")):
        em = db[nm]["eff_mu"]; Ks = sorted(em)
        series = [(lbl, [float(em[K][k]["evaluated"]) for K in Ks]) for k, lbl in (("exact", "UHUCM"), ("mean", "Mean bound"), ("noprune", "NoPrune"))]
        fs.bar_with_table(ax, [f"K={K}" for K in Ks], series, log2=True, ylabel="Evaluated nodes", fmt="{:.0f}")
        ax.set_title(lab, y=-0.62, fontsize=8)
    fig.subplots_adjust(wspace=0.45, bottom=0.30); save(fig, "fig_eff_nodes.pdf")


# ---------------------------------------------------------------- Fig: number of features / instances (synthetic)
def fig_scale():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
    sk = syn["scale_k"]; sn = syn["scale_n"]
    for ax, S, key, lab in ((axes[0], sk, "k", "(a) Number of features ($n=10^4$)"), (axes[1], sn, "n", "(b) Number of instances ($k=10$)")):
        cats = [(f"{r[key] // 1000}k" if key == "n" else str(r[key])) for r in S]
        series = [("UHUCM", [r["uhucm"]["time"] for r in S]), ("NoPrune", [r["noprune"]["time"] for r in S]),
                  ("Build (nbr.+cliques)", [r["build_s"] for r in S])]
        fs.bar_with_table(ax, cats, series, log2=True, ylabel="Run time (s)", fmt="{:.2f}")
        ax.set_title(lab, y=-0.62, fontsize=8)
    fig.subplots_adjust(wspace=0.45, bottom=0.30); save(fig, "fig_scale.pdf")


# ---------------------------------------------------------------- Fig: distance threshold (lines with value labels, as Fig. 10 of the template)
def fig_eff_d():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1))
    for ax, nm, lab in zip(axes, ("Phi", "FSQ_TKY"), ("(a) D1", "(b) D5")):
        ed = db[nm]["eff_d"]; ds = sorted(ed)
        fs.line(ax, ds, [("UHUCM", [ed[d]["exact"]["time"] for d in ds]), ("NoPrune", [ed[d]["noprune"]["time"] for d in ds])],
                fmt="{:.2f}", xlabel="Distance threshold $d$ (m)", ylabel="Run time (s)", logy=True)
        ax2 = ax.twinx(); ax2.plot(ds, [ed[d]["cliques"] for d in ds], color=fs.GRAY, linestyle=":", marker="^", label="Maximal cliques $C$")
        ax2.set_yscale("log"); ax2.set_ylabel("$C$"); ax2.grid(False)
        if nm == "Phi":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.subplots_adjust(wspace=0.45)
    fs.finalize(fig, axes, ["(a) D1", "(b) D5"], h1 + h2, l1 + l2, ncol=3, bottom=0.38, leg_y=0.16); save(fig, "fig_eff_d.pdf")


# ---------------------------------------------------------------- Fig: case study map (Foursquare NYC; local data)
def fig_case():
    import fsq_loader
    from fastmine import Problem, mine, pattern_atoms
    import paths
    rows, users, NU, info = fsq_loader.load(paths.fsq("NYC"), "NYC")
    names = sorted({r["feat"] for r in rows}); fi = {f: j for j, f in enumerate(names)}
    F = [fi[r["feat"]] for r in rows]; XY = np.array([(r["x"], r["y"]) for r in rows])
    mu = np.array([r["mu"] for r in rows]); s2 = np.array([r["s2"] for r in rows])
    P0 = Problem(F, XY, mu, s2, 50.0, 0.0, feat_names=names); P1 = Problem(F, XY, mu, s2, 50.0, 2.45, feat_names=names)   # calibrated choice for the activity posterior
    muK = db["FSQ_NYC"]["quality"][300]["mu"]
    R0 = mine(P0, muK)[0]; R1 = mine(P1, muK)[0]
    rej = sorted(set(R0) - set(R1), key=lambda p: -R0[p]); kept = sorted(set(R1), key=lambda p: -R1[p])
    pr, pk = rej[0], kept[0]
    Ar = pattern_atoms(P0, pr); Ak = pattern_atoms(P0, pk)
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.98)); caps = []
    for ax, A, p, lab, sc in ((axes[0], Ak, pk, "(a) Kept", R1[pk]), (axes[1], Ar, pr, "(b) Rejected", R0[pr])):
        ax.scatter(XY[:, 0] / 1000, XY[:, 1] / 1000, s=0.3, color="#D9D9D9", linewidths=0, rasterized=True)
        a = A[A < P0.n]; sig = np.sqrt(s2[a]); col = np.where(mu[a] >= 0, fs.BLUE, fs.ORANGE)
        alpha = np.clip(1 - sig / (sig.max() + 1e-12), 0.25, 1)
        for i, j in enumerate(a):
            ax.scatter(XY[j, 0] / 1000, XY[j, 1] / 1000, s=3, color=col[i], alpha=float(alpha[i]), linewidths=0)
        cx, cy = XY[a, 0].mean() / 1000, XY[a, 1].mean() / 1000; span = max(np.ptp(XY[a, 0]), np.ptp(XY[a, 1])) / 1000 / 2 + 1.0
        ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span); ax.set_aspect("equal")
        ax.set_xlabel("x (km, UTM 18N)", fontsize=7, labelpad=1); ax.set_ylabel("y (km)", fontsize=7, labelpad=1); ax.tick_params(labelsize=6); ax.grid(False)
        lbl = " + ".join(s.replace("___", " / ").replace("_and_", " & ").replace("_", " ") for s in p)
        caps.append(f"{lab}: {{{lbl}}}")
        ax.text(0.02, 0.98, f"$|A(P)|$={len(A)}, median $c$={np.median(np.array([rows[j]['a'] for j in a])):.0f}", transform=ax.transAxes, va="top", fontsize=5.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", color="w", markerfacecolor=fs.BLUE, label=r"$\mu_a\geq 0$"), Line2D([], [], marker="o", color="w", markerfacecolor=fs.ORANGE, label=r"$\mu_a<0$"),
         Line2D([], [], marker="o", color="w", markerfacecolor="#D9D9D9", label="Other venues")]
    fig.subplots_adjust(wspace=0.38, left=0.12, right=0.99, top=0.99)
    fs.finalize(fig, axes, caps, h, [x.get_label() for x in h], ncol=3, bottom=0.345, leg_y=0.175, lab_y=0.01, fontsize=6.5); save(fig, "fig_case.pdf")
    json.dump(dict(kept=list(pk), kept_score=R1[pk], kept_atoms=int(len(Ak)), rejected=list(pr), rejected_s0=R0[pr], rejected_lb=float(P1.score(Ar)), rejected_atoms=int(len(Ar)), muK=muK),
              open(os.path.join(OUT, "fig_case.json"), "w"), indent=1)


# ================================================================ single-column figures for the 14-page version
def _leg(fig, ax, ncol, y=-0.02):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.6, columnspacing=0.9)


def fig_e0_1c():
    fig, ax = plt.subplots(figsize=(W1, 1.75))
    zs = [0.0, 1.645, 2.45, 4.36]
    for i, nm in enumerate(ORDER):
        q = db[nm]["quality"][300]; R0 = q["R0"]
        ax.plot(zs, [1.0] + [q[f"R{z}"] / R0 for z in zs[1:]], marker=fs.MARKERS[i], color=fs.PALETTE[i], label=LAB[nm])
    ax.set_xticks(zs); ax.set_xticklabels(["0", "1.645", "2.45", "4.36"]); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("$z$"); ax.set_ylabel(r"$|\mathcal{R}_z|/|\mathcal{R}_0|$")
    fig.subplots_adjust(bottom=0.30); _leg(fig, ax, 5, 0.02); save(fig, "fig_e0.pdf")


def fig_e2_1c():
    e2 = syn["e2"]["results"]; deltas = [0.01, 0.05, 0.1, 0.2]
    knobs = ["correct", "estimated", "t3", "prior+0.3", "prior-0.3", "sigma x0.7", "spatial"]
    nice = {"correct": "Correct", "estimated": "Estimated", "t3": "$t_3$ noise", "prior+0.3": r"Prior $+0.3\tau$", "prior-0.3": r"Prior $-0.3\tau$",
            "sigma x0.7": r"$\sigma_e\times0.7$", "spatial": "Spatial 0.5"}
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for i, kn in enumerate(knobs):
        ax.plot(deltas, [e2[kn][f"normal d={d}"]["FDP"] for d in deltas], marker=fs.MARKERS[i % 6], color=fs.PALETTE[i], label=nice[kn],
                linestyle="-" if i < 6 else "--")
    ax.plot(deltas, deltas, color="black", linewidth=0.7, linestyle=":", label=r"FDP $=\delta$")
    ax.set_xticks(deltas); ax.set_xlabel(r"$\delta$ ($z=\Phi^{-1}(1-\delta)$)"); ax.set_ylabel("Mean FDP")
    fig.subplots_adjust(bottom=0.36); _leg(fig, ax, 4, 0.02); save(fig, "fig_e2.pdf")


def fig_eff_1c():
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.85))
    meth = (("exact", "UHUCM", fs.BLUE, "o"), ("mean", "Mean bound", fs.ORANGE, "s"), ("noprune", "NoPrune", fs.GRAY, "^"))
    for nm, ls in (("Phi", "-"), ("FSQ_TKY", "--")):
        em = db[nm]["eff_mu"]; Ks = sorted(em)
        for k, lbl, c, m in meth:
            lab = lbl
            axes[0].plot(Ks, [em[K][k]["evaluated"] for K in Ks], color=c, marker=m, linestyle=ls, label=lab)
            axes[1].plot(Ks, [em[K][k]["time"] for K in Ks], color=c, marker=m, linestyle=ls, label=lab)
    for ax, yl, t in ((axes[0], "Evaluated nodes", "(a) Nodes"), (axes[1], "Mining time (s)", "(b) Time")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$K$"); ax.set_ylabel(yl, labelpad=1)
        ax.set_xticks([30, 100, 300, 1000, 3000]); ax.set_xticklabels(["30", "100", "300", "1k", "3k"])
        ax.text(0.5, -0.42, t, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    axes[1].yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4])); axes[1].yaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4"]))
    axes[1].yaxis.set_minor_locator(NullLocator())
    h = [Line2D([], [], color=c, marker=m, label=l) for _, l, c, m in meth] + \
        [Line2D([], [], color="black", linestyle="-", label="D1"), Line2D([], [], color="black", linestyle="--", label="D5")]
    fig.subplots_adjust(wspace=0.42, bottom=0.42)
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=5, handlelength=1.6, columnspacing=0.8)
    save(fig, "fig_eff.pdf")


def fig_scale_1c():
    SM = [r for r in syn["scale_mem"] if r["kind"] == "syn"]
    def med(r, key): v = sorted(x[key] for x in r["reps"]); return v[1], v[0], v[2]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.85))
    for ax, key, fixed, t in ((axes[0], "k", lambda r: r["n"] == 10000, "(a) $k$ ($n=10^4$)"), (axes[1], "n", lambda r: r["k"] == 10, "(b) $n$ ($k=10$)")):
        for meth, lbl, c, m in (("uhucm", "UHUCM", fs.BLUE, "o"), ("noprune", "NoPrune", fs.GRAY, "^")):
            rs = sorted([r for r in SM if r["method"] == meth and fixed(r)], key=lambda r: r[key])
            xs = [r[key] for r in rs]; ms = [med(r, "mine_s") for r in rs]
            ax.errorbar(xs, [a for a, _, _ in ms], yerr=[[a - lo for a, lo, _ in ms], [hi - a for a, _, hi in ms]], color=c, marker=m, label=lbl, capsize=1.5, elinewidth=0.6)
        rs = sorted([r for r in SM if r["method"] == "uhucm" and fixed(r)], key=lambda r: r[key])
        ax.plot([r[key] for r in rs], [med(r, "build_s")[0] for r in rs], color=fs.ORANGE, marker="s", linestyle="--", label="Build (nbr.+cliques)")
        ax.set_yscale("log"); ax.set_ylabel("Time (s)", labelpad=1)
        if key == "n": ax.set_xscale("log")
        ax.text(0.5, -0.42, t, transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.subplots_adjust(wspace=0.42, bottom=0.42); _leg(fig, axes[0], 3, 0.05); save(fig, "fig_scale.pdf")


# ================================================================ two-column figures (14-page version)
def _sub(ax, t, y=-0.36):
    ax.text(0.5, y, t, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def fig_e2_2c():
    e2 = syn["e2"]["results"]; deltas = [0.01, 0.05, 0.1, 0.2]
    knobs = ["correct", "estimated", "t3", "prior+0.3", "prior-0.3", "sigma x0.7", "sigma x1.3", "spatial"]
    nice = {"correct": "Correct model", "estimated": "Re-estimated", "t3": "$t_3$ noise", "prior+0.3": r"Prior $+0.3\tau$", "prior-0.3": r"Prior $-0.3\tau$",
            "sigma x0.7": r"$\sigma_e\times0.7$", "sigma x1.3": r"$\sigma_e\times1.3$", "spatial": "Spatial 0.5"}
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.45))
    for i, kn in enumerate(knobs):
        st = dict(marker=fs.MARKERS[i % 6], color=fs.PALETTE[i], label=nice[kn], linestyle="-" if i < 6 else "--")
        axes[0].plot(deltas, [e2[kn][f"normal d={d}"]["FDP"] for d in deltas], **st)
        axes[1].plot(deltas, [e2[kn][f"normal d={d}"]["power"] for d in deltas], **st)
    axes[0].plot(deltas, deltas, color="black", linewidth=0.7, linestyle=":", label=r"FDP $=\delta$")
    axes[0].set_ylabel("Mean FDP"); axes[1].set_ylabel("Detection rate")
    for ax in axes: ax.set_xticks(deltas); ax.set_xlabel(r"$\delta$ ($z=\Phi^{-1}(1-\delta)$)")
    for ax in axes: ax.set_xlabel(r"$\delta$"); ax.set_xticklabels(["0.01", "0.05", "0.1", "0.2"], fontsize=6.5)
    _sub(axes[0], "(a) Mean FDP", -0.31); _sub(axes[1], "(b) Detection rate", -0.31)
    fig.subplots_adjust(wspace=0.42, left=0.13, right=0.98, top=0.97, bottom=0.50); _leg(fig, axes[0], 3, 0.31); save(fig, "fig_e2.pdf")


def fig_eff_2c():
    from matplotlib.lines import Line2D
    meth = (("exact", "UHUCM", fs.BLUE, "o"), ("mean", "Mean bound", fs.ORANGE, "s"), ("noprune", "NoPrune", fs.GRAY, "^"))
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.9))
    caps = ["(a) D1, nodes", "(b) D1, time", "(c) D5, nodes", "(d) D5, time"]
    for j, nm in enumerate(("Phi", "FSQ_TKY")):
        em = db[nm]["eff_mu"]; Ks = sorted(em)
        for k, lbl, c, m in meth:
            axes[2 * j].plot(Ks, [em[K][k]["evaluated"] for K in Ks], color=c, marker=m, label=lbl)
            axes[2 * j + 1].plot(Ks, [em[K][k]["time"] for K in Ks], color=c, marker=m, label=lbl)
    for i, ax in enumerate(axes):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$K$")
        ax.set_xticks([30, 100, 300, 1000, 3000]); ax.set_xticklabels(["30", "100", "300", "1k", "3k"]); ax.minorticks_off()
        ax.set_ylabel("Evaluated nodes" if i % 2 == 0 else "Mining time (s)", labelpad=1); _sub(ax, caps[i], -0.36)
        if i % 2 == 1:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            lo, hi = ax.get_ylim(); tk = [t for t in (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8) if lo <= t <= hi]; ax.yaxis.set_major_locator(FixedLocator(tk)); ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in tk]))
    fig.subplots_adjust(wspace=0.42, bottom=0.36); _leg(fig, axes[0], 3, 0.03); save(fig, "fig_effw.pdf")


def fig_eff_v2():
    """Mining time (mean and SD of 10 runs, binary-search suffix bound) of UHUCM, the mean bound and NoPrune:
    (a), (b) against the rank threshold K on D1 and D5 (k = 20); (c), (d) against the number of features k at K = 300."""
    T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_synth", "stats_timing2.json")))
    meth = (("exact", "UHUCM", fs.BLUE, "o"), ("mean", "Mean bound", fs.ORANGE, "s"), ("noprune", "NoPrune", fs.GRAY, "^"))
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.9))
    def vsK(ax, key):
        R = T[key]["rows"]; Ks = sorted(R, key=int)
        for k, lbl, c, m in meth:
            ax.errorbar([int(K) for K in Ks], [R[K]["table"][k][0] for K in Ks], yerr=[R[K]["table"][k][1] for K in Ks], color=c, marker=m, label=lbl, capsize=1.5, elinewidth=0.6)
        ax.set_xscale("log"); ax.set_xlabel("$K$"); ax.set_xticks([int(K) for K in Ks]); ax.set_xticklabels([("1k" if int(K) == 1000 else "3k" if int(K) == 3000 else K) for K in Ks]); ax.minorticks_off()
    def vsk(ax, city):
        ks = sorted(int(x.split("|")[2]) for x in T if x.startswith(city + "|50|"))
        for k, lbl, c, m in meth:
            ys = [T[f"{city}|50|{kk}"]["rows"]["300"]["table"][k] for kk in ks]
            ax.errorbar(ks, [y[0] for y in ys], yerr=[y[1] for y in ys], color=c, marker=m, label=lbl, capsize=1.5, elinewidth=0.6)
        ax.set_xlabel("Number of features $k$"); ax.set_xticks(ks)
    first = "D1" if "D1|50|20" in T else "D4"
    vsK(axes[0], f"{first}|50|20"); vsK(axes[1], "D5|50|20"); vsk(axes[2], "D4"); vsk(axes[3], "D5")
    caps = [f"(a) {first}, $k=20$", "(b) D5, $k=20$", "(c) D4, $K=300$", "(d) D5, $K=300$"]
    for ax, cp in zip(axes, caps):
        ax.set_yscale("log"); ax.set_ylabel("Mining time (s)", labelpad=1); _sub(ax, cp, -0.36)
    fig.subplots_adjust(wspace=0.45, bottom=0.36); _leg(fig, axes[0], 3, 0.03); save(fig, "fig_effw.pdf")


def fig_scale_2c():
    SM = [r for r in syn["scale_mem"] if r["kind"] == "syn"]
    def med(r, key): v = sorted(x[key] for x in r["reps"]); return v[1], v[0], v[2]
    fig, axes = plt.subplots(1, 4, figsize=(W2, 1.9))
    specs = ((axes[0], "k", lambda r: r["n"] == 10000, "mine_s", "Time (s)", "(a) Time vs. $k$ ($n=10^4$)"),
             (axes[1], "k", lambda r: r["n"] == 10000, "evaluated", "Evaluated nodes", "(b) Nodes vs. $k$"),
             (axes[2], "n", lambda r: r["k"] == 10, "mine_s", "Time (s)", "(c) Time vs. $n$ ($k=10$)"),
             (axes[3], "n", lambda r: r["k"] == 10, "evaluated", "Evaluated nodes", "(d) Nodes vs. $n$"))
    for ax, key, fixed, val, yl, cap in specs:
        for meth, lbl, c, m in (("uhucm", "UHUCM", fs.BLUE, "o"), ("noprune", "NoPrune", fs.GRAY, "^")):
            rs = sorted([r for r in SM if r["method"] == meth and fixed(r)], key=lambda r: r[key])
            ms = [med(r, val) for r in rs]
            ax.errorbar([r[key] for r in rs], [a for a, _, _ in ms], yerr=[[a - lo for a, lo, _ in ms], [hi - a for a, _, hi in ms]],
                        color=c, marker=m, label=lbl, capsize=1.5, elinewidth=0.6)
        if val == "mine_s":
            rs = sorted([r for r in SM if r["method"] == "uhucm" and fixed(r)], key=lambda r: r[key])
            ax.plot([r[key] for r in rs], [med(r, "build_s")[0] for r in rs], color=fs.ORANGE, marker="s", linestyle="--", label="Build (neighbors + cliques)")
        ax.set_yscale("log"); ax.set_ylabel(yl, labelpad=1); ax.set_xlabel(f"${key}$")
        if key == "n": ax.set_xscale("log")
        ax.minorticks_off(); _sub(ax, cap, -0.36)
    fig.subplots_adjust(wspace=0.45, bottom=0.36); _leg(fig, axes[0], 3, 0.03); save(fig, "fig_scalew.pdf")


def fig_e7_2c():
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.9))
    for ax, K, lab in zip(axes, (300, 1000), ("(a) $K=300$", "(b) $K=1000$")):
        x = np.arange(len(ORDER)); w = 0.27
        ax.bar(x - w, [db[n]["quality"][K]["hold_kept"] for n in ORDER], w, color=fs.BLUE, label="Kept by UHUCM ($z$=1.645)")
        ax.bar(x, [db[n]["quality"][K]["hold_matched"] for n in ORDER], w, color=fs.GRAY, label="Kept, matched on training score")
        ax.bar(x + w, [db[n]["quality"][K]["hold_rej"] for n in ORDER], w, color=fs.ORANGE, label="Accepted by $z$=0, rejected by UHUCM")
        ax.set_xticks(x); ax.set_xticklabels([LAB[n] for n in ORDER]); ax.set_ylim(0, 1.0); ax.set_ylabel("Hold rate"); _sub(ax, lab, -0.22)
    fig.subplots_adjust(wspace=0.18, bottom=0.30); _leg(fig, axes[0], 3, 0.03); save(fig, "fig_e7w.pdf")


def fig_e3_1c():
    fig, ax = plt.subplots(figsize=(W1, 1.7))
    bins = ["1-10", "11-50", "51-200", ">200"]; x = np.arange(4); names = ORDER; w = 0.8 / len(names)
    for i, nm in enumerate(names):
        b = db[nm]["quality"][1000]["FE_bins"]
        ax.bar(x - 0.4 + w * (i + 0.5), [(m / t if t else np.nan) for m, t in b], w, color=fs.PALETTE[i], label=LAB[nm])
    ax.set_xticks(x); ax.set_xticklabels(bins); ax.set_xlabel(r"Number of atoms $|A(P)|$"); ax.set_ylabel("FE miss rate"); ax.set_ylim(0, 1.05)
    fig.subplots_adjust(bottom=0.36); _leg(fig, ax, 5, 0.02); save(fig, "fig_e3.pdf")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["e0_1c", "e2_2c", "eff_2c", "scale_2c", "e7_2c", "e3_1c"]
    for t in todo: globals()["fig_" + t]()
