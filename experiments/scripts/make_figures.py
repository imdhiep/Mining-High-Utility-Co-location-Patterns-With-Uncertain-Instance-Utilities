"""Figures of Section VI from results/processed (claude/EXPERIMENT_PRESENTATION_RULES.md): one figure per varied
parameter, one subplot per dataset labelled "(a) New York", legend on top in one row outside the data, grid, no text
boxes in the data area, missing points left blank. Reads only the processed CSV files; writes PDF files.
Usage: python make_figures.py [--results DIR] [--out DIR]"""
import argparse, json, os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# categorical slots 1-3 of the validated palette, fixed per method, with a marker and a line style per method
METHODS = {"uhucpm": ("UHUCPM", "#2a78d6", "o", "-"),
           "cq": ("CQ-HUCPM", "#eb6834", "s", "--"),
           "pso": ("PSO-MUCP", "#1baf7a", "^", ":")}
SINGLE, DOUBLE = 3.5, 7.16          # IEEE column and page widths (in)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4,
                     "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42})


def name(dataset):
    return dataset.replace("_", " ")


def panels(count, width, height=1.75):
    fig, axes = plt.subplots(1, count, figsize=(width, height), squeeze=False)
    return fig, list(axes[0])


def finish(fig, axes, titles, handles, path, ncol=None):
    for i, (ax, t) in enumerate(zip(axes, titles)):
        ax.set_xlabel(f"{ax.get_xlabel()}\n({chr(97 + i)}) {t}")
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if (ncol or len(handles)) >= len(handles) else 0.8), w_pad=1.0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure:", path)


def method_handles(keys):
    return [Line2D([], [], color=METHODS[m][1], marker=METHODS[m][2], linestyle=METHODS[m][3], label=METHODS[m][0]) for m in keys]


def line(ax, x, y, m, err=None):
    _, color, marker, style = METHODS[m]
    if err is not None:
        ax.errorbar(x, y, yerr=err, color=color, marker=marker, linestyle=style, capsize=2, elinewidth=0.7)
    else:
        ax.plot(x, y, color=color, marker=marker, linestyle=style)


def accuracy_figure(acc, experiment, layouts, xcol, xlabel, path, width, metric="precision_at_k", K=100, log2=False):
    d = acc[(acc.experiment == experiment) & (acc.K == K) & acc.method.isin(METHODS)].copy()
    if d.empty:
        print("no data for", path); return
    d["x"] = d[xcol].map(lambda v: float(json.loads(v)) if isinstance(v, str) else float(v))
    fig, axes = panels(len(layouts), width)
    for ax, lay in zip(axes, layouts):
        for m in METHODS:
            s = d[(d.layout == lay) & (d.method == m)].dropna(subset=[f"{metric}_mean"]).sort_values("x")
            if not s.empty:
                line(ax, s.x, s[f"{metric}_mean"], m, s[f"{metric}_sd"])
        ax.set_xlabel(xlabel); ax.set_ylim(0, 1.02)
        if log2:
            ax.set_xscale("log", base=2); ax.set_xticks(sorted(d.x.unique())); ax.set_xticklabels([f"{v:g}" for v in sorted(d.x.unique())])
    axes[0].set_ylabel("Precision@K")
    finish(fig, axes, [name(l) for l in layouts], method_handles(METHODS), path)


def pr_figure(pr, path):
    """each method's ranking by its own score (B1_pr keeps the three rankings on the same draws)"""
    layouts = ["New_York", "Tokyo"]
    d = pr[(pr.experiment == "B1_pr") & (pr.K == 100)]
    if d.empty:
        print("no data for", path); return
    fig, axes = panels(len(layouts), SINGLE, 1.9)
    for ax, lay in zip(axes, layouts):
        for m in METHODS:
            s = d[(d.layout == lay) & (d.method == m)].sort_values("n")
            if not s.empty:
                ax.plot(s.recall_mean, s.precision_mean, color=METHODS[m][1], linestyle=METHODS[m][3])
        ax.set_xlabel("Recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Precision")
    handles = [Line2D([], [], color=METHODS[m][1], linestyle=METHODS[m][3], label=METHODS[m][0]) for m in METHODS]
    finish(fig, axes, [name(l) for l in layouts], handles, path)


def delta_figure(acc, path):
    layouts = ["Philadelphia", "New_York"]
    d = acc[(acc.experiment == "B3_threshold") & (acc.K == 100)]
    s_rule = d[d.method == "set_rule"]
    if s_rule.empty:
        print("no data for", path); return
    blue, orange = METHODS["uhucpm"][1], METHODS["cq"][1]
    fig, axes = panels(len(layouts), SINGLE, 1.9)
    for ax, lay in zip(axes, layouts):
        s = s_rule[s_rule.layout == lay].sort_values("delta")
        ax.errorbar(s.delta, s.precision_mean, yerr=s.precision_sd, color=blue, marker="o", linestyle="-", capsize=2, elinewidth=0.7)
        ax.errorbar(s.delta, s.recall_mean, yerr=s.recall_sd, color=blue, marker="o", markerfacecolor="white", linestyle="--",
                    capsize=2, elinewidth=0.7)
        ax.plot(s.delta, 1 - s.delta, color="#7f7f7f", linestyle=":", linewidth=1.0)
        cq = d[(d.layout == lay) & (d.method == "cq")]
        if not cq.empty:
            ax.axhline(cq.precision_mean.iloc[0], color=orange, linestyle="-", linewidth=1.0)
            ax.axhline(cq.recall_mean.iloc[0], color=orange, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Level δ"); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Precision, recall")
    handles = [Line2D([], [], color=blue, marker="o", linestyle="-", label="UHUCPM precision"),
               Line2D([], [], color=blue, marker="o", markerfacecolor="white", linestyle="--", label="UHUCPM recall"),
               Line2D([], [], color=orange, linestyle="-", label="CQ-HUCPM precision"),
               Line2D([], [], color=orange, linestyle="--", label="CQ-HUCPM recall"),
               Line2D([], [], color="#7f7f7f", linestyle=":", label="1 − δ")]
    finish(fig, axes, [name(l) for l in layouts], handles, path, ncol=3)


def timing_figure(tm, parts, xcol, xlabel, path, width):
    """parts: [(experiment, dataset)], one subplot each"""
    fig, axes = panels(len(parts), width)
    any_data = False
    for ax, (e, ds) in zip(axes, parts):
        d = tm[(tm.experiment == e) & (tm.dataset == ds)]
        for m in METHODS:
            s = d[d.program == m].sort_values(xcol)
            if not s.empty:
                line(ax, s[xcol], s.mean_seconds, m); any_data = True
        ax.set_xlabel(xlabel); ax.set_yscale("log")
        if xcol == "K" and not d.empty:        # K = 100, 300, 1000: log scale with the values as ticks
            ks = sorted(d.K.unique()); ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([str(int(v)) for v in ks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    if not any_data:
        plt.close(fig); print("no data for", path); return
    axes[0].set_ylabel("Running time (s)")
    finish(fig, axes, [name(ds) for _, ds in parts], method_handles(METHODS), path)


def hold_figure(curve, path, K=300):
    layouts = ["New_York", "Tokyo", "Tampa", "Philadelphia"]
    d = curve[curve.K == K]
    if d.empty:
        print("no data for", path); return
    fig, axes = panels(len(layouts), DOUBLE)
    for ax, city in zip(axes, layouts):
        for m in METHODS:
            s = d[(d.city == city) & (d.method == m) & (d.n <= K)].sort_values("n")
            if not s.empty:
                ax.plot(s.n, s.hold_rate, color=METHODS[m][1], linestyle=METHODS[m][3])
        ax.set_xlabel("Number of patterns n"); ax.set_xlim(0, K); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Hold rate")
    handles = [Line2D([], [], color=METHODS[m][1], linestyle=METHODS[m][3], label=METHODS[m][0]) for m in METHODS]
    finish(fig, axes, [name(c) for c in layouts], handles, path)


def case_figure(case, path, pair="ranking"):
    """participating instances of the two patterns of one pair of the case study (first: kept by UHUCPM, second: the
    other one); darker points have smaller posterior variance; the other venues of the city in light gray"""
    if pair not in case["pairs"]:
        print("no pair", pair, "for", path); return
    roles = case["pairs"][pair]
    fig, axes = panels(2, SINGLE, 2.0)
    vmax = max(max(case["patterns"][q]["variance"]) for q in roles)
    for ax, q in zip(axes, roles):
        v = case["patterns"][q]
        ax.scatter(np.array(case["all_x"]) / 1000, np.array(case["all_y"]) / 1000, s=0.2, color="#d9d9d9", linewidths=0, rasterized=True)
        shade = 1 - 0.75 * np.sqrt(np.array(v["variance"]) / vmax)
        ax.scatter(np.array(v["x"]) / 1000, np.array(v["y"]) / 1000, s=5, c=METHODS["uhucpm"][1], alpha=None, linewidths=0)
        ax.collections[-1].set_alpha(shade)
        cx, cy = np.mean(v["x"]) / 1000, np.mean(v["y"]) / 1000
        span = max(max(np.ptp(v["x"]), np.ptp(v["y"])) / 2000 + 0.5, 2.0)
        ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span); ax.set_aspect("equal")
        ax.set_xlabel("x (km)"); ax.tick_params(labelsize=6)
    axes[0].set_ylabel("y (km)")
    labels = [f"Kept pattern\n({len(case['patterns'][roles[0]]['instances'])} venues)",
              f"Rejected pattern\n({len(case['patterns'][roles[1]]['instances'])} venues)"]
    handles = [Line2D([], [], marker="o", color="w", markerfacecolor=METHODS["uhucpm"][1], label="Participating venue"),
               Line2D([], [], marker="o", color="w", markerfacecolor="#d9d9d9", label="Other venue")]
    finish(fig, axes, labels, handles, path)


def read(proc, f):
    p = os.path.join(proc, f)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(HERE)
    ap.add_argument("--results", default=os.path.join(root, "results")); ap.add_argument("--out", default=os.path.join(root, "figures"))
    ap.add_argument("--case-pair", default="ranking", help="pair of the case study shown in fig_case.pdf (see run_case.py)")
    a = ap.parse_args()
    proc = os.path.join(a.results, "processed"); os.makedirs(a.out, exist_ok=True)
    acc, pr, tm, curve = (read(proc, f) for f in ("accuracy.csv", "pr_curve.csv", "timing.csv", "hold_curve.csv"))
    out = lambda f: os.path.join(a.out, f)
    if not acc.empty:
        accuracy_figure(acc, "B1_evidence", ["Philadelphia", "Tampa", "New_York", "Tokyo"], "value", "Evidence factor c",
                        out("fig_evidence.pdf"), DOUBLE, log2=True)
        accuracy_figure(acc, "B2_overdispersion", ["Tampa", "New_York"], "value", "Overdispersion φ", out("fig_overdispersion.pdf"), SINGLE)
        k = acc[acc.experiment == "B3_threshold"]
        if not k.empty:
            accuracy_figure_k(k, out("fig_threshold.pdf"))
        delta_figure(acc, out("fig_level.pdf"))
    if not pr.empty:
        pr_figure(pr, out("fig_precision_recall.pdf"))
    if not tm.empty:
        timing_figure(tm, [("C1_features", "Synthetic"), ("C1_features", "New_York")], "features", "Number of features",
                      out("fig_time_features.pdf"), SINGLE)
        timing_figure(tm, [("C2_instances_synthetic", "Synthetic"), ("C2_instances_new_york", "New_York")], "n",
                      "Number of instances", out("fig_time_instances.pdf"), SINGLE)
        timing_figure(tm, [("C3_threshold", c) for c in ("New_Orleans", "Tampa", "New_York")], "K", "Threshold K",
                      out("fig_time_threshold.pdf"), DOUBLE)
    if not curve.empty:
        hold_figure(curve, out("fig_hold.pdf"))
    if os.path.exists(os.path.join(proc, "case.json")):
        case_figure(json.load(open(os.path.join(proc, "case.json"))), out("fig_case.pdf"), a.case_pair)


def accuracy_figure_k(k, path):
    """precision@K against K (B3_threshold keeps every K of one run in one row group)"""
    layouts = ["Philadelphia", "New_York"]
    fig, axes = panels(len(layouts), SINGLE)
    for ax, lay in zip(axes, layouts):
        for m in METHODS:
            s = k[(k.layout == lay) & (k.method == m)].dropna(subset=["precision_at_k_mean"]).sort_values("K")
            if not s.empty:
                line(ax, s.K, s.precision_at_k_mean, m, s.precision_at_k_sd)
        ax.set_xlabel("Threshold K"); ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ks = sorted(k.K.unique()); ax.set_xticks(ks); ax.set_xticklabels([str(v) for v in ks]); ax.minorticks_off()
    axes[0].set_ylabel("Precision@K")
    finish(fig, axes, [name(l) for l in layouts], method_handles(METHODS), path)


if __name__ == "__main__":
    main()
