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
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# colors and markers of the sample figures chosen by the author (red, blue, green for our method), one marker per method
METHODS = {"cq": ("CQ-HUCPM", "#d62f2f", "o", "-"),
           "pso": ("PSO-MUCP", "#2c62d4", "s", "-"),
           "uhucpm": ("UHUCPM", "#1f9a55", "D", "-")}
SINGLE, DOUBLE = 3.5, 7.16          # IEEE column and page widths (in)
# three values of every varied parameter (the default in the middle where there is one); other values stay in the results
LEVELS = {"c": [0.5, 1, 4], "K": [100, 300, 1000], "features": [20, 40, 80], "instances": [40000, 80000, 120000],
          "share": [0.25, 0.5, 1.0], "d_california": [1000, 1500, 2000], "n": [100, 200, 300]}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 1.2,
                     "lines.markersize": 3.5, "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#b0b0b0", "axes.linewidth": 0.6, "xtick.color": "#404040", "ytick.color": "#404040",
                     "axes.axisbelow": True, "pdf.fonttype": 42})
HEADROOM = 1.55                    # axis range stretched by this factor above the data, for the legend inside each panel


def name(dataset):
    return dataset.replace("_", " ")


def panels(count, width, height=2.0):
    fig, axes = plt.subplots(1, count, figsize=(width, height), squeeze=False)
    return fig, list(axes[0])


def headroom(ax):
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        ax.set_ylim(lo, 10 ** (np.log10(lo) + (np.log10(hi) - np.log10(lo)) * HEADROOM))
    else:
        ax.set_ylim(lo, lo + (hi - lo) * HEADROOM)
        if hi <= 1.05:                         # shares: ticks only up to 1
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def finish(fig, axes, titles, handles, path, ncol=None, inside=True):
    """legend inside every panel at the top (two columns, light frame); panel label "(a) Dataset" in serif below the
    axis label, as in the sample figures"""
    for i, (ax, t) in enumerate(zip(axes, titles)):
        ax.set_xlabel(ax.get_xlabel(), labelpad=1.5)
        ax.annotate(f"({chr(97 + i)}) {t}", xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -27), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, family="serif")
        if inside:
            headroom(ax)
            narrow = fig.get_figwidth() / len(axes) < 2.0     # panels of a one-column figure: a smaller legend
            ax.legend(handles=handles, loc="upper center", ncol=ncol or 2, frameon=True, framealpha=0.9, edgecolor="#d0d0d0",
                      handlelength=1.3 if narrow else 2.0, columnspacing=0.6 if narrow else 0.9, borderpad=0.3,
                      labelspacing=0.2, handletextpad=0.3 if narrow else 0.4, fontsize=5.6 if narrow else 6.5)
    if not inside:
        fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 if inside else 0.88), w_pad=1.2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure:", path)


def method_handles(keys):
    return [Line2D([], [], color=METHODS[m][1], marker=METHODS[m][2], linestyle=METHODS[m][3], label=METHODS[m][0]) for m in keys]


def bar_handles(keys):
    return [Patch(facecolor=METHODS[m][1], edgecolor="none", label=METHODS[m][0]) for m in keys]


def bars(ax, cats, values, errors=None):
    """grouped columns: one group per category, one column per method, 2 px white gap between neighbouring columns"""
    n = len(METHODS); w = 0.8 / n; x = np.arange(len(cats))
    for i, m in enumerate(METHODS):
        y = values.get(m)
        if y is None:
            continue
        e = None if errors is None else errors.get(m)
        ax.bar(x + (i - (n - 1) / 2) * w, y, w * 0.9, color=METHODS[m][1], edgecolor="none",
               yerr=e, error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor="#404040"))
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.grid(axis="x", visible=False)


def line(ax, x, y, m, err=None):
    _, color, marker, style = METHODS[m]
    if err is not None:
        ax.errorbar(x, y, yerr=err, color=color, marker=marker, linestyle=style, capsize=2, elinewidth=0.7)
    else:
        ax.plot(x, y, color=color, marker=marker, linestyle=style)


def accuracy_figure(acc, experiment, layouts, xcol, xlabel, path, width, metric="precision_at_k", K=100, log2=False, levels=None,
                    style="line"):
    d = acc[(acc.experiment == experiment) & (acc.K == K) & acc.method.isin(METHODS)].copy()
    if d.empty:
        print("no data for", path); return
    d["x"] = d[xcol].map(lambda v: float(json.loads(v)) if isinstance(v, str) else float(v))
    if levels:
        d = d[d.x.isin([float(v) for v in levels])]
    fig, axes = panels(len(layouts), width)
    for ax, lay in zip(axes, layouts):
        if style == "bar":
            cats = sorted(d.x.unique()); vals, errs = {}, {}
            for m in METHODS:
                s = d[(d.layout == lay) & (d.method == m)].set_index("x")
                vals[m] = [s[f"{metric}_mean"].get(c, np.nan) for c in cats]; errs[m] = [s[f"{metric}_sd"].get(c, np.nan) for c in cats]
            bars(ax, [f"{c:g}" for c in cats], vals, errs)
            ax.set_xlabel(xlabel); ax.set_ylim(0, 1.02)
            continue
        for m in METHODS:
            s = d[(d.layout == lay) & (d.method == m)].dropna(subset=[f"{metric}_mean"]).sort_values("x")
            if not s.empty:
                line(ax, s.x, s[f"{metric}_mean"], m, s[f"{metric}_sd"])
        ax.set_xlabel(xlabel); ax.set_ylim(0, 1.02)
        if log2:
            ax.set_xscale("log", base=2); ax.set_xticks(sorted(d.x.unique())); ax.set_xticklabels([f"{v:g}" for v in sorted(d.x.unique())])
    axes[0].set_ylabel("Precision@K")
    finish(fig, axes, [name(l) for l in layouts], bar_handles(METHODS) if style == "bar" else method_handles(METHODS), path)


def timing_figure(tm, parts, xcol, xlabel, path, width, style="line"):
    """parts: [(experiment, dataset, values of the varied parameter)], one subplot each"""
    fig, axes = panels(len(parts), width)
    any_data = False
    label = lambda v: f"{v / 1000:.3g}k" if v >= 1000 and xcol == "n" else f"{v:g}"
    for ax, (e, ds, levels) in zip(axes, parts):
        d = tm[(tm.experiment == e) & (tm.dataset == ds) & tm.value.astype(float).isin([float(v) for v in levels])]
        if style == "bar":
            cats = sorted(d[xcol].unique())
            vals = {m: [d[(d.program == m) & (d[xcol] == c)].mean_seconds.mean() for c in cats] for m in METHODS}
            any_data = any_data or not d.empty
            bars(ax, [label(c) for c in cats], vals)
            ax.set_xlabel(xlabel); ax.set_yscale("log")
            lo = np.nanmin([v for vs in vals.values() for v in vs])     # columns start at a power of ten below the smallest
            if np.isfinite(lo) and lo > 0:
                ax.set_ylim(bottom=10 ** np.floor(np.log10(lo * 0.9)))
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
            ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            continue
        for m in METHODS:
            s = d[d.program == m].sort_values(xcol)
            s = s.dropna(subset=["mean_seconds"])
            if not s.empty:
                line(ax, s[xcol], s.mean_seconds, m); any_data = True
        ax.set_xlabel(xlabel); ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if not d.empty:                        # the three values as ticks (K and the instances on a log scale)
            ticks = sorted(d[xcol].unique())
            if xcol in ("K", "n"):
                ax.set_xscale("log"); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks(ticks); ax.set_xticklabels([f"{v / 1000:.3g}k" if v >= 1000 and xcol == "n" else f"{v:g}" for v in ticks])
    if not any_data:
        plt.close(fig); print("no data for", path); return
    axes[0].set_ylabel("Running time (s)")
    finish(fig, axes, [name(ds) for _, ds, _ in parts], bar_handles(METHODS) if style == "bar" else method_handles(METHODS), path)


def hold_figure(curve, path, K=300):
    layouts = ["New_York", "Tokyo", "Tampa", "Philadelphia"]
    d = curve[curve.K == K]
    if d.empty:
        print("no data for", path); return
    fig, axes = panels(len(layouts), DOUBLE)
    for ax, city in zip(axes, layouts):
        vals = {}
        for m in METHODS:
            s = d[(d.city == city) & (d.method == m)].set_index("n")
            vals[m] = [s.hold_rate.get(n, np.nan) for n in LEVELS["n"]]
        bars(ax, [str(n) for n in LEVELS["n"]], vals)       # grouped columns: the three methods side by side, no overlap
        ax.set_xlabel("Number of patterns n"); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Hold rate")
    finish(fig, axes, [name(c) for c in layouts], bar_handles(METHODS), path)


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
    finish(fig, axes, labels, handles, path, inside=False)


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
    acc, tm, curve = (read(proc, f) for f in ("accuracy.csv", "timing.csv", "hold_curve.csv"))
    out = lambda f: os.path.join(a.out, f)
    if not acc.empty:
        accuracy_figure(acc, "B1_evidence", ["Philadelphia", "Tampa", "New_York", "Tokyo"], "value", "Evidence factor c",
                        out("fig_evidence.pdf"), DOUBLE, levels=LEVELS["c"], style="bar")
        k = acc[acc.experiment == "B3_threshold"]
        if not k.empty:
            accuracy_figure_k(k, out("fig_threshold.pdf"))
    if not tm.empty:
        timing_figure(tm, [("C1_features", "Synthetic", LEVELS["features"]), ("C1_features", "New_York", LEVELS["features"])],
                      "features", "Number of features", out("fig_time_features.pdf"), SINGLE)
        timing_figure(tm, [("C2_instances_synthetic", "Synthetic", LEVELS["instances"]),
                           ("C2_instances_new_york", "New_York", LEVELS["share"])], "n",
                      "Number of instances", out("fig_time_instances.pdf"), SINGLE, style="bar")
        timing_figure(tm, [("C3_threshold", c, LEVELS["K"]) for c in ("New_Orleans", "New_York")], "K",
                      "Threshold K", out("fig_time_threshold.pdf"), SINGLE, style="bar")
        timing_figure(tm, [("C4_california", "California", LEVELS["d_california"])], "d", "Distance threshold d (m)",
                      out("fig_time_distance.pdf"), SINGLE)
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
