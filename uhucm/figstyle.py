"""Figure style matched to the accepted TKDE paper used as the template (MPRIC, TKDE 38(7), 2026):
Office-style palette, serif (Times-like) text, legends below the axes, light horizontal grid, log2 bar charts with a
data table under the x-axis, marker line charts with value labels, sub-panels captioned (a), (b) underneath."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

BLUE, ORANGE, GRAY, YELLOW, LBLUE, GREEN = "#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#70AD47"
PALETTE = [BLUE, ORANGE, GRAY, YELLOW, LBLUE, GREEN, "#264478", "#9E480E", "#636363"]
MARKERS = ["o", "s", "^", "D", "v", "x"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": "#808080", "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
plt.rcParams["axes.grid.axis"] = "y"


def panel_label(ax, text, y=-0.30):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def legend_below(fig_or_ax, handles=None, labels=None, ncol=4, y=-0.02, **kw):
    if hasattr(fig_or_ax, "legend") and handles is not None:
        return fig_or_ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, **kw)
    return fig_or_ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, **kw)


def bar_with_table(ax, cats, series, log2=True, ylabel="", fmt="{:.2f}", colors=None):
    """Grouped bars (log2 y-axis) with an Excel-style data table under the x-axis, as in Fig. 9 of the template."""
    import numpy as np
    colors = colors or PALETTE
    n = len(series); w = 0.8 / n; x = np.arange(len(cats))
    for i, (name, vals) in enumerate(series):
        v = [np.nan if (val is None) else val for val in vals]
        ax.bar(x - 0.4 + w * (i + 0.5), v, w, color=colors[i % len(colors)], label=name, linewidth=0)
    allv = [v for _, vals in series for v in vals if v is not None and v > 0]
    if log2 and allv and max(allv) / min(allv) > 8:
        ax.set_yscale("log", base=2)
    ax.set_xticks([]); ax.set_xlim(-0.5, len(cats) - 0.5); ax.set_ylabel(ylabel)
    cell = [[("" if (val is None) else (fmt.format(val) if not isinstance(val, str) else val)) for val in vals] for _, vals in series]
    tab = ax.table(cellText=cell, rowLabels=[nm for nm, _ in series], colLabels=cats, loc="bottom", cellLoc="center",
                   rowColours=[colors[i % len(colors)] + "55" for i in range(n)])
    tab.auto_set_font_size(False); tab.set_fontsize(6); tab.scale(1, 1.05)
    for (r, c), cl in tab.get_celld().items():
        cl.set_linewidth(0.4); cl.set_edgecolor("#808080")
    return tab


def line(ax, x, series, labels=True, fmt="{:.2f}", xlabel="", ylabel="", logy=False):
    for i, (name, ys) in enumerate(series):
        ax.plot(x, ys, marker=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)], label=name)
        if labels:
            for xi, yi in zip(x, ys):
                if yi is not None:
                    ax.annotate(fmt.format(yi), (xi, yi), textcoords="offset points", xytext=(0, 4 if i % 2 == 0 else -9), ha="center", fontsize=5.5,
                                color=PALETTE[i % len(PALETTE)])
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def finalize(fig, axes, labels, handles=None, legend_labels=None, ncol=4, bottom=0.34, leg_y=0.13, lab_y=0.015):
    """Template layout: axes, then one shared legend, then the sub-captions (a), (b) at the bottom."""
    fig.subplots_adjust(bottom=bottom)
    if handles is not None:
        fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, leg_y), ncol=ncol)
    for ax, lab in zip(axes, labels):
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, lab_y, lab, ha="center", va="bottom", fontsize=8)
