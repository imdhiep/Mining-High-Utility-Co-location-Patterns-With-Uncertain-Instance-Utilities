"""Tables of Section VI from results/processed (claude/EXPERIMENT_PRESENTATION_RULES.md): one quantity per column,
"–" for a missing or empty result, no bold best values. Reads only the processed CSV files; writes one LaTeX file per
table (booktabs).
Usage: python make_tables.py [--results DIR] [--out DIR]"""
import argparse, json, math, os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [("uhucpm", "UHUCPM"), ("cq", "CQ-HUCPM"), ("pso", "PSO-MUCP")]
DASH = "--"   # en dash in LaTeX


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return DASH
    return f"{v:.{digits}f}"


def name(s):
    return s.replace("_", " ")


def value_of(v):
    return json.loads(v) if isinstance(v, str) else v


def write(path, caption, label, colspec, header, rows, wide=False, colsep=None):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize"] + ([f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}"] if colsep else []) + \
            [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"] + header + ["\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == "MIDRULE" else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("table:", path)


def pick(acc, experiment, layout, value, method, K=100):
    d = acc[(acc.experiment == experiment) & (acc.layout == layout) & (acc.K == K) & (acc.method == method)]
    d = d[d.value.map(lambda v: value_of(v) == value)]
    return d.iloc[0] if not d.empty else None


def method_header(metrics):
    top = " & ".join(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{label}}}" for _, label in METHODS)
    rules = " ".join(f"\\cmidrule(lr){{{3 + i * len(metrics)}-{2 + (i + 1) * len(metrics)}}}" for i in range(len(METHODS)))
    sub = " & ".join(" & ".join(lab for _, lab in metrics) for _ in METHODS)
    return top, rules, sub


def table_accuracy(acc, path):
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("n", "$|R|$")]
    top, rules, sub = method_header(metrics)
    rows = []
    for i, city in enumerate(["Philadelphia", "Tampa", "New_York", "Tokyo"]):
        if i:
            rows.append("MIDRULE")
        for j, c in enumerate([0.5, 1, 4]):
            cells = [name(city) if j == 0 else "", f"{c:g}"]
            for m, _ in METHODS:
                r = pick(acc, "B1_evidence", city, c, m)
                cells += [fmt(None if r is None else r[f"{k}_mean"], 1 if k == "n" else 2) for k, _ in metrics]
            rows.append(cells)
    write(path, "Precision, recall and number of reported patterns under different evidence factors $c$, with $K = 100$, "
          "$d = 50$~m and 20 features.", "tab:accuracy", "ll" + "rrr" * len(METHODS),
          [f" & & {top} \\\\", rules, f"City & $c$ & {sub} \\\\"], rows, wide=True)


def two_metric_rows(acc, items):
    """items: (label, city, experiment, value)"""
    rows = []
    for label, city, e, v in items:
        cells = [label, name(city)]
        for m, _ in METHODS:
            r = pick(acc, e, city, v, m)
            cells += [fmt(None if r is None else r["precision_mean"]), fmt(None if r is None else r["recall_mean"])]
        rows.append(cells)
    return rows


def table_model(acc, path):
    scen = [("Gamma log-rates", "B2_law", "gamma"), ("Mixture log-rates", "B2_law", "mixture"),
            ("Spatial correlation $\\ell = 200$~m", "B2_spatial", 200.0), ("Spatial correlation $\\ell = 500$~m", "B2_spatial", 500.0),
            ("Truncation $t = 1$", "B2_truncation", 1)]
    items = [(lab if i == 0 else "", city, e, v) for lab, e, v in scen for i, city in enumerate(["Tampa", "New_York"])]
    top, rules, sub = method_header([("p", "Precision"), ("r", "Recall")])
    write(path, "Precision and recall when the data depart from the check-in model, with $c = 1$, $K = 100$ and $d = 50$~m.",
          "tab:model", "ll" + "rr" * len(METHODS), [f" & & {top} \\\\", rules, f"Scenario & City & {sub} \\\\"],
          two_metric_rows(acc, items), wide=True)


def table_mining(acc, path):
    items = [(f"$d = {v:g}$~m", "New_Orleans", "B3_distance", v) for v in (30.0, 50.0, 70.0)] + \
            [(f"{v} features", "New_York", "B3_features", v) for v in (10, 20, 30, 40)] + \
            [(f"{v:.0%} of instances".replace("%", "\\%"), "New_York", "B3_share", v) for v in (0.25, 0.5, 0.75, 1.0)]
    top, rules, sub = method_header([("p", "Precision"), ("r", "Recall")])
    write(path, "Precision and recall under different distance thresholds $d$, numbers of features and shares of instances.",
          "tab:mining", "ll" + "rr" * len(METHODS), [f" & & {top} \\\\", rules, f"Setting & City & {sub} \\\\"],
          two_metric_rows(acc, items), wide=True)


def table_count(ans, path):
    """number of patterns and largest pattern size of UHUCPM (S_delta) by K and delta on the real datasets of C.3"""
    cities = ["New_Orleans", "Tampa", "New_York"]
    d = ans[ans.rule == "set"]
    rows = []
    for K in sorted(d.K.unique()):
        for j, delta in enumerate(sorted(d.delta.unique())):
            cells = [str(K) if j == 0 else "", f"{delta:g}"]
            for c in cities:
                r = d[(d.K == K) & (d.delta == delta) & (d.city == c)]
                cells += [DASH if r.empty else str(int(r.patterns.iloc[0])),
                          DASH if r.empty or pd.isna(r.largest.iloc[0]) else str(int(r.largest.iloc[0]))]
            rows.append(cells)
        rows.append("MIDRULE")
    rows = rows[:-1]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name(c)}}}" for c in cities)
    rules = " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(cities)))
    sub = " & ".join("Patterns & Largest" for _ in cities)
    write(path, "Number of patterns and largest pattern size of UHUCPM under different $K$ and $\\delta$.", "tab:count",
          "ll" + "rr" * len(cities), [f" & & {head} \\\\", rules, f"$K$ & $\\delta$ & {sub} \\\\"], rows, colsep=4)


def table_pso(tm, path):
    d = tm[tm.experiment.str.startswith("C4")]
    rows = []
    for ds in sorted(d.dataset.unique()):
        for v in sorted(d[d.dataset == ds].value.unique(), key=float):
            s = d[(d.dataset == ds) & (d.value == v)]
            get = lambda m, col: s[s.program == m][col].iloc[0] if (s.program == m).any() else None
            cells = [name(ds) if not rows or rows[-1][0] != name(ds) else "", f"{float(v):g}"]
            cells += [fmt(get(m, "mean_seconds"), 2) for m, _ in METHODS]
            cells += [fmt(None if get(m, "max_rss_kb") is None else get(m, "max_rss_kb") / 1024, 0) for m, _ in METHODS]
            rows.append(cells)
    names = " & ".join(lab for _, lab in METHODS)
    write(path, "Running time and peak memory on the datasets of PSO-MUCP.", "tab:psodata", "ll" + "rrr" * 2,
          ["& & \\multicolumn{3}{c}{Running time (s)} & \\multicolumn{3}{c}{Peak memory (MB)} \\\\",
           "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}", f"Dataset & $d$ (m) & {names} & {names} \\\\"], rows, wide=True)


def table_hold(curve, dis, path, K=300, n=100):
    cities = ["Philadelphia", "New_Orleans", "Tampa", "New_York", "Tokyo", "Tucson", "Indianapolis", "Nashville", "Reno",
              "Saint_Louis", "Edmonton", "Santa_Barbara", "Boise"]
    rows = []
    for c in cities:
        cells = [name(c)]
        for m, _ in METHODS:
            r = curve[(curve.city == c) & (curve.K == K) & (curve.method == m) & (curve.n == n)]
            cells.append(fmt(r.hold_rate.iloc[0] if not r.empty else None))
        for part in ("cq_only", "both"):
            r = dis[(dis.city == c) & (dis.K == K) & (dis.compared == "top") & (dis.part == part)]
            cells.append(fmt(r.hold_rate.iloc[0] if not r.empty else None))
        rows.append(cells)
    names = " & ".join(lab for _, lab in METHODS)
    write(path, f"Hold rate of the top {n} patterns at $K = {K}$ on the 13 real datasets.", "tab:hold", "l" + "r" * 5,
          ["& \\multicolumn{3}{c}{Top " + str(n) + " patterns} & \\multicolumn{2}{c}{Disagreement} \\\\",
           "\\cmidrule(lr){2-4} \\cmidrule(lr){5-6}", f"City & {names} & \\shortstack{{CQ-HUCPM\\\\only}} & Both \\\\"], rows, colsep=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.dirname(HERE)
    ap.add_argument("--results", default=os.path.join(root, "results")); ap.add_argument("--out", default=os.path.join(root, "tables"))
    a = ap.parse_args()
    proc = os.path.join(a.results, "processed"); os.makedirs(a.out, exist_ok=True)
    read = lambda f: pd.read_csv(os.path.join(proc, f)) if os.path.exists(os.path.join(proc, f)) else pd.DataFrame()
    acc, tm, curve, dis = read("accuracy.csv"), read("timing.csv"), read("hold_curve.csv"), read("hold_disagree.csv")
    out = lambda f: os.path.join(a.out, f)
    if not acc.empty:
        table_accuracy(acc, out("tab_accuracy.tex")); table_model(acc, out("tab_model.tex"))
        table_mining(acc, out("tab_mining.tex"))
    ans = read("answer_counts.csv")
    if not ans.empty:
        table_count(ans, out("tab_count.tex"))
    if not tm.empty:
        table_pso(tm, out("tab_psodata.tex"))
    if not curve.empty:
        table_hold(curve, dis, out("tab_hold.tex"))


if __name__ == "__main__":
    main()
