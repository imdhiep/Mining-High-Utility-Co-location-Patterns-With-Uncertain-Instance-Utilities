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
    return {"Saint_Louis": "St. Louis"}.get(s, s.replace("_", " "))


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


def table_mining(acc, path):
    """precision@K under three values of d, of the number of features and of the share of instances"""
    items = [(f"$d = {v:g}$~m", "New_Orleans", "B3_distance", v) for v in (30.0, 50.0, 70.0)] + \
            [(f"{v} features", "New_York", "B3_features", v) for v in (10, 20, 40)] + \
            [(f"{v:.0%} instances".replace("%", "\\%"), "New_York", "B3_share", v) for v in (0.25, 0.5, 1.0)]
    rows = []
    for i, (label, city, e, v) in enumerate(items):
        if i and i % 3 == 0:
            rows.append("MIDRULE")
        r = [pick(acc, e, city, v, m) for m, _ in METHODS]
        rows.append([label, name(city) if i % 3 == 0 else ""] + [fmt(None if x is None else x["precision_at_k_mean"]) for x in r])
    names = " & ".join(lab for _, lab in METHODS)
    write(path, "Precision@$K$ under different $d$, numbers of features and shares of instances "
          "(all cases $c=1$, $K=100$).", "tab:mining", "@{}ll" + "r" * len(METHODS) + "@{}", [f"Setting & City & {names} \\\\"], rows,
          colsep=3)

def table_count(cn, path):
    """number of reported patterns and largest pattern size of every method under the three values of K (C.3 counts)"""
    cities = ["New_Orleans", "New_York"]
    d = cn[cn.experiment == "C3_threshold"]
    rows = []
    for i, c in enumerate(cities):
        if i:
            rows.append("MIDRULE")
        for j, K in enumerate((100, 300, 1000)):
            cells = [name(c) if j == 0 else "", str(K)]
            for m, _ in METHODS:
                r = d[(d.dataset == c) & (d.value.astype(float) == K) & (d.program == m)]
                ok = not r.empty and not pd.isna(r.patterns.iloc[0]) and r.patterns.iloc[0] > 0
                cells += [str(int(r.patterns.iloc[0])) if ok else DASH,
                          str(int(r.largest.iloc[0])) if ok and not pd.isna(r.largest.iloc[0]) else DASH]
            rows.append(cells)
    top, rules, sub = method_header([("n", "Patterns"), ("l", "Largest")])
    write(path, "Number of reported patterns and largest pattern size under different threshold values $K$.", "tab:count",
          "ll" + "rr" * len(METHODS), [f" & & {top} \\\\", rules, f"City & $K$ & {sub} \\\\"], rows, wide=True)

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
        table_mining(acc, out("tab_mining.tex"))
    cn = read("counts.csv")
    if not cn.empty:
        table_count(cn, out("tab_count.tex"))
    if not curve.empty:
        table_hold(curve, dis, out("tab_hold.tex"))


if __name__ == "__main__":
    main()
