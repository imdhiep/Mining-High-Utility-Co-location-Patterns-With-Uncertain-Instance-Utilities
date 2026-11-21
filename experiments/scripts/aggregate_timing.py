"""Running-time experiments (Section VI-C): raw -> processed (design 5.1: the mean of the timed runs; the largest
relative spread between runs is kept for the sentence in the settings).

results/processed/timing.csv: one row per (experiment, dataset, value, program) with the mean and median wall time,
the relative spread (max - min) / mean of the timed runs, the number of reported patterns, cliques, evaluated and
visited candidates (UHUCPM and PSO-MUCP) and the peak memory (all programs).
results/processed/counts.csv (from the untimed runs, run_timing.py --counts): evaluated patterns of every program, the
number of candidate patterns (patterns with a row instance, all evaluated by CQ-HUCPM) and the pruning ratio
1 - evaluated / candidates (definition (7) of PSO-MUCP).
results/processed/local_rule.csv and answer_counts.csv (from run_counts_real.py): patterns missed by the local pruning
rules, and the number of patterns and largest pattern size of UHUCPM by K and delta on the real datasets.
Usage: python aggregate_timing.py [--results DIR] [EXPERIMENT ...]"""
import argparse, glob, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness.processed import raw_records, write_csv  # noqa: E402

COLUMNS = ["experiment", "dataset", "vary", "value", "n", "features", "d", "K", "gamma", "program", "runs", "mean_seconds",
           "median_seconds", "relative_spread", "patterns", "cliques", "evaluated", "visited", "max_rss_kb", "timeout"]


def cq_memory(raw):
    m = re.search(r"Peack memory usage\s*:\s*(\d+)", raw or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*"); ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results"))
    a = ap.parse_args()
    names = a.experiments or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.results, "raw", "C*")))
    count_rows = []
    for e in [n for n in names if n.endswith("_counts")]:
        for r in raw_records(a.results, e):
            total = (r["programs"].get("cq") or {}).get("evaluated")
            for prog, p in r["programs"].items():
                ev = p.get("evaluated")
                count_rows.append(dict(experiment=e[:-len("_counts")], dataset=r["dataset"], vary=r["vary"], value=r["value"], n=r["n"],
                                       features=r["features"], program=prog, evaluated=ev, candidates=total, patterns=p.get("n"),
                                       pruning_ratio=(1 - ev / total) if ev is not None and total else None))
    if count_rows:
        print("counts:", write_csv(os.path.join(a.results, "processed", "counts.csv"), count_rows,
                                   ["experiment", "dataset", "vary", "value", "n", "features", "program", "evaluated", "candidates",
                                    "patterns", "pruning_ratio"]))
    real = raw_records(a.results, "C0_counts_real")
    if real:
        lr, ans = [], []
        for r in real:
            for K, k in r["per_K"].items():
                if k.get("skipped"):
                    continue
                for z, e in k["local_rule"].items():
                    for rule in ("sum", "global"):
                        lr.append(dict(city=r["city"], K=int(K), z=float(z), rule=rule, exact=e["exact"], found=e[rule]["found"],
                                       missed=len(e[rule]["missed"]), extra=len(e[rule]["extra"])))
                for delta, v in k["set_rule"].items():
                    ans.append(dict(city=r["city"], K=int(K), rule="set", delta=float(delta), patterns=v["n"], largest=v["largest"]))
                t = k.get("threshold_rule")
                if t:
                    ans.append(dict(city=r["city"], K=int(K), rule="threshold", delta=None, patterns=t["n"], largest=t["largest"]))
        print("local rule:", write_csv(os.path.join(a.results, "processed", "local_rule.csv"), lr,
                                       ["city", "K", "z", "rule", "exact", "found", "missed", "extra"]))
        print("answer sizes:", write_csv(os.path.join(a.results, "processed", "answer_counts.csv"), ans,
                                         ["city", "K", "rule", "delta", "patterns", "largest"]))
    names = [n for n in names if not n.endswith("_counts") and n != "C0_counts_real"]
    rows = []
    for e in names:
        records = raw_records(a.results, e)
        for r in records:
            for prog, p in r["programs"].items():
                t = p["seconds"]; st = p.get("stats") or {}
                mean = statistics.fmean(t)
                rows.append(dict(experiment=e, dataset=r["dataset"], vary=r["vary"], value=r["value"], n=r["n"],
                                 features=r["features"], d=r["d"], K=r["K"], gamma=r["gamma"], program=prog, runs=len(t),
                                 mean_seconds=mean, median_seconds=statistics.median(t),
                                 relative_spread=(max(t) - min(t)) / mean if mean > 0 else None,
                                 patterns=p.get("n"), cliques=st.get("cliques"), evaluated=st.get("evaluated"),
                                 visited=st.get("visited"),
                                 max_rss_kb=st.get("max_rss_kb") if prog != "cq" else cq_memory(p.get("raw")),
                                 timeout=st.get("timeout")))
        print(e, len(records), "points", "code", sorted({r.get("code") for r in records}))
    spread = [r["relative_spread"] for r in rows if r["relative_spread"] is not None and r["mean_seconds"] >= 0.1]
    if spread:
        print(f"largest relative spread between timed runs (runs of at least 0.1 s): {max(spread):.3f}")
    path = write_csv(os.path.join(a.results, "processed", "timing.csv"), rows, COLUMNS)
    print("processed:", path)


if __name__ == "__main__":
    main()
