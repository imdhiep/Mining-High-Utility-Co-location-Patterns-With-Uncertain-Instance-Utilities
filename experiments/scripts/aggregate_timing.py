"""Running-time experiments (Section VI-C): raw -> processed (design 5.1: the mean of the timed runs; the largest
relative spread between runs is kept for the sentence in the settings).

results/processed/timing.csv: one row per (experiment, dataset, value, program) with the mean and median wall time,
the relative spread (max - min) / mean of the timed runs, the number of reported patterns, cliques, evaluated and
visited candidates (UHUCPM and PSO-MUCP) and the peak memory (all programs).
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
