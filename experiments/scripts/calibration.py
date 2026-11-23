"""Calibration of the probability p and paired tests of precision@K (Section VI-B), from the raw B1_evidence records.

calibration.csv   per city at K = 100, c = 1: the patterns of the UHUCPM ranking (its first RANKING_DEPTH * K patterns,
                  20 generated datasets pooled) grouped by p = Phi(t); for each group the number of patterns, their mean
                  p and the share that are true patterns
paired_tests.csv  per city and evidence factor c at K = 100: precision@K of UHUCPM minus that of CQ-HUCPM over the
                  generated datasets (mean difference, wins, ties, losses, two-sided Wilcoxon signed-rank p-value)
Usage: python calibration.py [--results DIR]"""
import argparse, csv, glob, json, os
from statistics import NormalDist

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
BINS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0 + 1e-12]


def records(raw, experiment):
    for f in sorted(glob.glob(os.path.join(raw, experiment, "*.jsonl"))):
        for line in open(f):
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results"))
    a = ap.parse_args()
    raw, proc = os.path.join(a.results, "raw"), os.path.join(a.results, "processed")
    phi = NormalDist().cdf
    calib, pairs = {}, {}
    for r in records(raw, "B1_evidence"):
        k = r["result"]["per_K"].get("100")
        if not k or "ranking" not in k:
            continue
        key = (r["layout"], r["value"])
        pairs.setdefault(key, {})[r["seed"]] = (k["uhucpm"]["precision_at_k"], k["cq"]["precision_at_k"])
        if r["value"] != 1:
            continue
        T = set(k["true"])
        for q, t in k["ranking"]["uhucpm"].items():
            p = phi(t)
            i = next(j for j in range(len(BINS) - 1) if BINS[j] <= p < BINS[j + 1])
            c = calib.setdefault((r["layout"], i), [0, 0.0, 0])
            c[0] += 1; c[1] += p; c[2] += q in T
    with open(os.path.join(proc, "calibration.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["city", "p_low", "p_high", "patterns", "mean_p", "true_share"])
        for (city, i), (n, ps, h) in sorted(calib.items()):
            w.writerow([city, BINS[i], min(BINS[i + 1], 1.0), n, ps / n, h / n])
    with open(os.path.join(proc, "paired_tests.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["city", "c", "datasets", "mean_difference", "wins", "ties", "losses", "wilcoxon_p"])
        for (city, c), by_seed in sorted(pairs.items()):
            d = np.array([u - q for u, q in by_seed.values()])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            w.writerow([city, c, len(d), d.mean(), int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum()), p])
    print("processed:", os.path.join(proc, "calibration.csv"), os.path.join(proc, "paired_tests.csv"))


if __name__ == "__main__":
    main()
