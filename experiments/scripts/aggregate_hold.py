"""Hold rates on real data (Section VI-D.1): raw -> processed. A pattern holds when its test score reaches gamma_K.

Files written to results/processed/:
  hold_sets.csv      hold rate of the decision set of each method (UHUCPM S_delta, CQ-HUCPM, PSO-MUCP), per city and K.
  hold_curve.csv     hold rate of the first n patterns of each ranking (UHUCPM by p, CQ-HUCPM by its total, PSO-MUCP by
                     its feature-level score), n = 10, 20, ... and the full ranking.
  hold_disagree.csv  at the same n (and for the decision sets): patterns kept by CQ-HUCPM only, by UHUCPM only, by both;
                     size and hold rate of each part.
  hold_diff.csv      difference of hold rates at the same n between UHUCPM and each other method: mean over the
                     cities and a 95% percentile bootstrap interval over the cities (design 5.1; BOOT resamples).
Usage: python aggregate_hold.py [--results DIR] [--n 100]"""
import argparse, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness.processed import ranked, raw_records, write_csv  # noqa: E402

BOOT = 10000
BOOT_SEED = 0
STEP = 10


def hold(patterns, test_score, gamma):
    """(number held, hold rate); a pattern without a test score is an error of the run"""
    patterns = list(patterns)
    if not patterns:
        return 0, None
    missing = [q for q in patterns if q not in test_score]
    if missing:
        raise KeyError(f"no test score for {missing[:5]}")
    held = sum(test_score[q] >= gamma for q in patterns)
    return held, held / len(patterns)


def bootstrap_ci(diffs):
    rng = random.Random(BOOT_SEED); n = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results")); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    records = [r for r in raw_records(a.results, "D1_hold") if "per_K" in r]
    sets, curve, dis, at_n = [], [], [], {}
    for r in records:
        for K, k in r["per_K"].items():
            if k.get("skipped"):
                continue
            K = int(K); ts, g = k["test_score"], k["gamma"]
            for m in ("uhucpm", "cq", "pso"):
                if k[m] is None:
                    continue
                held, rate = hold(k[m], ts, g)
                sets.append(dict(city=r["city"], K=K, gamma=g, method=m, n=len(k[m]), held=held, hold_rate=rate))
            tops = {}
            for m, score_of in k["ranking"].items():
                order = ranked(score_of); tops[m] = order
                for n in sorted(set(range(STEP, len(order) + 1, STEP)) | ({len(order)} if order else set())):
                    held, rate = hold(order[:n], ts, g)
                    curve.append(dict(city=r["city"], K=K, method=m, n=n, held=held, hold_rate=rate))
                if len(order) >= a.n:
                    at_n[(r["city"], K, m)] = hold(order[:a.n], ts, g)[1]
            pairs = [("top", a.n, set(tops["uhucpm"][:a.n]), set(tops["cq"][:a.n]))] if min(len(tops["uhucpm"]), len(tops["cq"])) >= a.n else []
            pairs.append(("decision", None, set(k["uhucpm"]), set(k["cq"])))
            for kind, n, U, C in pairs:
                for part, S in (("cq_only", C - U), ("uhucpm_only", U - C), ("both", U & C)):
                    held, rate = hold(S, ts, g)
                    dis.append(dict(city=r["city"], K=K, compared=kind, n=n, part=part, size=len(S), held=held, hold_rate=rate))
    diff = []
    for K in sorted({K for _, K, _ in at_n}):
        cities = sorted({c for c, k, _ in at_n if k == K})
        for other in ("cq", "pso"):
            d = [at_n[(c, K, "uhucpm")] - at_n[(c, K, other)] for c in cities if (c, K, "uhucpm") in at_n and (c, K, other) in at_n]
            if d:
                lo, hi = bootstrap_ci(d)
                diff.append(dict(K=K, n=a.n, compared=f"uhucpm-{other}", cities=len(d), mean_difference=sum(d) / len(d),
                                 ci_low=lo, ci_high=hi))
    out = os.path.join(a.results, "processed")
    write_csv(os.path.join(out, "hold_sets.csv"), sets, ["city", "K", "gamma", "method", "n", "held", "hold_rate"])
    write_csv(os.path.join(out, "hold_curve.csv"), curve, ["city", "K", "method", "n", "held", "hold_rate"])
    write_csv(os.path.join(out, "hold_disagree.csv"), dis, ["city", "K", "compared", "n", "part", "size", "held", "hold_rate"])
    write_csv(os.path.join(out, "hold_diff.csv"), diff, ["K", "n", "compared", "cities", "mean_difference", "ci_low", "ci_high"])
    skipped = [r["city"] for r in raw_records(a.results, "D1_hold") if "skipped" in r]
    print(len(records), "cities", "skipped:", skipped, "processed:", out)


if __name__ == "__main__":
    main()
