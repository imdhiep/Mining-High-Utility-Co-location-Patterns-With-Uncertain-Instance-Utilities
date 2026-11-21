"""Accuracy experiments (Section VI-B): raw -> processed. Means and standard deviations over the seeds (design 5.1).

Files written to results/processed/:
  accuracy.csv   one row per (experiment, layout, value, K, method): precision, recall, |R|, precision@K and the largest
                 pattern size (mean and
                 standard deviation over the seeds with a non-empty answer), the number of seeds, of empty answers and
                 of seeds where the method did not run. Methods: uhucpm (S_delta), cq, pso, uhucpm_threshold (with the
                 coverage of its lower bound), set_rule at every level delta, and the configurations stated only in the
                 text (extras of B.1).
  pr_curve.csv   precision and recall of the first n patterns of each ranking (UHUCPM by p, CQ-HUCPM by its total, PSO-MUCP by its
                 feature-level score where recorded),
                 n = 1 .. RANKING_DEPTH * K, averaged over the seeds whose ranking has at least n patterns.
  posterior.csv  largest posterior mass on the local-grid edges and the posterior time, per experiment and layout.
Usage: python aggregate_accuracy.py [--results DIR] [EXPERIMENT ...]"""
import argparse, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness.processed import mean_sd, ranked, raw_records, write_csv  # noqa: E402

EXPERIMENTS = ["B1_evidence", "B1_pr", "B2_overdispersion", "B2_law", "B2_spatial", "B2_truncation", "B2_rating", "B3_threshold", "B3_distance",
               "B3_features", "B3_share"]
METRICS = ["precision", "recall", "n", "precision_at_k", "coverage", "size_max"]
COLUMNS = ["experiment", "layout", "vary", "value", "K", "method", "delta", "seeds", "empty", "not_run"] + \
          [f"{m}_{s}" for m in METRICS for s in ("mean", "sd")]


def method_results(k):
    """(method, delta, result dict or None) of one per_K entry"""
    out = [(m, None, k.get(m)) for m in ("uhucpm", "cq", "pso")]
    thr = dict(k["uhucpm_threshold"] or {})
    thr["coverage"] = (k.get("uhucpm_lower_bound") or {}).get("coverage")
    out.append(("uhucpm_threshold", None, thr))
    out += [("set_rule", float(d), r) for d, r in k.get("set_rule", {}).items()]
    out += [(name, None, r) for name, r in (k.get("extras") or {}).items()]
    return out


def summarize(records, experiment):
    groups = defaultdict(list)
    for r in records:
        for K, k in r["result"]["per_K"].items():
            if k.get("skipped"):
                continue
            for method, delta, res in method_results(k):
                groups[(r["layout"], r["vary"], json.dumps(r["value"]), int(K), method, delta)].append(res)
    rows = []
    for (layout, vary, value, K, method, delta), results in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ran = [x for x in results if x is not None]
        row = dict(experiment=experiment, layout=layout, vary=vary, value=value, K=K, method=method, delta=delta,
                   seeds=len(results), not_run=len(results) - len(ran), empty=sum(1 for x in ran if not x.get("n")))
        for x in ran:                          # largest pattern size (number of features) in the reported set
            if x.get("patterns"):
                x["size_max"] = max(len(q) for q in x["patterns"])
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_sd"] = mean_sd([x.get(m) for x in ran])
        rows.append(row)
    return rows


def pr_curve(records, experiment):
    curves = defaultdict(lambda: defaultdict(list))   # key -> n -> [(precision, recall)]
    for r in records:
        for K, k in r["result"]["per_K"].items():
            if k.get("skipped"):
                continue
            T = set(k["true"])
            for method, score_of in k["ranking"].items():
                hits = 0
                for n, q in enumerate(ranked(score_of), 1):
                    hits += q in T
                    curves[(r["layout"], json.dumps(r["value"]), int(K), method)][n].append((hits / n, hits / len(T)))
    rows = []
    for (layout, value, K, method), by_n in sorted(curves.items()):
        for n in sorted(by_n):
            pts = by_n[n]
            p, p_sd = mean_sd([a for a, _ in pts]); rc, rc_sd = mean_sd([b for _, b in pts])
            rows.append(dict(experiment=experiment, layout=layout, value=value, K=K, method=method, n=n, seeds=len(pts),
                             precision_mean=p, precision_sd=p_sd, recall_mean=rc, recall_sd=rc_sd))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="*"); ap.add_argument("--results", default=os.path.join(os.path.dirname(HERE), "results"))
    a = ap.parse_args()
    acc, pr, post = [], [], []
    for e in a.experiments or EXPERIMENTS:
        records = raw_records(a.results, e)
        if not records:
            print(e, "no raw results"); continue
        acc += summarize(records, e); pr += pr_curve(records, e)
        by_layout = defaultdict(list)
        for r in records:
            by_layout[r["layout"]].append(r["result"])
        for layout, res in sorted(by_layout.items()):
            edge = [x["local_edge_mass_max"] for x in res if x.get("local_edge_mass_max") is not None]
            post.append(dict(experiment=e, layout=layout, runs=len(res), edge_mass_max=max(edge) if edge else None,
                             posterior_seconds_mean=mean_sd([x.get("posterior_seconds") for x in res])[0]))
        print(e, len(records), "runs", "code", sorted({r.get("code") for r in records}))
    out = os.path.join(a.results, "processed")
    write_csv(os.path.join(out, "accuracy.csv"), acc, COLUMNS)
    write_csv(os.path.join(out, "pr_curve.csv"), pr, ["experiment", "layout", "value", "K", "method", "n", "seeds",
                                                      "precision_mean", "precision_sd", "recall_mean", "recall_sd"])
    write_csv(os.path.join(out, "posterior.csv"), post, ["experiment", "layout", "runs", "edge_mass_max", "posterior_seconds_mean"])
    print("processed:", out)


if __name__ == "__main__":
    main()
