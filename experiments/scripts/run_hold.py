"""Hold rates on real data (Section VI-D.1): do the patterns found on the training period still reach the threshold on
the test period?

Per city:
  training posterior: Foursquare (New York, Tokyo) Definition 4 on the training check-ins (t = 1, the venues have at
  least one), Yelp the rating posterior of equation (4); B = 20 bootstrap samples for V_T; expected normalizer Z.
  test utility: Foursquare the posterior mean of the test check-ins (t = 0, test period exposure), Yelp the utility of
  the test ratings; instances without a test value count as 0; Z_test = sum of the positive test utilities.
  gamma_K = the K-th largest training point score.
Methods at gamma_K: UHUCPM (S_delta, default delta), CQ-HUCPM (authors' code, posterior means), PSO-MUCP (C++ port,
its K highest feature-level scores); the CQ-HUCPM decision is checked against the point scores of UHUCPM (z = 0).
Rankings kept for the comparison at the same number of patterns: UHUCPM by p (stored as the margin t, p = Phi(t)),
CQ-HUCPM by its total, PSO-MUCP by its feature-level score (first RANKING_DEPTH * K patterns each).
For every pattern that appears anywhere, its test score (test total / Z_test) is stored, so that hold rates (test score
>= gamma_K) are computed afterwards for any set and any n.
Usage: python run_hold.py CONFIG [--cities NAME,...]"""
import argparse, csv, json, os, subprocess, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import posterior_file  # noqa: E402
from harness.programs import Programs  # noqa: E402
from harness.results import RawWriter  # noqa: E402

TOP_START = 0.05
TIE_MARGIN = 1 - 1e-9
CQ_MARGIN = 1 - 1e-6        # CQ-HUCPM adds utilities in float32: its total of the K-th pattern can fall just below gamma
RANKING_DEPTH = 3


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def translated(rows):
    """coordinates translated to the origin: CQ-HUCPM stores them in float32, which loses centimetres at UTM magnitudes
    and changes neighbour relations at the distance threshold (the other programs read doubles)"""
    x0 = min(float(r["LocX"]) for r in rows); y0 = min(float(r["LocY"]) for r in rows)
    return [(float(r["LocX"]) - x0, float(r["LocY"]) - y0) for r in rows]


def prepare_city(city, cfg, root, programs, work):
    """instance file (coordinates translated to the origin), training posterior file and test utilities of one city"""
    data = os.path.join(work, "data.csv")
    if city["source"] == "foursquare":
        rows = read_csv(os.path.join(root, city["layout"]))
        with open(data, "w") as f:
            f.write("Feature,Instance,LocX,LocY,a,a_test\n")
            for r, (x, y) in zip(rows, translated(rows)):
                f.write(f"{r['Feature']},{r['Instance']},{x!r},{y!r},{r['a']},{r['a_test']}\n")
        post = os.path.join(work, "post.bin")
        programs.posterior(data, "a", city["exposure"], post, truncation=1, samples=cfg["bootstrap"], seed=0)
        test_post = os.path.join(work, "test.bin")
        programs.posterior(data, "a_test", city["test_exposure"], test_post, truncation=0, samples=0, seed=0)
        test = posterior_file.read(test_post)["mean"]
    else:
        rows = read_csv(os.path.join(root, city["csv"]))
        with open(data, "w") as f:
            f.write("Feature,Instance,LocX,LocY\n")
            for r, (x, y) in zip(rows, translated(rows)):
                f.write(f"{r['Feature']},{r['Instance']},{x!r},{y!r}\n")
        post = os.path.join(root, city["post"])
        test = np.array([float(r["mu_test"]) for r in rows])
    return data, post, np.nan_to_num(test, nan=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--cities")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.config)))
    programs = Programs(os.path.join(root, cfg["bin_dir"]), os.path.join(root, cfg["cq_binary"]))
    writer = RawWriter(os.path.join(root, cfg["results_dir"]), "D1_hold", os.path.join(root, ".."))
    index = json.load(open(os.path.join(root, cfg["hold_index"])))
    names = a.cities.split(",") if a.cities else cfg["hold_cities"]
    d, delta = cfg["d"], cfg["delta"]
    for name in names:
        city = index[name]
        with tempfile.TemporaryDirectory() as work:
            data, post, test = prepare_city(city, cfg, root, programs, work)
            P = posterior_file.read(post); mu = P["mean"]; Z = P["normalizer"] if P["normalizer"] > 0 else float(np.clip(mu, 0, None).sum())
            Z_test = float(np.clip(test, 0, None).sum())
            if Z_test <= 0:                    # no positive test utility (no test ratings or check-ins): no hold rate
                writer.write(dict(city=name, n=len(test), skipped="no positive test utility")); continue
            test_post = os.path.join(work, "test_values.bin"); posterior_file.write(test_post, test, np.zeros(len(test)), None, Z_test)
            labels = np.array([r["Feature"] for r in read_csv(data)])
            cq_file = os.path.join(work, "cq.csv")
            with open(cq_file, "w") as f, open(data) as g:
                rows = list(csv.DictReader(g)); f.write("Feature,Instance,LocX,LocY,Checkin\n")
                for r, v in zip(rows, mu):
                    f.write(f"{r['Feature']},{r['Instance']},{r['LocX']},{r['LocY']},{float(v)!r}\n")
            cq_all = programs.cq(cq_file, d, -1e30, Z)["scores"]
            point = programs.mine(data, post, TOP_START, d, 0.0, top=max(cfg["Ks"]))
            point_scores = sorted(point["scores"], reverse=True)
            fv = {lab: float(mu[labels == lab].mean()) for lab in sorted(set(labels.tolist()))}
            fl_post = os.path.join(work, "fl.bin")
            posterior_file.write(fl_post, np.array([fv[lab] for lab in labels]), np.zeros(len(labels)), None, Z)
            rec = dict(city=name, n=len(labels), d=d, delta=delta, Z=Z, Z_test=Z_test, per_K={})
            for K in cfg["Ks"]:
                if len(point_scores) < K:
                    rec["per_K"][K] = dict(skipped=True); continue
                gamma = point_scores[K - 1]
                S_u = set(programs.mine(data, post, gamma, d, 0.0, set_level=delta)["patterns"])
                S_cq = {q for q, s in cq_all.items() if s >= gamma * CQ_MARGIN}
                S_point = set(programs.mine(data, post, gamma, d, 0.0)["patterns"])   # same decision as CQ-HUCPM (check)
                rank_u = programs.mine(data, post, gamma, d, 0.0, top_probability=RANKING_DEPTH * K)["score_of"]
                rank_cq = dict(sorted(cq_all.items(), key=lambda kv: -kv[1])[:RANKING_DEPTH * K])
                S_pso, rank_pso = None, {}
                if max(fv.values()) > 0:
                    fl = programs.mine(data, fl_post, TOP_START, d, 0.0, top=RANKING_DEPTH * K)["score_of"]
                    rank_pso = fl
                    kth = sorted(fl.values(), reverse=True)[min(K, len(fl)) - 1] if fl else None
                    if kth is not None:
                        S_pso = set(programs.pso(data, fv, kth * TIE_MARGIN, d, Z)["patterns"])
                every = sorted(set().union(S_u, S_cq, S_pso or set(), rank_u, rank_cq, rank_pso))
                ts = programs.pattern_scores(data, test_post, d, 0.0, every, work) if every else dict(patterns=[], scores=[])
                rec["per_K"][K] = dict(gamma=gamma, uhucpm=sorted(S_u), cq=sorted(S_cq), pso=None if S_pso is None else sorted(S_pso),
                                       cq_equals_point=S_cq == S_point,
                                       ranking=dict(uhucpm=rank_u, cq=rank_cq, pso=rank_pso),
                                       test_score=dict(zip(ts["patterns"], ts["scores"])))
            writer.write(rec)
            k0 = rec["per_K"][cfg["Ks"][0]]
            if "gamma" in k0:
                hold = lambda S: (sum(k0["test_score"][q] >= k0["gamma"] for q in S) / len(S)) if S else None
                print(name, {m: (len(k0[m]) if k0[m] is not None else None, hold(k0[m] or [])) for m in ("uhucpm", "cq", "pso")}, flush=True)
    print("raw results:", writer.path)


if __name__ == "__main__":
    main()
