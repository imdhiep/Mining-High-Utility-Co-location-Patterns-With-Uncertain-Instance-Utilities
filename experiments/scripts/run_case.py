"""Case study on New York (Section VI-D.2): one pattern kept and one rejected by UHUCPM at K = 300.

Same inputs as the hold-rate experiment (run_hold.py): training posterior of Definition 4 (t = 1, B = 20, seed 0) and
test utilities of the test period; gamma = the 300-th largest training point score; d = 50 m. Pairs of patterns, each chosen by a fixed rule:
  ranking:        the top N of UHUCPM (by p, the posterior probability of reaching gamma) against the top N of CQ-HUCPM
                  (by its total): the CQ-HUCPM pattern outside the UHUCPM top N with the highest CQ-HUCPM score, and the
                  UHUCPM pattern outside the CQ-HUCPM top N with the closest point score;
  set_rule_DELTA: the decision sets: the pattern accepted by its point score (CQ-HUCPM) but outside S_DELTA with the
                  highest point score, and the pattern of S_DELTA with the closest point score (when S_DELTA leaves out
                  any pattern).
For every chosen pattern: features (category names), participating instances with their coordinates, training counts and posterior
means and variances, M, V, the point score, the lower bound at z*, p, and the test score.
Usage: python run_case.py CONFIG [--city New_York] [--K 300] [--top 100]"""
import argparse, csv, json, os, sys, tempfile

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from harness import posterior_file  # noqa: E402
from harness.processed import ranked  # noqa: E402
from harness.programs import Programs  # noqa: E402
from harness.results import RawWriter  # noqa: E402
from run_hold import prepare_city, read_csv  # noqa: E402

TOP_START = 0.05
Z_STAR = 1.645
CQ_MARGIN = 1 - 1e-6        # CQ-HUCPM adds utilities in float32: its total of the K-th pattern can fall just below gamma
DELTAS = [0.01, 0.05, 0.1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--city", default="New_York"); ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()
    cfg = json.load(open(a.config)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.config)))
    programs = Programs(os.path.join(root, cfg["bin_dir"]), os.path.join(root, cfg["cq_binary"]))
    layouts = json.load(open(os.path.join(root, cfg["layout_index"])))
    e = layouts[f"{a.city}_k20"]
    city = dict(source="foursquare", layout=e["file"], exposure=e["exposure"], test_exposure=e["test_exposure"])
    writer = RawWriter(os.path.join(root, cfg["results_dir"]), "D2_case", os.path.join(root, ".."))
    d = cfg["d"]
    with tempfile.TemporaryDirectory() as work:
        data, post, test = prepare_city(city, cfg, root, programs, work)
        P = posterior_file.read(post); mu, var = P["mean"], P["variance"]; Z = P["normalizer"]
        rows = read_csv(os.path.join(root, city["layout"]))
        point = sorted(programs.mine(data, post, TOP_START, d, 0.0, top=a.K)["scores"], reverse=True)
        gamma = point[a.K - 1]
        cq_file = os.path.join(work, "cq.csv")
        with open(cq_file, "w") as f:
            f.write("Feature,Instance,LocX,LocY,Checkin\n")
            for r, v in zip(read_csv(data), mu):                  # coordinates translated to the origin
                f.write(f"{r['Feature']},{r['Instance']},{r['LocX']},{r['LocY']},{float(v)!r}\n")
        cq = programs.cq(cq_file, d, -1e30, Z)["scores"]
        t_of = programs.mine(data, post, gamma, d, 0.0, top_probability=a.top)["score_of"]   # standardized margins, p = Phi(t)
        U = set(ranked(t_of, a.top)); C = set(ranked(cq, a.top))
        pairs = {}
        rejected = max(C - U, key=lambda q: (cq[q], q))
        pairs["ranking"] = (min(U - C, key=lambda q: (abs(cq.get(q, 0.0) - cq[rejected]), q)), rejected)
        # decision pair: point pool (CQ-HUCPM accepts) outside S_delta at each level, highest point score first
        pool = {q for q, v in cq.items() if v >= gamma * CQ_MARGIN}
        for delta in DELTAS:
            S = set(programs.mine(data, post, gamma, d, 0.0, set_level=delta)["patterns"])
            if pool - S:
                r = max(pool - S, key=lambda q: (cq[q], q))
                pairs[f"set_rule_{delta:g}"] = (min(S, key=lambda q: (abs(cq[q] - cq[r]), q)), r)
        Z_test = float(np.clip(test, 0, None).sum())
        test_post = os.path.join(work, "test_values.bin"); posterior_file.write(test_post, test, np.zeros(len(test)), None, Z_test)
        chosen = sorted({q for pr in pairs.values() for q in pr})
        train = programs.pattern_scores(data, post, d, Z_STAR, chosen, work, instances=True)
        margin = [(m - gamma * Z) / np.sqrt(v) if v > 0 else np.inf for m, v in zip(train["M"], train["V"])]
        tst = programs.pattern_scores(data, test_post, d, 0.0, chosen, work)
        label_name = {r["Feature"]: r["category"].replace("___", " / ").replace("_and_", " & ").replace("_", " ") for r in rows}
        out = dict(city=a.city, K=a.K, top=a.top, d=d, gamma=gamma, Z=Z, Z_test=Z_test, z=Z_STAR,
                   rule="ranking: second = highest CQ-HUCPM score in its top N outside the UHUCPM top N, first = UHUCPM top N "
                        "outside the CQ-HUCPM top N with the closest point score; set_rule_DELTA: second = highest point score "
                        "among the patterns with point score >= gamma outside S_DELTA, first = pattern of S_DELTA with the "
                        "closest point score", pairs={k: list(v) for k, v in pairs.items()}, patterns={})
        for i, q in enumerate(chosen):
            role = q
            inst = [int(x) for x in train["members"][i]["instances"]]
            out["patterns"][role] = dict(
                pattern=q, features=[label_name[c] for c in q], point_score=cq[q], margin=float(margin[i]), p=float(norm.cdf(margin[i])),
                M=train["M"][i], V=train["V"][i],
                lower_bound=train["scores"][i], test_score=tst["scores"][i], holds=tst["scores"][i] >= gamma,
                rank_uhucpm=ranked(t_of).index(q) + 1 if q in t_of else None, rank_cq=ranked(cq).index(q) + 1,
                instances=inst, x=[float(rows[j]["LocX"]) for j in inst], y=[float(rows[j]["LocY"]) for j in inst],
                count=[float(rows[j]["a"]) for j in inst], mean=[float(mu[j]) for j in inst], variance=[float(var[j]) for j in inst])
        out["all_x"] = [float(r["LocX"]) for r in rows]; out["all_y"] = [float(r["LocY"]) for r in rows]
        writer.write(out)
        for role, v in out["patterns"].items():
            print([k for k, pr in pairs.items() if role in pr], end=" ")
            print(role, v["pattern"], v["features"], len(v["instances"]), "point", round(v["point_score"], 5), "t", round(v["margin"], 2), "rank", v["rank_uhucpm"], v["rank_cq"],
                  "LB", round(v["lower_bound"], 5), "test", round(v["test_score"], 5), "gamma", round(gamma, 5), flush=True)
    print("raw results:", writer.path)


if __name__ == "__main__":
    main()
