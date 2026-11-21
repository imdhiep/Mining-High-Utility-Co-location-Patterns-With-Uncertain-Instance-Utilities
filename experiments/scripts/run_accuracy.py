"""Accuracy experiments on semi-synthetic data (Section VI-B): every method on the same posterior, judged against the
true patterns.

For one layout, one setting and one seed:
  1. semi-synthetic counts and true utilities (harness/semisyn.py);
  2. posterior of Definition 4 with the bootstrap of V_T (cpp/bin/post_counts);
  3. true patterns T: the K patterns with the highest true score (cpp/bin/uhucpm_mine --top on the true values);
     gamma = the K-th true score;
  4. decision of every method at gamma, with the same posterior means and normalizer Z:
       UHUCPM    the set rule S_delta at the default level delta    (uhucpm_mine --set-level); the threshold rule
                 Lambda_z* >= gamma is also recorded, with the coverage of its lower bound
       CQ-HUCPM  total of the posterior means >= gamma Z          (authors' C++ code, fixed build, total score)
       PSO-MUCP  its K highest feature-level scores               (C++ port of the authors' code; feature utility =
                 mean posterior mean of the feature; its threshold is the K-th feature-level score)
  5. top K of every method by its own score (precision@K): UHUCPM by the posterior probability p of reaching gamma
     (uhucpm_mine --top-probability), CQ-HUCPM by its total, PSO-MUCP by its feature-level score; the first
     RANKING_DEPTH * K patterns of each ranking are kept (precision-recall curves);
  6. coverage of the lower bound of UHUCPM, the set rule S_delta, and (when "extras") the configurations reported in one
     sentence each: per-instance lower bound with CQ-HUCPM, PSO-MUCP with the true feature utilities, prior-median
     center, Gamma-Poisson prior, V_T without its shared term, sum of positive means as normalizer.
An experiment with "data": "rating" uses semi-synthetic star ratings on a Yelp layout (harness/ratingsim.py) and the
rating posterior (4) (cpp/bin/post_ratings) in step 2, with the same methods afterwards.
One JSON line per (layout, setting, seed) goes to results/raw/<experiment>/<start time>.jsonl.

Usage: python run_accuracy.py CONFIG EXPERIMENT [--seeds N] [--layouts LAYOUT,...]"""
import argparse, json, os, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import layouts, metrics, posterior_file, ratingsim, semisyn  # noqa: E402
from harness.programs import Programs  # noqa: E402
from harness.results import RawWriter  # noqa: E402

TOP_START = 0.05            # first threshold of the top-K search (halved until K patterns reach it)
TIE_MARGIN = 1 - 1e-9       # thresholds passed to programs that compare in a different precision
CQ_MARGIN = 1 - 1e-6        # CQ-HUCPM adds utilities in float32: its total of the K-th pattern can fall just below gamma
RANKING_DEPTH = 3           # scores of the first RANKING_DEPTH * K patterns of each ranking are kept (precision-recall curves)


class SeedRun:
    """all methods on one semi-synthetic draw"""

    def __init__(self, programs, layout, setting, draw, work_dir, z, samples, extras):
        self.p, self.layout, self.s, self.g, self.work = programs, layout, setting, draw, work_dir
        self.z, self.samples, self.extras = z, samples, extras
        self.labels = np.array(layout.features)[draw["F"]]
        self.d = setting["d"]
        self.data = self._write_instances("data.csv", "count", draw["counts"])

    # ---- files
    def _write_instances(self, name, column, values):
        """instance file; coordinates translated to the origin (CQ-HUCPM stores them in float32)"""
        xy = self.g["xy"] - self.g["xy"].min(0)
        path = os.path.join(self.work, name); counter = {}
        with open(path, "w") as f:
            f.write(f"Feature,Instance,LocX,LocY,{column}\n")
            for lab, (x, y), v in zip(self.labels, xy, values):
                counter[lab] = counter.get(lab, 0) + 1
                f.write(f"{lab},{counter[lab]},{float(x)!r},{float(y)!r},{float(v)!r}\n")
        return path

    def _posterior(self, name, mean, variance=None, bootstrap=None, normalizer=0.0):
        path = os.path.join(self.work, name)
        posterior_file.write(path, mean, np.zeros(len(mean)) if variance is None else variance, bootstrap, normalizer)
        return path

    # ---- methods
    def feature_level(self, values):
        """utility per feature (mean over its instances) and its instance-wise copy"""
        v = {lab: float(values[self.labels == lab].mean()) for lab in sorted(set(self.labels.tolist()))}
        return v, np.array([v[lab] for lab in self.labels])

    def pso_top(self, values, K, Z):
        """PSO-MUCP at the K-th feature-level score of the given instance values (None when no feature is positive)"""
        v, per_instance = self.feature_level(values)
        if max(v.values()) <= 0:
            return None
        ranking = self.p.mine(self.data, self._posterior("fl.post", per_instance, normalizer=Z), TOP_START, self.d, 0.0,
                              top=RANKING_DEPTH * K)["score_of"]
        if not ranking:
            return None
        scores = sorted(ranking.values(), reverse=True)
        threshold = scores[min(K, len(scores)) - 1]          # the K-th feature-level score
        r = self.p.pso(self.data, v, threshold * TIE_MARGIN, self.d, Z)
        return dict(patterns=set(r["patterns"]), threshold=threshold, seconds=r["seconds"], ranking=ranking)

    def lower_bound_check(self, mined, Z, truth_post):
        """coverage of the lower bound: true total of every reported pattern against Lambda_z * Z"""
        pats = mined["patterns"]
        if not pats:
            return dict(coverage=None, sizes=[], lower=[], true=[])
        true = self.p.pattern_scores(self.data, truth_post, self.d, 0.0, pats, self.work)
        lower = [mined["score_of"][q] * Z for q in pats]
        return dict(coverage=metrics.coverage(lower, true["M"]), sizes=[len(q) for q in pats], lower=lower, true=true["M"])

    def make_posterior(self, post_path):
        """posterior of Definition 4 from the semi-synthetic counts"""
        g = self.g
        return self.p.posterior(self.data, "count", g["exposure"], post_path, truncation=self.s["t"], samples=self.samples, seed=g["seed"])

    def run(self, Ks, deltas):
        t0 = time.time(); g = self.g
        post_path = os.path.join(self.work, "post.bin")
        post = self.make_posterior(post_path)
        Z = post["Z"]; P = posterior_file.read(post_path)
        mu, s2 = P["mean"], P["variance"]
        Z_true = float(np.clip(g["utility"], 0, None).sum())
        truth_post = self._posterior("truth.post", g["utility"], normalizer=Z_true)
        truth = self.p.mine(self.data, truth_post, TOP_START, self.d, 0.0, top=max(Ks))
        true_scores = sorted(truth["scores"], reverse=True)
        cq_all = self.p.cq(self._write_instances("cq.csv", "Checkin", mu), self.d, -1e30, Z)
        out = dict(n=len(mu), Z=Z, Z_true=Z_true, local_edge_mass_max=post.get("local_edge_mass_max"),
                   posterior_seconds=post["seconds"], cq_seconds=cq_all["seconds"], per_K={})
        for K in Ks:
            if len(true_scores) < K:
                out["per_K"][K] = dict(skipped=True, n_true=len(true_scores)); continue
            gamma = true_scores[K - 1]
            T = {q for q, sc in truth["score_of"].items() if sc >= gamma}
            u_threshold = self.p.mine(self.data, post_path, gamma, self.d, self.z)
            u = self.p.mine(self.data, post_path, gamma, self.d, 0.0, set_level=self.s["delta"])
            point = self.p.mine(self.data, post_path, gamma, self.d, 0.0)
            S_u, S_point = set(u["patterns"]), set(point["patterns"])
            S_cq = {q for q, sc in cq_all["scores"].items() if sc >= gamma * CQ_MARGIN}
            ranked_u = self.p.mine(self.data, post_path, gamma, self.d, 0.0, top_probability=RANKING_DEPTH * K)["score_of"]
            top_u = metrics.top_k(ranked_u, K)
            top_cq = metrics.top_k(cq_all["scores"], K)
            ranked_cq = {q: cq_all["scores"][q] for q in metrics.top_k(cq_all["scores"], RANKING_DEPTH * K)}
            pso = self.pso_top(mu, K, Z)
            S_pso = None if pso is None else pso["patterns"]

            def rec(S, top=None):
                if S is None:
                    return None
                r = dict(n=len(S), precision=metrics.precision(S, T), recall=metrics.recall(S, T), patterns=sorted(S))
                if top is not None:
                    r.update(precision_at_k=metrics.precision(top, T), n_top=len(top), top=sorted(top))
                return r

            reported = set().union(S_u, set(u_threshold["patterns"]), S_cq, S_pso or set())
            true_of = {}
            if reported:
                ts = self.p.pattern_scores(self.data, truth_post, self.d, 0.0, sorted(reported), self.work)
                true_of = dict(zip(ts["patterns"], ts["scores"]))
            k_out = dict(gamma=gamma, n_true=len(T), true=sorted(T),
                         uhucpm=rec(S_u, top_u), cq=rec(S_cq, top_cq), pso=rec(S_pso, S_pso),
                         cq_equals_point=S_cq == S_point, point_n=len(S_point),
                         ranking=dict(uhucpm=ranked_u, cq=ranked_cq, **({} if pso is None else dict(pso=pso["ranking"]))),
                         uhucpm_threshold=rec(set(u_threshold["patterns"])),
                         uhucpm_lower_bound=self.lower_bound_check(u_threshold, Z, truth_post),
                         true_score_of_reported=true_of, set_rule={})
            for delta in deltas:
                sd = self.p.mine(self.data, post_path, gamma, self.d, 0.0, set_level=delta)
                k_out["set_rule"][str(delta)] = rec(set(sd["patterns"]))
            if self.extras:
                k_out["extras"] = self.extra_configurations(K, gamma, T, Z, mu, s2, P, truth_post)
            out["per_K"][K] = k_out
        out["seconds"] = round(time.time() - t0, 2)
        return out

    def extra_configurations(self, K, gamma, T, Z, mu, s2, P, truth_post):
        ex = {}
        lb = self.p.cq(self._write_instances("cq_lb.csv", "Checkin", mu - self.z * np.sqrt(s2)), self.d, gamma * TIE_MARGIN, Z)
        S = set(lb["scores"]); ex["cq_instance_lower_bound"] = dict(n=len(S), precision=metrics.precision(S, T), recall=metrics.recall(S, T))
        oracle = self.pso_top(self.g["utility"], K, Z)
        ex["pso_true_feature_utility"] = None if oracle is None else dict(
            n=len(oracle["patterns"]), precision=metrics.precision(oracle["patterns"], T), recall=metrics.recall(oracle["patterns"], T))
        variants = {"prior_median_center": dict(center="median"), "gamma_poisson_prior": dict(prior="gamma")}
        for name, opts in variants.items():
            if name == "gamma_poisson_prior" and self.s["t"] > 0:
                continue
            path = os.path.join(self.work, name + ".post")
            r = self.p.posterior(self.data, "count", self.g["exposure"], path, truncation=self.s["t"], samples=self.samples,
                                 seed=self.g["seed"], **opts)
            ex[name] = self._decision(path, gamma, T, r["Z"], truth_post)
        ex["no_shared_term"] = self._decision(self._posterior("noshared.post", mu, s2, None, Z), gamma, T, Z, truth_post)
        Zpos = float(np.clip(mu, 0, None).sum())
        ex["sum_positive_means_normalizer"] = self._decision(self._posterior("zpos.post", mu, s2, P["bootstrap"], 0.0), gamma, T, Zpos, truth_post)
        return ex

    def _decision(self, post_path, gamma, T, Z, truth_post):
        """S_delta at the default level, and the coverage of the lower bound of the threshold rule, under one posterior"""
        S = set(self.p.mine(self.data, post_path, gamma, self.d, 0.0, set_level=self.s["delta"])["patterns"])
        lbc = self.lower_bound_check(self.p.mine(self.data, post_path, gamma, self.d, self.z), Z, truth_post)
        return dict(n=len(S), precision=metrics.precision(S, T), recall=metrics.recall(S, T), coverage=lbc["coverage"],
                    sizes=lbc["sizes"], lower=lbc["lower"], true=lbc["true"])


class RatingSeedRun(SeedRun):
    """the same methods on the rating posterior (4) of semi-synthetic star ratings (harness/ratingsim.py)"""

    def make_posterior(self, post_path):
        path = os.path.join(self.work, "ratings.csv"); xy = self.g["xy"] - self.g["xy"].min(0); counter = {}
        with open(path, "w") as f:
            f.write("Feature,Instance,LocX,LocY,a,stars\n")
            for lab, (x, y), a, st in zip(self.labels, xy, self.g["counts"], self.g["stars"]):
                counter[lab] = counter.get(lab, 0) + 1
                f.write(f"{lab},{counter[lab]},{float(x)!r},{float(y)!r},{float(a)!r},{';'.join(map(str, st))}\n")
        return self.p.post_ratings(path, post_path, samples=self.samples, seed=self.g["seed"])


def rating_layout(path):
    """Yelp layout with its rating statistics: labels, xy, a, n, S, SS"""
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    col = lambda c: np.array([float(r[c]) for r in rows])
    return np.array([r["Feature"] for r in rows]), np.array([(float(r["LocX"]), float(r["LocY"])) for r in rows]), \
        col("a"), col("n_r"), col("S"), col("SS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("experiment")
    ap.add_argument("--seeds", type=int); ap.add_argument("--layouts")
    ap.add_argument("--skip-done", action="store_true", help="skip (layout, value, seed) already in results/raw/EXPERIMENT")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.config)))   # experiments/
    exp = cfg["experiments"][a.experiment]
    index = json.load(open(os.path.join(root, cfg["layout_index"])))
    programs = Programs(os.path.join(root, cfg["bin_dir"]), os.path.join(root, cfg["cq_binary"]))
    writer = RawWriter(os.path.join(root, cfg["results_dir"]), a.experiment, os.path.join(root, ".."))
    seeds = a.seeds or cfg["seeds"]
    done = writer.completed() if a.skip_done else set()
    names = a.layouts.split(",") if a.layouts else exp["layouts"]
    for value in exp["values"]:
        setting = dict(cfg["defaults"], **exp.get("set", {}), **{exp["vary"]: value})
        for name in names:
            entry = index[f"{name}_k{setting['k']}"]
            layout = layouts.load(os.path.join(root, entry["file"]), name, entry["exposure"])
            rating = exp.get("data") == "rating"
            if rating:
                labels, xy, a_real, n_r, S_r, SS_r = rating_layout(os.path.join(root, entry["file"]))
            else:
                loc, scale = layouts.truth_law(layout)
            for seed in range(seeds):
                if (name, json.dumps(value), seed) in done:
                    continue
                if rating:
                    draw = ratingsim.draw(layout.feature_index(), xy, a_real, n_r, S_r, SS_r, setting, seed)
                else:
                    draw = semisyn.draw(layout, loc, scale, setting, seed)
                with tempfile.TemporaryDirectory() as work:
                    out = (RatingSeedRun if rating else SeedRun)(programs, layout, setting, draw, work, cfg["z"], cfg["bootstrap"],
                                                                 exp.get("extras", False)).run(setting["Ks"], setting["deltas"])
                writer.write(dict(layout=name, vary=exp["vary"], value=value, setting=setting, seed=seed, result=out))
                k0 = out["per_K"][setting["Ks"][0]]
                brief = {m: (k0[m] or {}).get("precision") for m in ("uhucpm", "cq", "pso")} if "gamma" in k0 else k0
                print(a.experiment, name, exp["vary"], value, seed, json.dumps(brief), f"{out['seconds']}s", flush=True)
    print("raw results:", writer.path)


if __name__ == "__main__":
    main()
