"""Running time (Section VI-C): UHUCPM, CQ-HUCPM and PSO-MUCP on the same input files, one program at a time.

For every point (dataset, parameter value) the inputs are prepared untimed:
  instance file (coordinates translated to the origin), posterior of Definition 4 with the bootstrap of V_T
  (cpp/bin/post_counts), gamma = the K-th largest point score, the CQ-HUCPM file (posterior means as utilities), and the
  PSO-MUCP utilities (mean posterior mean of each feature) with its own threshold (the K-th feature-level score).
  PSO-MUCP is not run when no feature-level pattern has a positive score (e.g. every feature has a negative mean
  utility): it reports nothing; the record keeps "pso_skipped".
Programs timed, every one as a separate process from start to exit (reading, neighbour search, cliques and mining all
count): one warm-up run, then REPS timed runs; the median is reported.
  uhucpm            the answer of UHUCPM: the set rule S_delta at gamma (uhucpm_mine --set-level)
  uhucpm_threshold  the threshold rule Lambda_z* >= gamma (recorded for the text)
  cq                the authors' CQ-HUCPM (fixed build) with the total-utility score at gamma
  pso               the C++ port of the authors' PSO-MUCP at its threshold
Each record also keeps the programs' own statistics (cliques, evaluated candidates, peak memory).

Datasets: "synthetic" (harness/synthetic.py, n and k given), a layout of data/layouts/index.json (New_York, Tampa, ...)
with an optional share of its instances, or a dataset released with PSO-MUCP (feature-level problem: the authors'
utilities, every method on the same feature utilities).
With --counts nothing is timed: every program runs once and its candidate counts and peak memory are recorded.
Peak memory is the program's own VmHWM (one process image; see cpp/src/common/process_stats.cpp), so it does not
include the memory of this harness.
A point whose inputs cannot be prepared (the top-K search for gamma fails) is recorded with "failed"; a program that
fails or exceeds --run-timeout seconds in one run is recorded with "did_not_finish" and the other programs still run.
Usage: python run_timing.py CONFIG EXPERIMENT [--reps N] [--counts] [--datasets NAME ...] [--values V ...]
                            [--run-timeout SEC]"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import layouts, posterior_file, synthetic  # noqa: E402
from harness.programs import ProgramError, Programs  # noqa: E402
from harness.results import RawWriter  # noqa: E402

TOP_START = 0.05
TIE_MARGIN = 1 - 1e-9
CQ_MARGIN = 1 - 1e-6
SHARE_SEED = 7


class DidNotFinish(Exception):
    pass


def run_once(cmd, timeout=None):
    """(wall seconds, stdout) of one run; DidNotFinish when the program fails (e.g. killed for memory) or times out"""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DidNotFinish(f"{os.path.basename(cmd[0])} exceeded {timeout} s")
    t = time.perf_counter() - t0
    if p.returncode not in (0, 3):
        raise DidNotFinish(f"{os.path.basename(cmd[0])} failed ({p.returncode}): {p.stderr[-1500:]}")
    return t, p.stdout


def timed(cmd, reps, timeout=None):
    """one warm-up run, then reps timed runs; wall seconds of every timed run and the last output"""
    def once():
        return run_once(cmd, timeout)
    once()
    runs = [once() for _ in range(reps)]
    return [t for t, _ in runs], runs[-1][1]


def cq_memory(out):
    m = re.search(r"Peack memory usage\s*:\s*(\d+)", out or "")
    return int(m.group(1)) if m else None


def candidate_counts(cmd, name, timeout=None):
    """one untimed run: number of patterns whose score is computed (evaluated), the other counts of the program and its
    peak memory. CQ-HUCPM computes the utility of every pattern that has a row instance, so its count is also the
    number of candidate patterns, the denominator of the pruning ratio (definition (7) of PSO-MUCP)."""
    seconds, out = run_once(cmd, timeout)
    if name == "cq":
        m = re.search(r"Number of evaluated patterns: (\d+)", out)
        k = re.search(r"The number of patterns: (\d+)", out)
        return dict(evaluated=int(m.group(1)) if m else None, n=int(k.group(1)) if k else None, max_rss_kb=cq_memory(out),
                    seconds=seconds)
    st = last_json(out)
    return dict(evaluated=st.get("evaluated"), visited=st.get("visited"), n=len(st.get("patterns", [])),
                all_pattern_num=st.get("all_pattern_num"), cliques=st.get("cliques"), max_rss_kb=st.get("max_rss_kb"),
                seconds=seconds)


def last_json(text):
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {}


class Point:
    """inputs of one timing point"""

    def __init__(self, labels, xy, counts, exposure, d, K, work, programs, feature_utility=None):
        self.p, self.work, self.d, self.K = programs, work, d, K
        xy = xy - xy.min(0)
        self.data = os.path.join(work, "data.csv"); self.cq_file = os.path.join(work, "cq.csv")
        self.post = os.path.join(work, "post.bin")
        with open(self.data, "w") as f:
            f.write("Feature,Instance,LocX,LocY,count\n"); c = {}
            for lab, (x, y), v in zip(labels, xy, counts):
                c[lab] = c.get(lab, 0) + 1; f.write(f"{lab},{c[lab]},{float(x)!r},{float(y)!r},{float(v)!r}\n")
        if feature_utility is None:            # posterior of the counts
            r = self.p.posterior(self.data, "count", exposure, self.post, samples=20, seed=0)
            P = posterior_file.read(self.post); mu = P["mean"]; self.Z = r["Z"]
            bad = sorted(set(labels[~np.isfinite(mu)].tolist()))
            if bad:
                raise ValueError(f"posterior mean not finite for features {bad}")
        else:                                  # feature-level problem of PSO-MUCP: exact utilities, no variance
            mu = np.array([feature_utility[lab] for lab in labels], float)
            self.Z = float(sum(max(v, 0) for v in mu))
            posterior_file.write(self.post, mu, np.zeros(len(mu)), None, self.Z)
        with open(self.cq_file, "w") as f:
            f.write("Feature,Instance,LocX,LocY,Checkin\n"); c = {}
            for lab, (x, y), v in zip(labels, xy, mu):
                c[lab] = c.get(lab, 0) + 1; f.write(f"{lab},{c[lab]},{float(x)!r},{float(y)!r},{float(v)!r}\n")
        self.gamma, self.gamma_source = self.kth(self.post, self.cq_file, K)
        v = feature_utility or {lab: float(mu[labels == lab].mean()) for lab in sorted(set(labels.tolist()))}
        self.pso_utility = v
        fl = np.array([v[lab] for lab in labels], float)
        fl_post = os.path.join(work, "fl.post"); posterior_file.write(fl_post, fl, np.zeros(len(fl)), None, self.Z)
        fl_cq = os.path.join(work, "fl_cq.csv")
        with open(fl_cq, "w") as f:
            f.write("Feature,Instance,LocX,LocY,Checkin\n"); c = {}
            for lab, (x, y), u in zip(labels, xy, fl):
                c[lab] = c.get(lab, 0) + 1; f.write(f"{lab},{c[lab]},{float(x)!r},{float(y)!r},{float(u)!r}\n")
        if feature_utility is not None:        # the feature-level problem is the problem itself: same threshold
            fl_top = [self.gamma]
        elif max(v.values()) > 0:
            kth, _ = self.kth(fl_post, fl_cq, K)
            fl_top = [kth] if kth > 0 else []
        else:
            fl_top = []
        self.pso_threshold = min(fl_top) if fl_top else None
        self.pso_skipped = None if fl_top else dict(largest_feature_utility=max(v.values()), feature_level_patterns=len(fl_top))

    def kth(self, post, cq_file, K):
        """the K-th largest score of the patterns (gamma): the top-K search of uhucpm_mine; when that search fails (it
        keeps every candidate of a lower threshold and can run out of memory), the K-th total of CQ-HUCPM, which scores
        every candidate, lowered by CQ_MARGIN since CQ-HUCPM adds the utilities in float32"""
        try:
            top = self.p.mine(self.data, post, TOP_START, self.d, 0.0, top=K)
            return (sorted(top["scores"], reverse=True)[min(K, len(top["scores"])) - 1] if top["scores"] else 0.0), "uhucpm_top"
        except ProgramError:
            scores = sorted(self.p.cq(cq_file, self.d, -1e30, self.Z)["scores"].values(), reverse=True)
            return (scores[min(K, len(scores)) - 1] * CQ_MARGIN if scores else 0.0), "cq"

    def commands(self, bin_dir, cq_binary, delta, z):
        spec = ",".join(f"{k}:{float(u)!r}" for k, u in sorted(self.pso_utility.items()))
        cmds = dict(
            uhucpm=[os.path.join(bin_dir, "uhucpm_mine"), self.data, self.post, repr(self.gamma), repr(self.d), "0", "--set-level", repr(delta)],
            uhucpm_threshold=[os.path.join(bin_dir, "uhucpm_mine"), self.data, self.post, repr(self.gamma), repr(self.d), repr(z)],
            cq=[cq_binary, self.cq_file, repr(self.d), repr(self.gamma * TIE_MARGIN), "--score", "total", "--norm", repr(self.Z)])
        if self.pso_threshold is not None:
            cmds["pso"] = [os.path.join(bin_dir, "pso_mucp"), self.data, spec, repr(self.pso_threshold * TIE_MARGIN), repr(self.d),
                           "--norm", repr(self.Z)]
        return cmds


def load_dataset(spec, value, index, root):
    """labels, xy, counts, exposure (and the PSO utilities for a PSO-MUCP dataset)"""
    kind = spec["kind"]
    if kind == "synthetic":
        n = int(value if spec["vary"] == "n" else spec.get("n", 50000)); k = int(value if spec["vary"] == "k" else spec.get("k", 20))
        g = synthetic.generate(n, k=k, seed=7, d=spec["d"])
        return g["labels"], g["xy"], g["counts"], g["exposure"], None
    if kind == "layout":
        k = int(value if spec["vary"] == "k" else spec.get("k", 20))
        entry = index[f"{spec['name']}_k{k}"]
        lay = layouts.load(os.path.join(root, entry["file"]), spec["name"], entry["exposure"])
        labels, xy, counts = lay.labels, lay.xy, lay.counts
        share = float(value if spec["vary"] == "frac" else spec.get("frac", 1.0))
        if share < 1:
            keep = np.sort(np.random.default_rng(SHARE_SEED).choice(len(labels), int(round(share * len(labels))), replace=False))
            labels, xy, counts = labels[keep], xy[keep], counts[keep]
        return labels, xy, counts, lay.exposure, None
    if kind == "psomucp":
        import csv
        with open(os.path.join(root, spec["file"]), newline="") as f:
            rows = list(csv.DictReader(f))
        labels = np.array([r["Feature"] for r in rows]); xy = np.array([(float(r["LocX"]), float(r["LocY"])) for r in rows])
        counts = np.array([float(r["Checkin"] or 0) for r in rows])
        return labels, xy, counts, 1.0, spec["utility"]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("experiment"); ap.add_argument("--reps", type=int)
    ap.add_argument("--counts", action="store_true",
                    help="no timing: run every program once and record its candidate counts and peak memory "
                         "(results/raw/EXPERIMENT_counts)")
    ap.add_argument("--datasets", nargs="*", help="only these datasets of the experiment")
    ap.add_argument("--values", nargs="*", type=float, help="only these parameter values")
    ap.add_argument("--run-timeout", type=float, help="seconds allowed for one run of one program")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.config)))
    exp = cfg["experiments"][a.experiment]; reps = a.reps or cfg["reps"]
    index_path = os.path.join(root, cfg["layout_index"]); index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    bin_dir = os.path.join(root, cfg["bin_dir"]); cq_binary = os.path.join(root, cfg["cq_binary"])
    programs = Programs(bin_dir, cq_binary, a.run_timeout)   # the preparation runs have the same limit
    writer = RawWriter(os.path.join(root, cfg["results_dir"]), a.experiment + ("_counts" if a.counts else ""), os.path.join(root, ".."))
    for spec in exp["datasets"]:
        spec = dict(spec, vary=exp["vary"])
        if a.datasets and spec.get("name", spec["kind"]) not in a.datasets:
            continue
        for value in exp["values"]:
            if a.values and float(value) not in a.values:
                continue
            labels, xy, counts, exposure, utility = load_dataset(spec, value, index, root)
            d = float(value if exp["vary"] == "d" else spec.get("d", cfg["d"]))
            K = int(value if exp["vary"] == "K" else spec.get("K", cfg["K"]))
            rec = dict(dataset=spec.get("name", spec["kind"]), vary=exp["vary"], value=value, n=len(labels),
                       features=len(set(labels.tolist())), d=d, K=K, programs={})
            with tempfile.TemporaryDirectory() as work:
                try:
                    pt = Point(labels, xy, counts, exposure, d, K, work, programs, utility)
                except Exception as e:         # e.g. the top-K search for gamma is killed for memory
                    rec["failed"] = f"preparation: {type(e).__name__}: {str(e)[-800:]}"
                    writer.write(rec); print(a.experiment, rec["dataset"], exp["vary"], value, "failed:", rec["failed"][:200], flush=True)
                    continue
                rec.update(gamma=pt.gamma, gamma_source=pt.gamma_source, Z=pt.Z, pso_skipped=pt.pso_skipped)
                for name, cmd in pt.commands(bin_dir, cq_binary, cfg["delta"], cfg["z"]).items():
                    try:
                        if a.counts:
                            rec["programs"][name] = candidate_counts(cmd, name, a.run_timeout)
                            continue
                        times, out = timed(cmd, reps, a.run_timeout)
                    except DidNotFinish as e:
                        rec["programs"][name] = dict(did_not_finish=str(e)[-800:]); continue
                    stats = last_json(out) if name != "cq" else dict(max_rss_kb=cq_memory(out))
                    rec["programs"][name] = dict(seconds=times, median=statistics.median(times),
                                                 stats={k: v for k, v in stats.items() if k not in ("patterns", "scores", "probabilities")},
                                                 n=len(stats.get("patterns", [])) if name != "cq" else None, raw=out[-800:] if name == "cq" else None)
                writer.write(rec)
                print(a.experiment, rec["dataset"], exp["vary"], value,
                      {k: ("did not finish" if "did_not_finish" in v else v.get("evaluated") if a.counts else round(v["median"], 3))
                       for k, v in rec["programs"].items()}, "pso skipped" if pt.pso_skipped else "", flush=True)
    print("raw results:", writer.path)


if __name__ == "__main__":
    main()
