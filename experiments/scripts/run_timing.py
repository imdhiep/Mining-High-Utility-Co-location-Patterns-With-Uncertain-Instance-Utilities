"""Running time (Section VI-C): UHUCPM, CQ-HUCPM and PSO-MUCP on the same input files, one program at a time.

For every point (dataset, parameter value) the inputs are prepared untimed:
  instance file (coordinates translated to the origin), posterior of Definition 4 with the bootstrap of V_T
  (cpp/bin/post_counts), gamma = the K-th largest point score, the CQ-HUCPM file (posterior means as utilities), and the
  PSO-MUCP utilities (mean posterior mean of each feature) with its own threshold (the K-th feature-level score).
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
With --counts nothing is timed: every program runs once and its candidate counts are recorded (pruning ratio).
Usage: python run_timing.py CONFIG EXPERIMENT [--reps N] [--counts]"""
import argparse, json, os, re, statistics, subprocess, sys, tempfile, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import layouts, posterior_file, synthetic  # noqa: E402
from harness.programs import Programs  # noqa: E402
from harness.results import RawWriter  # noqa: E402

TOP_START = 0.05
TIE_MARGIN = 1 - 1e-9
SHARE_SEED = 7


def timed(cmd, reps):
    """one warm-up run, then reps timed runs; wall seconds of every timed run and the last output"""
    def once():
        t0 = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True); t = time.perf_counter() - t0
        if p.returncode not in (0, 3):
            raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {p.stderr[-1500:]}")
        return t, p.stdout
    once()
    runs = [once() for _ in range(reps)]
    return [t for t, _ in runs], runs[-1][1]


def candidate_counts(cmd, name):
    """one untimed run: number of patterns whose score is computed (evaluated) and the other counts of the program.
    CQ-HUCPM computes the utility of every pattern that has a row instance, so its count is also the number of
    candidate patterns, the denominator of the pruning ratio (definition (7) of PSO-MUCP)."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode not in (0, 3):
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {p.stderr[-1500:]}")
    if name == "cq":
        m = re.search(r"Number of evaluated patterns: (\d+)", p.stdout)
        k = re.search(r"The number of patterns: (\d+)", p.stdout)
        return dict(evaluated=int(m.group(1)) if m else None, n=int(k.group(1)) if k else None)
    st = last_json(p.stdout)
    return dict(evaluated=st.get("evaluated"), visited=st.get("visited"), n=len(st.get("patterns", [])),
                all_pattern_num=st.get("all_pattern_num"), cliques=st.get("cliques"))


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
        else:                                  # feature-level problem of PSO-MUCP: exact utilities, no variance
            mu = np.array([feature_utility[lab] for lab in labels], float)
            self.Z = float(sum(max(v, 0) for v in mu))
            posterior_file.write(self.post, mu, np.zeros(len(mu)), None, self.Z)
        with open(self.cq_file, "w") as f:
            f.write("Feature,Instance,LocX,LocY,Checkin\n"); c = {}
            for lab, (x, y), v in zip(labels, xy, mu):
                c[lab] = c.get(lab, 0) + 1; f.write(f"{lab},{c[lab]},{float(x)!r},{float(y)!r},{float(v)!r}\n")
        top = self.p.mine(self.data, self.post, TOP_START, d, 0.0, top=K)
        self.gamma = min(top["scores"]) if top["scores"] else 0.0
        self.n_point = len(top["scores"])
        v = feature_utility or {lab: float(mu[labels == lab].mean()) for lab in sorted(set(labels.tolist()))}
        self.pso_utility = v
        fl = np.array([v[lab] for lab in labels], float)
        fl_post = os.path.join(work, "fl.post"); posterior_file.write(fl_post, fl, np.zeros(len(fl)), None, self.Z)
        fl_top = self.p.mine(self.data, fl_post, TOP_START, d, 0.0, top=K)["scores"] if max(v.values()) > 0 else []
        self.pso_threshold = min(fl_top) if fl_top else None

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
                    help="no timing: run every program once and record its candidate counts (results/raw/EXPERIMENT_counts)")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.config)))
    exp = cfg["experiments"][a.experiment]; reps = a.reps or cfg["reps"]
    index_path = os.path.join(root, cfg["layout_index"]); index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    bin_dir = os.path.join(root, cfg["bin_dir"]); cq_binary = os.path.join(root, cfg["cq_binary"])
    programs = Programs(bin_dir, cq_binary)
    writer = RawWriter(os.path.join(root, cfg["results_dir"]), a.experiment + ("_counts" if a.counts else ""), os.path.join(root, ".."))
    for spec in exp["datasets"]:
        spec = dict(spec, vary=exp["vary"])
        for value in exp["values"]:
            labels, xy, counts, exposure, utility = load_dataset(spec, value, index, root)
            d = float(value if exp["vary"] == "d" else spec.get("d", cfg["d"]))
            K = int(value if exp["vary"] == "K" else spec.get("K", cfg["K"]))
            with tempfile.TemporaryDirectory() as work:
                pt = Point(labels, xy, counts, exposure, d, K, work, programs, utility)
                rec = dict(dataset=spec.get("name", spec["kind"]), vary=exp["vary"], value=value, n=len(labels),
                           features=len(set(labels.tolist())), d=d, K=K, gamma=pt.gamma, Z=pt.Z, programs={})
                for name, cmd in pt.commands(bin_dir, cq_binary, cfg["delta"], cfg["z"]).items():
                    if a.counts:
                        rec["programs"][name] = candidate_counts(cmd, name)
                        continue
                    times, out = timed(cmd, reps)
                    stats = last_json(out) if name != "cq" else {}
                    rec["programs"][name] = dict(seconds=times, median=statistics.median(times),
                                                 stats={k: v for k, v in stats.items() if k not in ("patterns", "scores", "probabilities")},
                                                 n=len(stats.get("patterns", [])) if stats else None, raw=out[-800:] if name == "cq" else None)
                writer.write(rec)
                print(a.experiment, rec["dataset"], exp["vary"], value,
                      {k: (v.get("evaluated") if a.counts else round(v["median"], 3)) for k, v in rec["programs"].items()}, flush=True)
    print("raw results:", writer.path)


if __name__ == "__main__":
    main()
