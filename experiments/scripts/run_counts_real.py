"""Counts on the real datasets, no timing (Section VI-C): what the local pruning rules miss, and the size of the answer
of UHUCPM under different K and delta.

For every city (20-feature layouts of data/layouts, the same inputs as the running-time points of C.3: posterior of
Definition 4 on the training counts with t = 0, B = 20 bootstrap samples, seed 0; d = 50 m):
  gamma_K = the K-th largest point score, K in Ks;
  local_rule: at every gamma_K and z in Zs, the exact answer of the threshold rule and the answers with each local rule
              of Section IV ("sum": skip P + g when Lambda_z(P) < gamma and the summed mean of g in the cliques of
              P + g is at most 0; "global": when every instance of g has mean at most 0); the patterns missed are kept;
  answer:     the set rule S_delta at every gamma_K and delta in DELTAS, and the threshold rule at z*: number of patterns
              and largest pattern size.
Usage: python run_counts_real.py CONFIG"""
import argparse, json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness import layouts  # noqa: E402
from harness.programs import Programs  # noqa: E402
from harness.results import RawWriter  # noqa: E402

TOP_START = 0.05
CITIES = ["New_York", "Tokyo", "Philadelphia", "Tampa", "New_Orleans"]
KS = [100, 300, 1000]
ZS = [0.0, 1.645]
DELTAS = [0.01, 0.05, 0.1, 0.2]
RULES = ["sum", "global"]


def largest(patterns):
    return max((len(q) for q in patterns), default=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--cities")
    a = ap.parse_args()
    cfg = json.load(open(a.config)); root = os.path.dirname(os.path.dirname(os.path.abspath(a.config)))
    programs = Programs(os.path.join(root, cfg["bin_dir"]), os.path.join(root, cfg["cq_binary"]))
    index = json.load(open(os.path.join(root, cfg["layout_index"])))
    writer = RawWriter(os.path.join(root, cfg["results_dir"]), "C0_counts_real", os.path.join(root, ".."))
    d = cfg["d"]
    for name in (a.cities.split(",") if a.cities else CITIES):
        entry = index[f"{name}_k20"]
        lay = layouts.load(os.path.join(root, entry["file"]), name, entry["exposure"])
        with tempfile.TemporaryDirectory() as work:
            data = os.path.join(work, "data.csv"); xy = lay.xy - lay.xy.min(0); c = {}
            with open(data, "w") as f:
                f.write("Feature,Instance,LocX,LocY,count\n")
                for lab, (x, y), v in zip(lay.labels, xy, lay.counts):
                    c[lab] = c.get(lab, 0) + 1; f.write(f"{lab},{c[lab]},{float(x)!r},{float(y)!r},{float(v)!r}\n")
            post = os.path.join(work, "post.bin")
            programs.posterior(data, "count", lay.exposure, post, truncation=0, samples=cfg.get("bootstrap", 20), seed=0)
            point = sorted(programs.mine(data, post, TOP_START, d, 0.0, top=max(KS))["scores"], reverse=True)
            rec = dict(city=name, n=len(lay.labels), d=d, per_K={})
            for K in KS:
                if len(point) < K:
                    rec["per_K"][K] = dict(skipped=True, point_patterns=len(point)); continue
                gamma = point[K - 1]; k = dict(gamma=gamma, local_rule={}, set_rule={})
                for z in ZS:
                    exact = programs.mine(data, post, gamma, d, z)
                    E = set(exact["patterns"]); entry_z = dict(exact=len(E), evaluated=exact["evaluated"])
                    for rule in RULES:
                        r = programs.mine(data, post, gamma, d, z, rule=rule)
                        R = set(r["patterns"])
                        entry_z[rule] = dict(found=len(R), missed=sorted(E - R), extra=sorted(R - E), evaluated=r["evaluated"])
                    k["local_rule"][str(z)] = entry_z
                    if z > 0:
                        k["threshold_rule"] = dict(z=z, n=len(E), largest=largest(E))
                for delta in DELTAS:
                    S = programs.mine(data, post, gamma, d, 0.0, set_level=delta)["patterns"]
                    k["set_rule"][str(delta)] = dict(n=len(S), largest=largest(S))
                rec["per_K"][K] = k
            writer.write(rec)
            print(name, {K: (v.get("set_rule", {}).get("0.1"), {z: {r: len(e[r]["missed"]) for r in RULES} for z, e in v.get("local_rule", {}).items()})
                         for K, v in rec["per_K"].items()}, flush=True)
    print("raw results:", writer.path)


if __name__ == "__main__":
    main()
