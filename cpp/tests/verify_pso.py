"""C++ port of PSO-MUCP (bin/pso_mucp) against the authors' Python code (run through uhucm/ext/run_pso.py, which only
substitutes the input file, the utilities, Min_utility, D and the normalizer). Both get the same normalizer: the total
positive utility of the instances. Random small datasets with utilities of both signs, plus optional real files.
Usage: python verify_pso.py RUNS [CSV UTIL MINUTIL D ...]"""
import json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
PSO_SRC = os.environ.get("PSO_SRC", "/home/claude/yuanshuaikang/pso-mucp/HUCPM/PSO-MUCP.py")
BIN = os.path.join(HERE, "..", "bin", "pso_mucp")


def py_run(csvp, util, mu, d):
    out = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "uhucm", "ext", "run_pso.py"), PSO_SRC, csvp, repr(util),
                          repr(float(mu)), repr(float(d))], capture_output=True, text=True)
    if out.returncode: raise RuntimeError(out.stderr[-1500:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def cpp_run(csvp, util, mu, d, norm):
    s = ",".join(f"{k}:{float(v)!r}" for k, v in util.items())
    out = subprocess.run([BIN, csvp, s, repr(float(mu)), repr(float(d)), "--norm", repr(float(norm))], capture_output=True, text=True)
    if out.returncode: raise RuntimeError(out.stderr[-1500:])
    return json.loads(out.stdout)


def norm_of(csvp, util):
    import csv
    return sum(max(float(util[r["Feature"]]), 0.0) for r in csv.DictReader(open(csvp)))


def check(csvp, util, mu, d, tag):
    nrm = norm_of(csvp, util)
    a = py_run(csvp, util, mu, d); b = cpp_run(csvp, util, mu, d, nrm)
    ok = sorted(a["patterns"]) == b["patterns"]
    print(json.dumps(dict(tag=tag, ok=ok, n_py=a["n"], n_cpp=b["n"], py_s=a["seconds"], cpp_s=round(b["seconds"], 3),
                          only_py=sorted(set(a["patterns"]) - set(b["patterns"]))[:5],
                          only_cpp=sorted(set(b["patterns"]) - set(a["patterns"]))[:5])), flush=True)
    return ok


def main():
    runs = int(sys.argv[1]); rng = np.random.default_rng(7); bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for it in range(runs):
            k = int(rng.integers(3, 8)); n = int(rng.integers(20, 160)); d = float(rng.uniform(1.0, 4.0))
            feats = [chr(65 + j) for j in range(k)]
            F = rng.integers(0, k, n); F[:k] = np.arange(k); XY = rng.uniform(0, 12, (n, 2))
            util = {f: float(np.round(rng.normal(1.0, 2.0), 3)) or 1.0 for f in feats}
            if all(v <= 0 for v in util.values()): util[feats[0]] = 2.0
            csvp = os.path.join(tmp, f"r{it}.csv")
            with open(csvp, "w") as f:
                f.write("Feature,Instance,LocX,LocY,Checkin\n"); cnt = {}
                for j, (x, y) in zip(F.tolist(), XY):
                    cnt[j] = cnt.get(j, 0) + 1; f.write(f"{feats[j]},{cnt[j]},{float(x)!r},{float(y)!r},0\n")
            mu = float(rng.choice([0.02, 0.05, 0.1, 0.2, 0.3]))
            bad += not check(csvp, util, mu, d, f"random{it}")
    args = sys.argv[2:]
    for i in range(0, len(args), 4):
        csvp, util, mu, d = args[i], eval(args[i + 1]), float(args[i + 2]), float(args[i + 3])
        bad += not check(csvp, util, mu, d, os.path.basename(csvp))
    print("MISMATCHES", bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
