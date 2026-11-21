"""C++ miner (bin/uhucpm_mine) against brute force and against the Python miner (uhucm/fastmine.py) on random data.

Each run draws instances, posterior means of both signs, variances and (sometimes) a bootstrap matrix, picks z and a
threshold, and checks that the C++ answer equals (i) brute force over all feature subsets, scored on the participating
instances read from the maximal cliques, and (ii) fastmine.mine; scores must agree to 1e-9.
Usage: python verify_miner.py [RUNS]"""
import itertools, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "uhucm"))
import fastmine  # noqa: E402

BIN = os.path.join(HERE, "..", "bin", "uhucpm_mine")
NAMES = [chr(65 + i) for i in range(26)]


def write_inputs(tmp, feat, xy, mu, s2, U, Z):
    csvp = os.path.join(tmp, "d.csv"); post = os.path.join(tmp, "p.bin")
    with open(csvp, "w") as f:
        f.write("Feature,Instance,LocX,LocY,Checkin\n")
        cnt = {}
        for j, (x, y) in zip(feat, xy):
            cnt[j] = cnt.get(j, 0) + 1; f.write(f"{NAMES[j]},{cnt[j]},{float(x)!r},{float(y)!r},0\n")
    B = 0 if U is None else U.shape[0]
    with open(post, "wb") as f:
        np.array([len(mu), B], np.int64).tofile(f); np.array([Z], float).tofile(f)
        np.asarray(mu, float).tofile(f); np.asarray(s2, float).tofile(f)
        if B: np.asarray(U, float).tofile(f)
    return csvp, post


def run_cpp(csvp, post, gamma, d, z, bound="two", rule=None):
    cmd = [BIN, csvp, post, repr(float(gamma)), repr(float(d)), repr(float(z)), "--bound", bound, "--scores"]
    if rule: cmd += ["--rule", rule]
    r = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    return dict(zip(r["patterns"], r["scores"])), r


def main(runs):
    rng = np.random.default_rng(12345); bad = 0; nonempty = 0
    with tempfile.TemporaryDirectory() as tmp:
        for it in range(runs):
            n = int(rng.integers(8, 70)); k = int(rng.integers(3, 8)); d = float(rng.uniform(2.0, 6.0))
            feat = rng.integers(0, k, n); feat[:k] = np.arange(k)          # every feature present, names sorted = indices
            xy = rng.uniform(0, 20, (n, 2))
            mu = rng.normal(0.2, 1.0, n) * rng.choice([1.0, 0.0], n, p=[0.9, 0.1])
            s2 = rng.exponential(0.5, n) * rng.choice([1.0, 0.0], n, p=[0.9, 0.1])
            U = rng.normal(0, 0.3, (int(rng.choice([5, 20])), n)) + mu if rng.random() < 0.5 else None
            Z = float(rng.uniform(3, 15)) if rng.random() < 0.5 else 0.0
            z = float(rng.choice([0.0, 1.0, 1.645, 3.0]))
            P = fastmine.Problem(list(feat), [tuple(p) for p in xy], mu, s2, d, z, feat_names=NAMES[:k])
            if U is not None: P.set_boot(U, Z if Z > 0 else None)
            elif Z > 0: P.Z = Z
            # brute force over all feature subsets
            scores = {}
            hasf = P.Mem >= 0
            for r in range(2, k + 1):
                for S in itertools.combinations(range(k), r):
                    L = np.nonzero(hasf[:, list(S)].all(1))[0]
                    if len(L): scores["".join(NAMES[f] for f in S)] = P.score(P.atoms(L, S))
            vals = sorted(scores.values(), reverse=True)
            gamma = vals[int(rng.integers(0, len(vals)))] if vals and rng.random() < 0.8 else float(rng.uniform(0.0, 0.5))
            brute = {p for p, v in scores.items() if v >= gamma - 1e-12}
            py = {"".join(sorted(p)) for p in fastmine.mine(P, gamma)[0]}
            csvp, post = write_inputs(tmp, feat, xy, mu, s2, U, Z if Z > 0 else 0.0)
            for bound in ("two", "exact", "mean", "none"):
                cpp, _ = run_cpp(csvp, post, gamma, d, z, bound)
                ok = set(cpp) == brute == py and all(abs(cpp[p] - scores[p]) <= 1e-9 for p in cpp)
                if not ok:
                    bad += 1
                    print("MISMATCH", it, bound, dict(n=n, k=k, z=z, gamma=gamma, only_cpp=sorted(set(cpp) - brute),
                                                     only_brute=sorted(brute - set(cpp)), py_equal=py == brute))
            nonempty += bool(brute)
    print(json.dumps(dict(runs=runs, checks=4 * runs, mismatches=bad, nonempty=nonempty)))
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 300) else 0)
