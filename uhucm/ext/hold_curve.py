"""Hold rates at matched answer sizes (root cause of PSO-MUCP's higher hold rate on some datasets).

The hold-rate table compares answer sets of different sizes. Here every method ranks the candidate patterns by its own
score and the top n patterns are compared for the same n:
  uhucpm  lower-bound score at z* = 1.645 (only patterns with a positive score can be ranked)
  cq      point score (z = 0), the score of CQ-HUCPM
  pso     exact feature-level score of PSO-MUCP (one utility per feature, the mean posterior mean of the feature);
          the authors' code returns a subset of the exact answer at each threshold
A pattern holds when its test score reaches gamma_K, the K-th largest training point score, for K = 100, 300, 1000.
Usage: python hold_curve.py DATASET  -> out_hold/holdcurve_<DATASET>.json"""
import sys, os, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import paths
from suite import Suite
from fastmine import mine, pattern_atoms

ZS = 1.645; KS = (100, 300, 1000); d = 50.0
PSO = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "PSO-MUCP.py")
ALPHA = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
OUT = "out_hold"; os.makedirs(OUT, exist_ok=True)
ds = sys.argv[1]
if ds in ("NYC", "TKY"):
    import fsq_loader
    rows, users, NU, info = fsq_loader.load(paths.fsq(ds), ds, model="ceb")
    S = Suite(f"H_{ds}", rows, d, OUT, log=lambda s: None, UB=info.pop("UB"), Zhat=info.pop("Zhat"))
else:
    import yelp_full
    city = ds.replace("_", " ")
    rows = yelp_full.build(os.environ["YELP_DIR"], [city], split="2019-01-01", test_end="2020-01-01", log=lambda s: None,
                           review_dir=os.environ.get("REVIEW_DIR"))[city]["rows"]
    S = Suite(f"H_{ds}", rows, d, OUT, log=lambda s: None, UB=yelp_full.vt_boot(rows, B=20, seed=0),
              Zhat=yelp_full.expected_pos([r["mu"] for r in rows], [r["s2"] for r in rows]))
P0 = S.mk(); Pz = S.mk(z=ZS); Z = P0.Z
R0all, sc = S.topk(P0, max(KS))
F = np.array(S.F); fv = {j: float(S.mu[F == j].mean()) for j in range(len(S.names))}
Ufl = float(sum(max(fv[j], 0.0) for j in F))
csvp = os.path.abspath(os.path.join(OUT, f"{ds}.csv")); cnt = {}
with open(csvp, "w", encoding="utf-8") as f:
    f.write("Feature,Instance,LocX,LocY,Checkin\n")
    for j, (x, y) in zip(S.F, S.XY):
        cnt[j] = cnt.get(j, 0) + 1; f.write(f"{ALPHA[j]},{cnt[j]},{x:.4f},{y:.4f},0\n")
util_fl = repr({ALPHA[j]: v for j, v in fv.items()})
has_t = ~np.isnan(S.mut); Zt = float(np.clip(S.mut[has_t], 0, None).sum())

NGRID = (25, 50, 100, 200, 300, 500, 1000)


def ranked(P, N, start=0.05):
    m = start
    while True:
        R = mine(P, m)[0]
        if len(R) >= N or m < 1e-9: return sorted(R.items(), key=lambda kv: -kv[1])[:N]
        m /= 2


import functools
from fastmine import Problem
mu_fl = np.array([fv[j] for j in F])
Pf = Problem.__new__(Problem); Pf.__dict__.update(P0.__dict__)
Pf.amu = mu_fl; Pf.as2 = np.zeros(len(F)); Pf.z = 0.0; Pf.UB = None; Pf.Z = Z
posi = [i for i in range(len(F)) if mu_fl[i] > 0]
Pf.rank = -np.ones(len(F), dtype=np.int64); Pf.rank[posi] = np.arange(len(posi)); Pf.smu = mu_fl[posi]; Pf.ss2 = np.zeros(len(posi))
NMAX = max(NGRID)
rank = {"uhucpm": ranked(Pz, NMAX), "cq": ranked(P0, NMAX), "pso": ranked(Pf, NMAX) if posi else []}
A = {}


def te(q):
    if q not in A: A[q] = pattern_atoms(P0, q)
    a = A[q]; return float(np.nansum(S.mut[a][has_t[a]]) / Zt)


res = dict(dataset=ds, n=len(F), available={m: len(r) for m, r in rank.items()}, K={})
for K in KS:
    if K > len(sc): continue
    g = sc[K - 1]; row = {}
    for n in sorted(set(NGRID) | {K}):
        row[n] = {m: (round(float(np.mean([te(p) >= g for p, _ in r[:n]])), 4) if len(r) >= n else None) for m, r in rank.items()}
    res["K"][K] = dict(gamma=float(g), hold_at_n=row)
    print(ds, K, json.dumps(row), flush=True)
json.dump(res, open(os.path.join(OUT, f"holdcurve_{ds}.json"), "w"), indent=1)
os.remove(csvp)
print("DONE", flush=True)
