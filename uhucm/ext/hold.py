"""Hold rates on real data for UHUCPM, CQ-HUCPM and PSO-MUCP (Section VI-B, real data).

The training posterior (ratings on Yelp, Definition (count utility) on Foursquare), V_T with B = 20 and the expected
normalizer Z are those of the other real-data experiments. gamma_K is the K-th largest training point score.
  uhucpm  patterns whose lower bound at z* = 1.645 reaches gamma_K
  cq      CQ-HUCPM at z = 0 with the posterior means as instance utilities: the patterns whose point score reaches
          gamma_K (the exact z = 0 answer, which equals the output of the authors' code, see ext/bench.py)
  pso     the authors' PSO-MUCP at z = 0 with one utility per feature (the mean posterior mean of the feature), at the
          same absolute threshold gamma_K * Z; empty when no feature has a positive utility
A pattern holds when its test score, the sum of the test utilities of its participating instances over the total
positive test utility, still reaches gamma_K. Reported per method and K: the number of patterns, the hold rate, and
for the patterns reported by UHUCPM and not by the other method (and the reverse) their hold rates.
Usage: python hold.py DATASET     DATASET in NYC, TKY or a Yelp city with "_" for spaces (env YELP_DIR, REVIEW_DIR)
-> out_hold/hold_<DATASET>.json"""
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
res = dict(dataset=ds, n=len(F), d=d, z=ZS, Z=Z, Zt=Zt, U_fl=Ufl, feature_utility={S.names[j]: v for j, v in fv.items()}, K={})
for K in KS:
    if K > len(sc): continue
    g = sc[K - 1]
    S_cq = {p for p, v in R0all.items() if v >= g}
    S_u = set(mine(Pz, g)[0])
    if Ufl > 0:
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "run_pso.py"), PSO, csvp, util_fl,
                            repr(float(g * Z / Ufl * (1 - 1e-9))), str(d)], capture_output=True, text=True)
        if p.returncode != 0: raise RuntimeError(p.stderr[-2000:])
        S_pso = {tuple(sorted(S.names[ALPHA.index(c)] for c in s)) for s in json.loads(p.stdout.strip().splitlines()[-1])["patterns"]}
    else:
        S_pso = set()
    A = {}

    def te(q):
        if q not in A: A[q] = pattern_atoms(P0, q)
        a = A[q]; return float(np.nansum(S.mut[a][has_t[a]]) / Zt)

    def hr(Ps): return round(float(np.mean([te(q) >= g for q in Ps])), 4) if Ps else None
    row = dict(gamma=float(g), uhucpm=dict(n=len(S_u), hold=hr(S_u)), cq=dict(n=len(S_cq), hold=hr(S_cq)),
               pso=dict(n=len(S_pso), hold=hr(S_pso)), uhucpm_subset_of_cq=S_u <= S_cq)
    for m, Sm in (("cq", S_cq), ("pso", S_pso)):
        row[f"only_uhucpm_vs_{m}"] = dict(n=len(S_u - Sm), hold=hr(S_u - Sm))
        row[f"only_{m}"] = dict(n=len(Sm - S_u), hold=hr(Sm - S_u))
    res["K"][str(K)] = row
    print(ds, K, json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(OUT, f"hold_{ds}.json"), "w"), indent=1)
os.remove(csvp)
print("DONE", flush=True)
