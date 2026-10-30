"""Exact-posterior check of the normal approximation on Foursquare (D4/D5): for every point-utility HUCP at the
threshold of K in {300, 1000}, Pr(U(P) >= gamma_K Z) is estimated by Monte Carlo from the Gamma posteriors of
Definition 4 (2,000 draws of every atom, seed 0), U(P) being the sum of ln(lambda_o) - c_f over the atoms of P.
Reported: agreement of the rule "probability >= 0.95" with UHUCM at z = 1.645 and z = 2.45. Writes out_rev2/mc_<city>.json."""
import sys, json, numpy as np
import fsq_loader, paths
from suite import Suite
from fastmine import mine, pattern_atoms
city = sys.argv[1]; B = 2000
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city)
S = Suite(f"FSQ_{city}", rows, 50.0, "out_rev2", log=lambda s: None)
P0 = S.mk(); R0all, sc = S.topk(P0, 1000)
shape = np.array([r["alpha"] + r["a"] for r in rows]); rate = np.array([r["beta"] + r["W"] for r in rows])
cf = np.array([r["med"] for r in rows])
rng = np.random.default_rng(0); out = {}
for K in (300, 1000):
    m = sc[K - 1]; thr = m * P0.Z; R0 = [p for p, v in R0all.items() if v >= m]
    R16 = set(mine(S.mk(z=1.645), m)[0]); R24 = set(mine(S.mk(z=2.45), m)[0])
    prob = {}
    for p in R0:
        A = pattern_atoms(P0, p); A = A[A < P0.n]
        tot = np.zeros(B)
        for a0 in range(0, len(A), 512):
            a = A[a0:a0 + 512]
            draws = np.log(rng.gamma(shape[a][None, :], 1.0 / rate[a][None, :], size=(B, len(a)))) - cf[a][None, :]
            tot += draws.sum(1)
        prob[p] = float((tot >= thr).mean())
    exact = {p for p, q in prob.items() if q >= 0.95}
    res = dict(R0=len(R0), R_mc95=len(exact), R16=len(R16), R24=len(R24),
               R16_not_mc=len(R16 - exact), mc_not_R16=len(exact - R16), R24_not_mc=len(R24 - exact), mc_not_R24=len(exact - R24),
               min_prob_R24=round(min(prob[p] for p in R24), 4) if R24 else None, min_prob_R16=round(min(prob[p] for p in R16), 4) if R16 else None)
    out[K] = res; print(city, K, res, flush=True)
json.dump(out, open(f"out_rev2/mc_{city}.json", "w"), indent=1)
