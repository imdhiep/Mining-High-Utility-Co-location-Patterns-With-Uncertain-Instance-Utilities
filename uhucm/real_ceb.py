"""Real data with the compound empirical Bayes model (ceb.py) versus the Gamma-Poisson model of the earlier version.

For each layout: candidates R0 = point-score patterns at the threshold gamma of the K-th largest point score; answer sets
  old-LB  : Gamma-Poisson posterior (moments, median center), z = 2.45, Z from posterior means
  new-LB  : compound EB posterior, V_T with the conditional bootstrap, Z = sum E[u+], z = 1.645
  new-BFDR: Bayes-optimal set with posterior expected FDP <= 0.05 (same posterior)
Reported: sizes, Jaccard overlaps, posterior expected FDP of R0 under the new model, and (Foursquare) the fraction of
patterns whose test-period score (compound EB on test counts, untruncated given the rates) reaches gamma.
usage: python3 real_ceb.py CITY [K ...]
"""
import sys, os, json, math, time
import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import norm
from fastmine import Problem, mine, pattern_atoms
from ceb import ceb_posterior, cboot, bfdr_select
import fsq_loader, paths

city = sys.argv[1]; KS = [int(k) for k in sys.argv[2:]] or [100, 300, 1000]
CSV = {"LasVegas": ("LasVegas_POI 10f.csv", 100.0), "California": ("California_POI 13f.csv", 1300.0),
       "Toronto": ("Toronto_POI 16f.csv", 100.0)}
if city in CSV:
    import pandas as pd
    df = pd.read_csv(os.path.join(os.environ["D6_DIR"], CSV[city][0])); DIST = CSV[city][1]
    x = df.Checkin.to_numpy(float); E = 1.0; feat = df.Feature.astype(str).to_numpy(); XY = list(zip(df.LocX, df.LocY))
    t = int(x.min()); xte = None
else:
    rows, users, NU, info = fsq_loader.load(paths.fsq(city), city); DIST = 50.0
    x = np.array([r["a"] for r in rows], float); E = rows[0]["W"]; feat = np.array([r["feat"] for r in rows])
    XY = [(r["x"], r["y"]) for r in rows]; t = 1; xte = np.array([r["a_test"] for r in rows], float); Ete = info["Wte"]
fs = sorted(set(feat)); F = np.array([fs.index(g) for g in feat]); n = len(x)
t0 = time.time()
# old model: Gamma-Poisson with moment hyperparameters and median center
mo = np.zeros(n); so = np.zeros(n)
for j in range(len(fs)):
    c = x[F == j]; m_, v_ = c.mean(), c.var(ddof=1); A = m_ * m_ / (v_ - m_) if v_ > m_ else 1e6; B = A * E / m_
    mo[F == j] = digamma(A + c) - np.log(B + E); so[F == j] = polygamma(1, A + c)
mo -= np.array([np.median(mo[F == j]) for j in range(len(fs))])[F]
# new model
mu, s2, ep, st = ceb_posterior(x, E, F, t=t); U = cboot(x, E, st, B=20, seed=1)
if xte is not None:
    mte, _, _, _ = ceb_posterior(xte, Ete, F, t=0); Zte = float(np.clip(mte, 0, None).sum())
tpost = time.time() - t0
Pn = Problem(F, XY, mu, s2, DIST, 0.0, feat_names=fs); Pn.Z = float(ep.sum())
Po = Problem(F, XY, mo, so, DIST, 0.0, feat_names=fs)


def topk(P, K, start=0.05):
    m = start
    while True:
        R = mine(P, m)[0]
        if len(R) >= K or m < 1e-9: return R, sorted(R.values(), reverse=True)
        m /= 2


def jac(a, b): return round(len(a & b) / max(1, len(a | b)), 3)


Rn, scn = topk(Pn, max(KS)); Ro, sco = topk(Po, max(KS))
out = dict(city=city, n=n, k=len(fs), t=t, E=round(E, 2), post_s=round(tpost, 1), Z_new=round(Pn.Z, 1),
           Z_plugin=round(float(np.clip(mu, 0, None).sum()), 1), res={})
for K in KS:
    if len(scn) < K or len(sco) < K: continue
    g = scn[K - 1]; R0 = [p for p, v in Rn.items() if v >= g]
    As = [pattern_atoms(Pn, p) for p in R0]; As = [a[a < n] for a in As]
    M = np.array([mu[a].sum() for a in As]); V = np.array([s2[a].sum() + U[:, a].sum(1).var(ddof=1) for a in As])
    lb = set(p for p, m, v in zip(R0, M, V) if m - 1.645 * math.sqrt(v) >= g * Pn.Z)
    sel, p = bfdr_select(M, V, g * Pn.Z, 0.05); bf = set(p_ for p_, s_ in zip(R0, sel) if s_)
    go = sco[K - 1]; R0o = set(p for p, v in Ro.items() if v >= go)
    Ao = {p_: pattern_atoms(Po, p_) for p_ in R0o}
    lbo = set(p_ for p_ in R0o if mo[Ao[p_][Ao[p_] < n]].sum() - 2.45 * math.sqrt(so[Ao[p_][Ao[p_] < n]].sum()) >= go * Po.Z)
    r = dict(gamma=round(g, 5), R0=len(R0), new_lb=len(lb), new_bfdr=len(bf), old_R0=len(R0o), old_lb=len(lbo),
             J_R0_old_new=jac(set(R0), R0o), J_bfdr_oldlb=jac(bf, lbo), J_lb_bfdr=jac(lb, bf),
             pEFDP_R0=round(float((1 - p).mean()), 3), pEFDP_bfdr=round(float((1 - p[sel]).mean()), 3) if sel.any() else None,
             pEFDP_oldlb=round(float(np.mean([1 - p[i] for i, q in enumerate(R0) if q in lbo])), 3) if lbo & set(R0) else None,
             oldlb_in_R0=round(len(lbo & set(R0)) / max(1, len(lbo)), 3),
             size_R0=round(float(np.mean([len(q) for q in R0])), 2), size_bfdr=round(float(np.mean([len(q) for q in bf])), 2) if bf else None)
    if xte is not None:
        hold = lambda S: round(float(np.mean([mte[As[i]].sum() / Zte >= g for i, q in enumerate(R0) if q in S])), 3) if S else None
        r.update(hold_bfdr=hold(bf), hold_rest=hold(set(R0) - bf), hold_lb=hold(lb), hold_R0=hold(set(R0)))
    out["res"][K] = r; print(city, K, json.dumps(r), flush=True)
os.makedirs("out_ceb", exist_ok=True)
json.dump(out, open(f"out_ceb/real_{city}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "res"}))
