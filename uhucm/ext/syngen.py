"""Synthetic check-in data for the scalability experiment (Section VI-C).

Locations follow the generator of Huang et al. (2004), as used by PSO-MUCP: N core patterns of 2 to 5 features
are drawn from k features, and each row instance of a core pattern places one instance of each of its features within
d/2 of a random point, so that the row is a clique. A share `noise` of the instances is placed uniformly with a random
feature. The square area grows with n so that the density of the noise instances, and with it the mean number of
neighbours, stays the same for every n.
Check-ins follow Definition 4: for each feature the log-rates per week are normal with a feature mean drawn uniformly
in [ln 0.02, ln 0.2] and standard deviation `sd`, and the counts over E weeks are Poisson. The posteriors, V_T and the
expected normalizer are computed with ceb.py exactly as on the Foursquare data (t = 0).
Usage (as a library): G = colgen(n, k=20, seed=7) -> dict(feat, xy, cnt, mu, s2, UB, Z, d, E)"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from ceb import ceb_posterior, cboot


def colgen(n, k=20, seed=7, d=50.0, N=20, noise=0.5, sd=1.0, E=34.3, B=20, smin=2, smax=5):
    rng = np.random.default_rng(seed)
    side = math.sqrt(n * math.pi * d * d)                    # about one noise instance per disc of radius d
    pats = [rng.choice(k, size=int(rng.integers(smin, smax + 1)), replace=False) for _ in range(N)]
    n_row = int(round((1 - noise) * n))
    feat, xy = [], []
    while len(feat) < n_row:
        P = pats[int(rng.integers(N))]; c = rng.uniform(0, side, 2)
        ang = rng.uniform(0, 2 * math.pi, len(P)); rad = 0.5 * d * np.sqrt(rng.random(len(P)))
        for f, a, r in zip(P, ang, rad):
            feat.append(int(f)); xy.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    feat, xy = feat[:n_row], xy[:n_row]
    m = n - len(feat)
    feat += rng.integers(0, k, m).tolist(); xy += [tuple(p) for p in rng.uniform(0, side, (m, 2))]
    feat = np.array(feat); xy = np.array(xy)
    used = sorted(set(feat.tolist())); feat = np.array([used.index(f) for f in feat])      # features 0..k'-1
    loc = rng.uniform(math.log(0.02), math.log(0.2), len(used))
    lt = loc[feat] + sd * rng.normal(size=n)
    cnt = rng.poisson(np.exp(lt) * E).astype(float)
    mu, s2, ep, st = ceb_posterior(cnt, E, feat, t=0)
    UB = cboot(cnt, E, st, B=B, seed=seed)
    return dict(feat=feat, xy=xy, cnt=cnt, mu=np.asarray(mu, float), s2=np.asarray(s2, float), UB=np.asarray(UB, float),
                Z=float(np.sum(ep)), d=d, E=E)


if __name__ == "__main__":
    import time
    for n in [int(x) for x in sys.argv[1:]] or [20000]:
        t0 = time.time(); G = colgen(n)
        from scipy.spatial import cKDTree
        nb = np.array([len(x) - 1 for x in cKDTree(G["xy"]).query_ball_point(G["xy"], G["d"])])
        print(n, "features", G["feat"].max() + 1, "median count", np.median(G["cnt"]), "mean neighbours %.2f" % nb.mean(),
              "UB", G["UB"].shape, "%.1fs" % (time.time() - t0), flush=True)
