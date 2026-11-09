"""Exactness of mining with the bootstrap variance term: UHUCPM (all bounds) vs NoPrune on random instances."""
import sys, os, numpy as np, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastmine import Problem, mine
bad = 0; tot = 0
for t in range(300):
    rng = np.random.default_rng(t); n = int(rng.integers(20, 60)); k = int(rng.integers(3, 6))
    F = rng.integers(0, k, n); XY = rng.random((n, 2)) * 3
    mu = rng.normal(0.2, 1, n); s2 = rng.gamma(1, 0.3, n); B = 15; U = mu + rng.normal(0, 0.3, (B, n)) + rng.normal(0, 0.3, (B, 1)) * (F == 0)
    z = float(rng.choice([0.5, 1.645, 2.45])); P = Problem(F, XY, mu, s2, 1.0, z); P.set_boot(U, Z=float(np.clip(mu, 0, None).sum() * rng.uniform(0.8, 1.3)))
    if P.Z <= 0: continue
    for thr in (0.01, 0.05, 0.1, 0.2):
        for kind in ("exact", "mean", "chord"):
            a = mine(P, thr, (1, 1, 1, 1), kind)[0]; b = mine(P, thr, (0, 0, 0, 0), kind)[0]
            tot += 1; bad += set(a) != set(b) or any(abs(a[q] - b[q]) > 1e-9 for q in a)
print("runs", tot, "mismatches", bad)
