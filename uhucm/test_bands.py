"""Exactness of mining under the distance-banded correlation model: UHUCM (bounds computed under independence) against
enumeration of every feature set on random instances. Usage: python test_bands.py [n_instances]"""
import sys, itertools, numpy as np
import fastmine
from fastmine import mine, pattern_atoms
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
bad = 0; nonempty = 0
for seed in range(N):
    rng = np.random.default_rng(seed); n = int(rng.integers(10, 40)); k = int(rng.integers(3, 6))
    feat = rng.integers(0, k, n); feat[:k] = np.arange(k)
    xy = rng.uniform(0, 250, (n, 2)); mu = rng.normal(0.4, 1.0, n); s2 = rng.uniform(0.0, 0.6, n)
    z = float(rng.choice([0.0, 1.0, 1.645, 3.0]))
    P = fastmine.Problem(feat, xy, mu, s2, 60.0, z, feat_names=[chr(65 + i) for i in range(k)])
    P.set_bands([0, 40, 80, 160], list(rng.uniform(0, 0.5, 3)))
    if P.Z <= 0 or len(P.cliques) == 0: continue
    thr = float(rng.choice([0.0, 0.02, 0.05, 0.1, 0.2]))
    for kind, use in (("two", (1, 1, 1, 1)), ("exact", (1, 1, 1, 1)), ("mean", (1, 1, 1, 1)), ("two", (0, 0, 0, 0))):
        R, st, _ = mine(P, thr, use, kind)
        truth = {}
        for r in range(2, k + 1):
            for c in itertools.combinations(range(k), r):
                names = tuple(P.names[f] for f in c); A = pattern_atoms(P, names)
                if len(A) == 0: continue
                v = P.score(A)
                if v >= thr - 1e-12: truth[tuple(sorted(names))] = v
        if set(R) != set(truth) or any(abs(R[p] - truth[p]) > 1e-9 for p in R): bad += 1
        nonempty += bool(truth)
print(f"checked {N} instances x 4 variants, mismatches {bad}, non-empty answers {nonempty}")
