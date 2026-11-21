"""Synthetic check-in data for the running-time experiments (Section VI-C).

Locations follow the generator of Huang et al. (2004), as used by PSO-MUCP: N core patterns of 2 to 5 features are
drawn from k features, and each row instance of a core pattern places one instance of each of its features within d/2
of a random point, so that the row is a clique. A share `noise` of the instances is placed uniformly with a random
feature. The square area grows with n so that the density of the noise instances, and with it the mean number of
neighbours, stays the same for every n.
Check-ins follow Definition 4: for each feature the log-rates per week are normal with a feature mean drawn uniformly
in [ln 0.02, ln 0.2] and standard deviation `sd`, and the counts over E weeks are Poisson (same as uhucm/ext/syngen.py;
the posterior is computed by cpp/bin/post_counts)."""
import math

import numpy as np

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%()*+-./:;<=>?@[]^_`{|}~"


def generate(n, k=20, seed=7, d=50.0, N=20, noise=0.5, sd=1.0, E=34.3, smin=2, smax=5):
    rng = np.random.default_rng(seed)
    side = math.sqrt(n * math.pi * d * d)                    # about one noise instance per disc of radius d
    patterns = [rng.choice(k, size=int(rng.integers(smin, smax + 1)), replace=False) for _ in range(N)]
    n_row = int(round((1 - noise) * n))
    feat, xy = [], []
    while len(feat) < n_row:
        P = patterns[int(rng.integers(N))]; c = rng.uniform(0, side, 2)
        ang = rng.uniform(0, 2 * math.pi, len(P)); rad = 0.5 * d * np.sqrt(rng.random(len(P)))
        for f, a, r in zip(P, ang, rad):
            feat.append(int(f)); xy.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    feat, xy = feat[:n_row], xy[:n_row]
    m = n - len(feat)
    feat += rng.integers(0, k, m).tolist(); xy += [tuple(p) for p in rng.uniform(0, side, (m, 2))]
    feat = np.array(feat); xy = np.array(xy)
    used = sorted(set(feat.tolist())); feat = np.array([used.index(f) for f in feat])
    loc = rng.uniform(math.log(0.02), math.log(0.2), len(used))
    counts = rng.poisson(np.exp(loc[feat] + sd * rng.normal(size=n)) * E).astype(float)
    return dict(labels=np.array([LABELS[f] for f in feat]), xy=xy, counts=counts, exposure=E, d=d)
