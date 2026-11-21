"""Layouts of the semi-synthetic data: the instances (feature, location) and real counts of a real city, as written by
cpp/bin/fsq_prep (Foursquare) or cpp/bin/yelp_prep (Yelp), and the law of the true log-rates fitted to the real counts."""
import csv
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma


@dataclass
class Layout:
    name: str
    labels: np.ndarray        # one-character feature label of every instance
    xy: np.ndarray            # n x 2, metres
    counts: np.ndarray        # real counts of the training period
    exposure: float           # length of the training period (weeks for Foursquare, 1 for Yelp)

    @property
    def features(self):
        return sorted(set(self.labels.tolist()))

    def feature_index(self):
        index = {f: j for j, f in enumerate(self.features)}
        return np.array([index[f] for f in self.labels])


def load(path, name, exposure, count_column="a"):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return Layout(name, np.array([r["Feature"] for r in rows]), np.array([(float(r["LocX"]), float(r["LocY"])) for r in rows]),
                  np.array([float(r[count_column]) for r in rows]), float(exposure))


def truth_law(layout):
    """location and scale of the true log-rates of every feature: moments of the Gamma-Poisson posterior means of the
    real counts over the instances with at least one record (as in uhucm/ext/acc.py)"""
    F = layout.feature_index(); a = layout.counts; W = layout.exposure
    k = int(F.max()) + 1
    loc = np.zeros(k); scale = np.zeros(k)
    for j in range(k):
        x = a[F == j]
        m, v = x.mean(), x.var(ddof=1) if len(x) > 1 else 0.0
        A = m * m / (v - m) if v > m else 1e6
        beta = A * W / max(m, 1e-9)
        lr = digamma(A + x[x >= 1]) - np.log(beta + W)
        loc[j] = lr.mean() if len(lr) else np.log(max(m, 1e-9) / W)
        scale[j] = lr.std() if len(lr) else 0.0
    return loc, scale
