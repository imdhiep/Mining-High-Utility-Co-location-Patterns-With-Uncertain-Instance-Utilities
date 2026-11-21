"""Binary posterior file read by the C++ programs (cpp/src/common/utility_posterior.hpp):
int64 n, int64 B, double Z, n means, n variances, B x n bootstrap means (row major), little endian."""
import numpy as np


def write(path, mean, variance, bootstrap=None, normalizer=0.0):
    mean = np.asarray(mean, "<f8"); variance = np.asarray(variance, "<f8")
    B = 0 if bootstrap is None else int(np.asarray(bootstrap).shape[0])
    with open(path, "wb") as f:
        np.array([len(mean), B], "<i8").tofile(f)
        np.array([normalizer], "<f8").tofile(f)
        mean.tofile(f); variance.tofile(f)
        if B:
            np.asarray(bootstrap, "<f8").reshape(B, len(mean)).tofile(f)


def read(path):
    with open(path, "rb") as f:
        n, B = (int(v) for v in np.fromfile(f, "<i8", 2))
        Z = float(np.fromfile(f, "<f8", 1)[0])
        mean = np.fromfile(f, "<f8", n); variance = np.fromfile(f, "<f8", n)
        bootstrap = np.fromfile(f, "<f8", B * n).reshape(B, n) if B else None
    return dict(mean=mean, variance=variance, bootstrap=bootstrap, normalizer=Z)
