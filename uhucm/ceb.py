"""Compound empirical Bayes posteriors for count evidence (Section IV-A).

For every feature f, counts x_o | lambda_o ~ Poisson(lambda_o * E).  The prior of theta = ln(lambda) is the NPMLE of
Kiefer and Wolfowitz on a grid (EM); the utility is u(o) = theta_o - c_f with the moment center
c_f = 2 ln m - 0.5 ln s (m, s: first and second moments of the rates, estimated by factorial moments of the counts).
Returned per instance: posterior mean and variance of u(o), and E[u(o)^+] (expected normalizer).
`cboot` returns the conditional (compound) bootstrap matrix: counts redrawn x* ~ Poisson(x) with the instances kept,
prior and center re-estimated, posterior means recomputed for the observed counts.  Its column sums over the atoms of a
pattern give the second term of the total variance V_T(P).
"""
import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson


def moment_center(x, E):
    """c = ln m - 0.5 ln(1 + v / m^2) with m, v estimated without bias given the rates."""
    x = np.asarray(x, float); n = len(x)
    m = x.mean() / E
    v = max((x * (x - 1)).mean() / E ** 2 - m * m + m / (n * E), 0.0)
    m = max(m, 1e-12)
    return float(np.log(m) - 0.5 * np.log1p(v / (m * m)))


def center_sd(x, E):
    """Delta-method standard deviation of the moment center given the rates (Poisson factorial moments)."""
    x = np.asarray(x, float); n = len(x)
    m = x.mean() / E; s = (x * (x - 1)).mean() / E ** 2
    if m <= 0 or s <= 0: return 0.0
    f2 = x * (x - 1); f3 = f2 * (x - 2)
    vm = x.sum() / (n * E) ** 2; cms = 2 * f2.sum() / (n ** 2 * E ** 3); vs = (4 * f3 + 2 * f2).sum() / (n ** 2 * E ** 4)
    return float(np.sqrt(max(4 * vm / m ** 2 - 2 * cms / (m * s) + vs / (4 * s * s), 0.0)))


def _grid(x, E, J):
    lo = np.log((x.min() + 0.5) / E) - 3; hi = np.log((x.max() + 0.5) / E) + 1
    return np.linspace(lo, hi, J)


def _lik(x, th, E, t=0):
    """Poisson likelihood on the grid, conditioned on x >= t (instances present only with at least t events)."""
    ll = x[:, None] * th[None, :] - E * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
    if t > 0: ll = ll - np.log(np.maximum(poisson.sf(t - 1, E * np.exp(th)), 1e-300))[None, :]
    return np.exp(ll - ll.max(1, keepdims=True))


def plugin_center(th, w):
    """Moment center of the estimated prior: 2 ln E[lambda] - 0.5 ln E[lambda^2]."""
    lam = np.exp(th); return float(2 * np.log(w @ lam) - 0.5 * np.log(w @ lam ** 2))


def _em(L, w, iters):
    for _ in range(iters):
        post = L * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
    return w


def ceb_posterior(counts, E, F, J=300, iters=300, t=0):
    """Returns mu, s2, ep (arrays over instances) and a state dict for cboot.  t > 0: counts observed only when >= t
    (zero-truncated for t = 1); the center is then the moment center of the estimated prior."""
    counts = np.asarray(counts, float); F = np.asarray(F); n = len(counts)
    mu = np.zeros(n); s2 = np.zeros(n); ep = np.zeros(n); state = {}
    for f in np.unique(F):
        idx = np.nonzero(F == f)[0]; x = counts[idx]
        th = _grid(x, E, J); L = _lik(x, th, E, t); w = _em(L, np.full(J, 1.0 / J), iters)
        post = L * w; post /= post.sum(1, keepdims=True)
        c = moment_center(x, E) if t == 0 else plugin_center(th, w); mm = post @ th
        mu[idx] = mm - c; s2[idx] = np.maximum(post @ th ** 2 - mm ** 2, 1e-12); ep[idx] = post @ np.clip(th - c, 0, None)
        state[f] = (idx, th, L, w, c, t)
    return mu, s2, ep, state


def cboot(counts, E, state, B=20, iters=40, seed=0):
    """Conditional bootstrap matrix U (B x n): posterior mean of u(o) under prior and center re-estimated from
    redrawn counts x* ~ Poisson(x) (instances fixed)."""
    counts = np.asarray(counts, float); rng = np.random.default_rng(seed)
    n = len(counts); U = np.zeros((B, n))
    for f, (idx, th, L, w0, c0, t) in state.items():
        x = counts[idx]
        for b in range(B):
            xb = rng.poisson(x).astype(float)
            while t > 0 and (xb < t).any():                       # redraw below the inclusion threshold
                bad = xb < t; xb[bad] = rng.poisson(x[bad])
            Lb = _lik(xb, th, E, t); w = _em(Lb, w0.copy(), iters)
            post = L * w; post /= post.sum(1, keepdims=True)
            U[b, idx] = post @ th - (moment_center(xb, E) if t == 0 else plugin_center(th, w))
    return U


def total_var(U, s2, atoms):
    """V_T(P) for one pattern given its atom indices into the instance arrays."""
    atoms = np.asarray(atoms)
    return float(s2[atoms].sum() + U[:, atoms].sum(1).var(ddof=1))


def bfdr_select(M, V, thr, delta):
    """Bayes-optimal set selection (largest prefix in decreasing p with mean(1-p) <= delta); returns a boolean mask."""
    from scipy.stats import norm
    p = norm.cdf((np.asarray(M) - thr) / np.sqrt(np.asarray(V)))
    o = np.argsort(-p, kind="stable"); cm = np.cumsum(1 - p[o]) / np.arange(1, len(o) + 1)
    k = int(np.max(np.nonzero(cm <= delta)[0]) + 1) if (cm <= delta).any() else 0
    sel = np.zeros(len(p), bool); sel[o[:k]] = True
    return sel, p
