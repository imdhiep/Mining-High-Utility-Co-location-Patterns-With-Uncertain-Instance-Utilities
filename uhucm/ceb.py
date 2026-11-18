"""Compound empirical Bayes posteriors for count evidence (Section IV-A).

For every feature f, counts x_o | lambda_o ~ Poisson(lambda_o * E).  The prior of theta = ln(lambda) is the NPMLE of
Kiefer and Wolfowitz on a grid (EM), smoothed by a Gaussian kernel (Gu and Koenker, Econometrica 2023, Sec. 6.3) with the
bandwidth 0.9 sd n^(-1/5) of the prior; the NPMLE itself has a few atoms, and without smoothing the posterior of an
instance with many events sits on one atom, so its variance is close to zero.  Posteriors of instances whose likelihood is
narrower than the grid (sd 1/sqrt(x + 0.5) below five grid steps) are computed on a local grid of 201 points around
ln((x + 0.5) / E), with the smoothed prior density interpolated; the utility is u(o) = theta_o - c_f with the moment center
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


def smooth_prior(th, w, n):
    """Gaussian-kernel smoothing of the grid prior w with bandwidth 0.9 * sd(prior) * n^(-1/5)."""
    m = w @ th; sd = np.sqrt(max(w @ th ** 2 - m * m, 1e-12)); h = 0.9 * sd * n ** (-0.2)
    if h <= 0.5 * (th[1] - th[0]): return w
    K = np.exp(-0.5 * ((th[:, None] - th[None, :]) / h) ** 2); ws = K @ w
    return ws / ws.sum()


LOCAL_PTS, LOCAL_SD, LOCAL_STEPS = 201, 7.0, 5.0


def post_moments(x, E, th, w, t=0, L=None):
    """Posterior mean, variance and the posterior (on the global grid) of theta for counts x under the grid prior w;
    instances with a likelihood narrower than LOCAL_STEPS grid steps get mean and variance from a local fine grid.
    Returns mm, vv, post (rows of the sharp instances are set to the nearest global grid point for E[(theta-c)+]), loc
    where loc = (indices, local grids, local posteriors) for the sharp instances."""
    x = np.asarray(x, float)
    if L is None: L = _lik(x, th, E, t)
    post = L * w; post /= post.sum(1, keepdims=True); mm = post @ th; vv = np.maximum(post @ th ** 2 - mm ** 2, 1e-12)
    h = th[1] - th[0]; sd = 1.0 / np.sqrt(x + 0.5); sh = np.nonzero(sd < LOCAL_STEPS * h)[0]
    loc = (sh, None, None)
    if len(sh):
        lg = np.log(np.maximum(w, 1e-300)); xs = x[sh]
        tl = np.log((xs + 0.5) / E)[:, None] + sd[sh][:, None] * np.linspace(-LOCAL_SD, LOCAL_SD, LOCAL_PTS)[None, :]
        ll = xs[:, None] * tl - E * np.exp(tl) + np.interp(tl, th, lg)
        if t > 0: ll = ll - np.log(np.maximum(poisson.sf(t - 1, E * np.exp(tl)), 1e-300))
        q = np.exp(ll - ll.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        m2 = (q * tl).sum(1); mm[sh] = m2; vv[sh] = np.maximum((q * tl * tl).sum(1) - m2 * m2, 1e-12)
        loc = (sh, tl, q)
    return mm, vv, post, loc


def ceb_posterior(counts, E, F, J=300, iters=300, t=0):
    """Returns mu, s2, ep (arrays over instances) and a state dict for cboot.  t > 0: counts observed only when >= t
    (zero-truncated for t = 1); the center is then the moment center of the estimated prior."""
    counts = np.asarray(counts, float); F = np.asarray(F); n = len(counts)
    mu = np.zeros(n); s2 = np.zeros(n); ep = np.zeros(n); state = {}
    for f in np.unique(F):
        idx = np.nonzero(F == f)[0]; x = counts[idx]
        th = _grid(x, E, J); L = _lik(x, th, E, t); w = _em(L, np.full(J, 1.0 / J), iters)
        ws = smooth_prior(th, w, len(x))
        c = moment_center(x, E) if t == 0 else plugin_center(th, w)
        mm, vv, post, (sh, tl, q) = post_moments(x, E, th, ws, t, L)
        e = post @ np.clip(th - c, 0, None)
        if len(sh): e[sh] = (q * np.clip(tl - c, 0, None)).sum(1)
        mu[idx] = mm - c; s2[idx] = vv; ep[idx] = e
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
            mm = post_moments(x, E, th, smooth_prior(th, w, len(x)), t, L)[0]
            U[b, idx] = mm - (moment_center(xb, E) if t == 0 else plugin_center(th, w))
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
