"""Pilot: semi-synthetic ground truth on a real layout.  Venues, categories and neighbourhoods are those of the city;
true log-rates are drawn per category from a normal fitted to the full-data posterior means (so the Gamma prior of
Def. 4 is misspecified); counts ~ Poisson(lambda * W * f).  True utility u* = log-rate - true category center (moment
center when TRUTHC=momc, the setting of the paper); with TRUTHK=1 the threshold gamma is the K-th largest TRUE score and
the true HUCPs are the patterns whose true normalized score reaches it, the same set for every method.  Reported: FDP and detection of
point-utility mining (z = 0), the lower bound (z = 1.645, 2.45), Bayesian set selection (largest top-P set with
mean(1-P) <= delta), and equal-size precision of point vs posterior ranking."""
import sys, os, json
import numpy as np
from scipy.special import digamma, polygamma
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fsq_loader, paths
from fastmine import Problem, mine, pattern_atoms
city = sys.argv[1]; FR = tuple(float(x) for x in os.environ.get('FR', '0.25,1.0').split(',')); METHOD = sys.argv[4] if len(sys.argv) > 4 else 'gamma'; ZPOST = 'zpost' in sys.argv[5:]; DIAG = 'diag' in sys.argv[5:]; MCC = 'mcc' in sys.argv[5:]; CAL = 'cal' in sys.argv[5:]; BBM = 'bbm' in sys.argv[5:]; MEANC = 'meanc' in sys.argv[5:]; MOMC = 'momc' in sys.argv[5:]; BCC = 'bcc' in sys.argv[5:]; KS = tuple(int(x) for x in os.environ.get('KS', '100,300,1000').split(',')); BOOT = 'boot' in sys.argv[5:] or 'cboot' in sys.argv[5:]; CBOOT = 'cboot' in sys.argv[5:]; SEEDS = range(int(os.environ.get('SEED0', '0')), int(sys.argv[2]) if len(sys.argv) > 2 else 3)
CSV = {"LasVegas": ("LasVegas_POI 10f.csv", 100.0), "California": ("California_POI 13f.csv", 1300.0), "Toronto": ("Toronto_POI 16f.csv", 100.0)}
if city in CSV:                                                # D6 layouts: total check-ins per POI, exposure 1
    import pandas as pd
    df = pd.read_csv(os.path.join(os.environ["D6_DIR"], CSV[city][0])); DIST = CSV[city][1]
    a = df.Checkin.to_numpy(float); W = 1.0; featn = df.Feature.astype(str).to_numpy(); fs = sorted(set(featn))
    F = np.array([fs.index(g) for g in featn]); XY = list(zip(df.LocX, df.LocY)); n = len(df)
    lr = np.log(a + 0.5)
else:
    rows, users, NU, info = fsq_loader.load(paths.fsq(city), city); DIST = 50.0
    a = np.array([r["a"] for r in rows]).astype(float); W = rows[0]["W"]; featn = np.array([r["feat"] for r in rows]); fs = sorted(set(featn))
    F = np.array([fs.index(g) for g in featn]); XY = [(r["x"], r["y"]) for r in rows]; n = len(rows)
    al0 = np.array([r["alpha"] for r in rows]); be0 = np.array([r["beta"] for r in rows])
    lr = digamma(al0 + a) - np.log(be0 + W)                  # full-data posterior mean log-rates
loc = np.array([lr[F == j].mean() for j in range(len(fs))]); scl = np.array([lr[F == j].std() for j in range(len(fs))])
base = Problem(F, XY, np.zeros(n), np.ones(n), DIST, 0.0, feat_names=fs)
print("layout", city, "n", n, "k", len(fs), "cliques", len(base.cliques), flush=True)
import functools
def with_values(mu, s2, z):
    P = Problem.__new__(Problem); P.__dict__.update(base.__dict__)
    amu = np.asarray(mu, float); as2 = np.asarray(s2, float); P.amu, P.as2, P.z = amu, as2, z
    P.Z = float(np.clip(amu, 0, None).sum())
    posi = [i for i in range(len(amu)) if amu[i] > 0]
    cmp = lambda i, j: int(as2[i] * amu[j] > as2[j] * amu[i]) - int(as2[i] * amu[j] < as2[j] * amu[i])
    posi.sort(key=functools.cmp_to_key(cmp))
    P.rank = -np.ones(len(amu), dtype=np.int64); P.rank[posi] = np.arange(len(posi)); P.smu = amu[posi]; P.ss2 = as2[posi]
    return P
def topk(P, K, start=0.05):
    m = start
    while True:
        R = mine(P, m)[0]
        if len(R) >= K or m < 1e-9: return R, sorted(R.values(), reverse=True)
        m /= 2
def posterior(cnt, Wexp):
    al = np.zeros(n); be = np.zeros(n)
    for j in range(len(fs)):
        x = cnt[F == j]; m_, v_ = x.mean(), x.var(ddof=1)
        A = m_ * m_ / (v_ - m_) if v_ > m_ else 1e6; al[F == j] = A; be[F == j] = A * Wexp / max(m_, 1e-9)
    m = digamma(al + cnt) - np.log(be + Wexp); v = polygamma(1, al + cnt)
    if MEANC: return m - np.array([np.log(cnt[F == j].sum() / ((F == j).sum() * Wexp)) for j in range(len(fs))])[F], v
    return m - np.array([np.median(m[F == j]) for j in range(len(fs))])[F], v
from scipy.special import gammaln
def momc(x, Wexp):
    """Moment center: ln m - ln(1 + v/m^2)/2 with m, v the mean and variance of the rates, estimated without
    deconvolution from the first two factorial moments of the counts (equals the median if the rates are log-normal)."""
    m = x.mean() / Wexp; v = max((x * (x - 1)).mean() / Wexp ** 2 - m * m + m / (len(x) * Wexp), 0.0)   # unbiased given lambda
    return float(np.log(m) - 0.5 * np.log1p(v / (m * m)))
def momc_true(lam):
    m = lam.mean(); v = lam.var(); return float(np.log(m) - 0.5 * np.log1p(v / (m * m)))
def posterior_np(cnt, Wexp, J=300, iters=300):
    """Nonparametric empirical Bayes (Kiefer-Wolfowitz NPMLE on a grid of log-rates, EM) per category; utility =
    posterior mean of the log-rate minus the median of the estimated prior; variance = posterior variance."""
    m = np.zeros(n); v = np.zeros(n); global EP; EP = np.zeros(n)
    for j in range(len(fs)):
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        lo = np.log((x.min() + 0.5) / Wexp) - 2; hi = np.log((x.max() + 0.5) / Wexp) + 1
        th = np.linspace(lo, hi, J); ll = x[:, None] * th[None, :] - Wexp * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
        L = np.exp(ll - ll.max(1, keepdims=True)); w = np.full(J, 1.0 / J)
        for _ in range(iters):
            post = L * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
        post = L * w; post /= post.sum(1, keepdims=True)
        mm = post @ th; vv = post @ th ** 2 - mm ** 2
        cw = np.cumsum(w); med = th[np.searchsorted(cw, 0.5)]; W0[j] = w; TH0[j] = th
        if MEANC: med = float(np.log(x.sum() / (len(x) * Wexp)))
        if MOMC: med = momc(x, Wexp)
        if BCC:                                                    # parametric-bootstrap bias correction of the median
            rb = np.random.default_rng(4242 + j); diffs = []
            for _ in range(10):
                ts = rb.choice(th, len(x), p=w / w.sum()) + rb.uniform(-0.5, 0.5, len(x)) * (th[1] - th[0])
                xs = rb.poisson(np.exp(ts) * Wexp).astype(float)
                lls = xs[:, None] * th[None, :] - Wexp * np.exp(th)[None, :] - gammaln(xs + 1)[:, None]
                Ls = np.exp(lls - lls.max(1, keepdims=True)); ws = w.copy()
                for _ in range(100):
                    ps = Ls * ws; ps /= ps.sum(1, keepdims=True); ws = np.maximum(ps.mean(0), 1e-300)
                diffs.append(th[np.searchsorted(np.cumsum(ws), 0.5)] - np.median(ts))
            med = med - float(np.mean(diffs))
        if MCC:                                                    # posterior of the realized category median
            rngc = np.random.default_rng(777 + j); cp = np.cumsum(post, 1); S_ = 400
            draws = th[np.minimum((cp[:, None, :] < rngc.random((len(idx), S_, 1))).sum(2), J - 1)]
            cd = np.median(draws, 0); med = float(cd.mean()); CVAR[j] = float(cd.var())
        m[idx] = mm - med; v[idx] = np.maximum(vv, 1e-12); EP[idx] = post @ np.clip(th - med, 0, None); CEN[j] = med
    return m, v
LT = None
def posterior_oracle(cnt, Wexp, J=400):
    """Oracle: posterior under the true prior N(loc_f, scl_f) of the log-rate (lognormal truth) and the true realized
    center; the ceiling for any method that observes the same counts."""
    m = np.zeros(n); v = np.zeros(n); global EP; EP = np.zeros(n)
    for j in range(len(fs)):
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        th = np.linspace(loc[j] - 8 * scl[j], loc[j] + 6 * scl[j], J)
        if TRUTH == "gamma":
            k = 1.0 / max(scl[j] ** 2, 1e-3); y = th - loc[j] + digamma(k); w = np.exp(k * y - np.exp(y) - gammaln(k))
        elif TRUTH == "mixture":
            w = 0.7 * norm.pdf(th, loc[j] - 0.45 * scl[j], 0.7 * scl[j]) + 0.3 * norm.pdf(th, loc[j] + 1.05 * scl[j], 0.7 * scl[j])
        else: w = norm.pdf(th, loc[j], scl[j])
        w = w / w.sum()
        ll = x[:, None] * th[None, :] - Wexp * np.exp(th)[None, :]; post = np.exp(ll - ll.max(1, keepdims=True)) * w
        post /= post.sum(1, keepdims=True); lam = np.exp(LT[idx])
        med = momc_true(lam) if MOMC else float(np.log(lam.mean())) if MEANC else float(np.median(LT[idx]))
        mm = post @ th; m[idx] = mm - med; v[idx] = np.maximum(post @ th ** 2 - mm ** 2, 1e-12)
        EP[idx] = post @ np.clip(th - med, 0, None); CEN[j] = med
    return m, v
AB = None
def cat_moments(cnt, Wexp, J=300, iters=300):
    """Mean and standard deviation of the per-category NPMLE prior of the log-rate."""
    out = []
    for j in range(len(fs)):
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        lo = np.log((x.min() + 0.5) / Wexp) - 2; hi = np.log((x.max() + 0.5) / Wexp) + 1
        th = np.linspace(lo, hi, J); ll = x[:, None] * th[None, :] - Wexp * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
        L = np.exp(ll - ll.max(1, keepdims=True)); w = np.full(J, 1.0 / J)
        for _ in range(iters):
            post = L * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
        mth = float(w @ th); out.append((mth, max(float(np.sqrt(w @ (th - mth) ** 2)), 0.05)))
    return out
def _np_pool(cnt, Wexp, J=300, iters=300, boot_rng=None):
    """Location-scale NPMLE pooled over categories: log-rate = a_f + b_f * eta with eta ~ G0 common to all categories
    (a_f, b_f by moments of log((x+0.5)/E)); G0 by EM on a common grid; returns posterior mean, variance, E[u+] and
    the category medians a_f + b_f * median(G0)."""
    eta = np.linspace(-5, 5, J); af = np.zeros(len(fs)); bf = np.zeros(len(fs)); ll = np.zeros((n, J))
    sel = np.arange(n) if boot_rng is None else np.concatenate([boot_rng.choice(np.nonzero(F == j)[0], (F == j).sum()) for j in range(len(fs))])
    for j in range(len(fs)):
        idx = np.nonzero(F == j)[0]
        if AB is None:
            y = np.log((cnt[idx] + 0.5) / Wexp); af[j] = y.mean(); bf[j] = np.sqrt(max(y.var() - np.mean(1.0 / (cnt[idx] + 0.5)), 0.05 * y.var()))
        else: af[j], bf[j] = AB[j]
        th = af[j] + bf[j] * eta[None, :]
        ll[idx] = cnt[idx][:, None] * th - Wexp * np.exp(th) - gammaln(cnt[idx] + 1)[:, None]
    L = np.exp(ll - ll.max(1, keepdims=True)); w = np.full(J, 1.0 / J); Ls = L[sel]
    for _ in range(iters):
        post = Ls * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
    post = L * w; post /= post.sum(1, keepdims=True)
    med = eta[np.searchsorted(np.cumsum(w), 0.5)]
    TH = af[F][:, None] + bf[F][:, None] * eta[None, :]; c = af + bf * med
    m = (post * TH).sum(1) - c[F]; v = (post * TH ** 2).sum(1) - (post * TH).sum(1) ** 2
    ep = (post * np.clip(TH - c[F][:, None], 0, None)).sum(1)
    return m, np.maximum(v, 1e-12), ep
def posterior_pool(cnt, Wexp):
    global EP, AB
    if METHOD == 'pool2': AB = cat_moments(cnt, Wexp)
    m, v, EP = _np_pool(cnt, Wexp); return m, v
def boot_pool(cnt, Wexp, B=20, seed=0):
    rng = np.random.default_rng(seed); return np.array([_np_pool(cnt, Wexp, iters=150, boot_rng=rng)[0] for _ in range(B)])
W0 = {}; TH0 = {}
CEN = np.zeros(64); CVAR = np.zeros(64)
def posterior_cf(cnt, Wexp, J=300, iters=300, folds=5):
    """Cross-fitted NPMLE: the prior used for a venue is estimated without that venue's fold (no double use of a
    venue's own evidence in the tail of its prior); centering by the median of the full-sample prior."""
    global EP; m = np.zeros(n); v = np.zeros(n); EP = np.zeros(n); rng = np.random.default_rng(12345)
    for j in range(len(fs)):
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        lo = np.log((x.min() + 0.5) / Wexp) - 2; hi = np.log((x.max() + 0.5) / Wexp) + 1
        th = np.linspace(lo, hi, J); ll = x[:, None] * th[None, :] - Wexp * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
        L = np.exp(ll - ll.max(1, keepdims=True))
        def em(Ls, w=None, it=iters):
            w = np.full(J, 1.0 / J) if w is None else w.copy()
            for _ in range(it):
                post = Ls * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
            return w
        wf = em(L); W0[j] = wf; med = th[np.searchsorted(np.cumsum(wf), 0.5)]
        fold = rng.permutation(len(idx)) % folds
        for k in range(folds):
            tr, te = fold != k, fold == k
            w = em(L[tr], wf, 100); post = L[te] * w; post /= post.sum(1, keepdims=True)
            mm = post @ th; m[idx[te]] = mm - med; v[idx[te]] = np.maximum(post @ th ** 2 - mm ** 2, 1e-12)
            EP[idx[te]] = post @ np.clip(th - med, 0, None)
    return m, v
def boot_np(cnt, Wexp, B=20, J=300, iters=40, seed=0):
    """Prior-estimation uncertainty: venues are resampled within each category, the NPMLE prior (EM warm-started at the
    full-sample estimate) and its median are re-estimated, and every venue's posterior mean is recomputed under the
    bootstrap prior (law of total variance)."""
    rng = np.random.default_rng(seed); U = np.zeros((B, n))
    for j in range(len(fs)):
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        lo = np.log((x.min() + 0.5) / Wexp) - 2; hi = np.log((x.max() + 0.5) / Wexp) + 1
        th = np.linspace(lo, hi, J); ll = x[:, None] * th[None, :] - Wexp * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
        L = np.exp(ll - ll.max(1, keepdims=True))
        for b in range(B):
            if CBOOT:                                              # conditional (compound) bootstrap: counts redrawn, venues fixed
                xb = rng.poisson(x).astype(float); bi = None
                llb = xb[:, None] * th[None, :] - Wexp * np.exp(th)[None, :]; Lb = np.exp(llb - llb.max(1, keepdims=True))
            else:
                bi = rng.integers(0, len(idx), len(idx)); Lb = L[bi]; xb = x[bi]
            w = W0[j].copy()
            for _ in range(iters):
                post = Lb * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
            post = L * w; post /= post.sum(1, keepdims=True)
            U[b, idx] = post @ th - (momc(xb, Wexp) if MOMC else float(np.log(xb.sum() / (len(xb) * Wexp))) if MEANC else th[np.searchsorted(np.cumsum(w), 0.5)])
    return U
TRUTH = sys.argv[3] if len(sys.argv) > 3 else "lognormal"
TRUTHK = os.environ.get("TRUTHK") == "1"   # threshold = K-th largest TRUE score (same truth set for every method)
TRUTHC = os.environ.get("TRUTHC", "momc" if TRUTHK else "")   # truth center fixed by the problem (moment center), not by the estimator
SPF = os.environ.get("SPF")            # spatial field for review item A2: "range_m,share" (Gaussian kernel, share of log-rate variance)
XYa = np.asarray(XY, float)
def spatial_field(rng, rng_m, M=500):
    """Gaussian random field with kernel exp(-h^2 / (2 range^2)) at the venue locations, unit variance (random Fourier features)."""
    Wf = rng.normal(0.0, 1.0 / rng_m, size=(2, M)); bf = rng.uniform(0, 2 * np.pi, M)
    g = np.sqrt(2.0 / M) * np.cos((XYa - XYa.mean(0)) @ Wf + bf).sum(1); return g / g.std()
def draw_truth(rng):
    if TRUTH == "lognormal" and SPF:
        rm, sh = (float(x) for x in SPF.split(","))
        return loc[F] + scl[F] * (np.sqrt(1 - sh) * rng.normal(size=n) + np.sqrt(sh) * spatial_field(rng, rm))
    if TRUTH == "lognormal": return rng.normal(loc[F], scl[F])
    if TRUTH == "gamma":                                           # log of a Gamma rate with the same log-mean and log-sd
        k = 1.0 / np.maximum(scl[F] ** 2, 1e-3); lam = rng.gamma(k, 1.0, size=n); return np.log(lam) - digamma(k) + loc[F]
    if TRUTH == "mixture":                                         # two-component log-rate mixture, 30% at +1.5 sd
        z = rng.random(n) < 0.3; return rng.normal(loc[F] - 0.45 * scl[F] + z * 1.5 * scl[F], 0.7 * scl[F])
ZG = np.array([1.0, 1.28, 1.645, 2.0, 2.45, 3.0, 3.5, 4.0, 5.0])
def pipeline(cnt, Wexp, seed):
    """posterior (NPMLE, expected normalizer, bootstrap variance) -> (mu, s2, problem, UB)"""
    mu, s2 = posterior_np(cnt, Wexp); P0 = with_values(mu, s2, 0.0)
    P0.Z = float(EP.sum()); UB = boot_np(cnt, Wexp, B=10, seed=seed) if BOOT else None
    return mu, s2, P0, UB
def calibrate(cnt, Wexp, Ks, R=8, seed=0, delta=0.05):
    """Parametric-bootstrap calibration: datasets are simulated from the fitted priors (same layout and exposure), the
    whole pipeline is rerun, and z (lower bound) and delta' (set selection) are chosen so that the mean simulated FDP is
    at most delta, separately for each K."""
    rng = np.random.default_rng(99 + seed); Gs = {j: (TH0[j], W0[j].copy()) for j in range(len(fs))}
    fdp_z = {K: [] for K in Ks}; fdp_d = {K: [] for K in Ks}; DG = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1])
    for r in range(R):
        lt = np.zeros(n)
        for j in range(len(fs)):
            th, w = Gs[j]; idx = np.nonzero(F == j)[0]; lt[idx] = rng.choice(th, len(idx), p=w / w.sum()) + rng.uniform(-0.5, 0.5, len(idx)) * (th[1] - th[0])
        ut = lt - np.array([np.median(lt[F == j]) for j in range(len(fs))])[F]
        c = rng.poisson(np.exp(lt) * Wexp).astype(float)
        mu, s2, P0, UB = pipeline(c, Wexp, 1000 + r); R0, sc = topk(P0, max(Ks)); Z = P0.Z
        Pt = with_values(ut, np.zeros(n), 0.0)
        for K in Ks:
            if len(sc) < K: continue
            g = sc[K - 1]; T = set(mine(Pt, g)[0]); pool = [p for p, v in R0.items() if v >= g]
            As = [pattern_atoms(P0, p) for p in pool]; M = np.array([mu[x].sum() for x in As]); V = np.array([s2[x].sum() for x in As])
            if UB is not None: V = V + np.array([UB[:, x].sum(1).var(ddof=1) for x in As])
            tr = np.array([p in T for p in pool])
            fdp_z[K].append([float(1 - tr[M - z * np.sqrt(V) >= g * Z].mean()) if (M - z * np.sqrt(V) >= g * Z).any() else 0.0 for z in ZG])
            prob = norm.cdf((M - g * Z) / np.sqrt(V)); o = np.argsort(-prob); cm = np.cumsum(1 - prob[o]) / np.arange(1, len(o) + 1)
            row = []
            for d in DG:
                k = int(np.max(np.nonzero(cm <= d)[0]) + 1) if (cm <= d).any() else 0
                row.append(float(1 - tr[o[:k]].mean()) if k else 0.0)
            fdp_d[K].append(row)
    zc, dc = {}, {}
    for K in Ks:
        if not fdp_z[K]: zc[K] = 1.645; dc[K] = delta; continue
        mz = np.mean(fdp_z[K], 0); md = np.mean(fdp_d[K], 0)
        zc[K] = float(ZG[np.argmax(mz <= delta)]) if (mz <= delta).any() else float(ZG[-1])
        dc[K] = float(DG[::-1][np.argmax(md[::-1] <= delta)]) if (md <= delta).any() else float(DG[0])
    return zc, dc
out = {}
for f in FR:
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        lt = draw_truth(rng)
        ut = lt - np.array([(momc_true(np.exp(lt[F == j])) if (MOMC or TRUTHC == 'momc') else np.log(np.exp(lt[F == j]).mean()) if MEANC else np.median(lt[F == j])) for j in range(len(fs))])[F]
        cnt = rng.poisson(np.exp(lt) * W * f).astype(float)
        LT = lt; mu, s2 = {'oracle': posterior_oracle, 'np': posterior_np, 'cf': posterior_cf, 'pool': posterior_pool, 'pool2': posterior_pool}.get(METHOD, posterior)(cnt, W * f)
        P0 = with_values(mu, s2, 0.0)
        if ZPOST:
            sd = np.sqrt(s2); ep = EP if METHOD in ('np', 'cf', 'pool', 'pool2', 'oracle') else mu * norm.cdf(mu / sd) + sd * norm.pdf(mu / sd)
            P0.Z = float(ep.sum())
        R0all, sc = topk(P0, max(KS)); Z = P0.Z
        if BOOT: UB = (boot_pool if METHOD in ('pool', 'pool2') else boot_np)(cnt, W * f, seed=seed)
        if CAL: ZC, DC = calibrate(cnt, W * f, (100, 300, 1000), seed=seed); print("cal", f, seed, ZC, DC, flush=True)
        Pt = with_values(ut, np.zeros(n), 0.0); Zt = Pt.Z
        if TRUTHK:
            Rt, sct = topk(Pt, max(KS)); GS = {K: sct[K - 1] for K in KS if len(sct) >= K}
            if GS: R0all = mine(P0, min(GS.values()))[0]
        if SPF:
            from scipy.spatial import cKDTree
            EDGES = [0, 50, 100, 200, 400, 800, 1600]; rr = mu - mu.mean(); TT = cKDTree(XYa); RHO = []
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                PP = TT.query_pairs(hi, output_type="ndarray"); dd = np.linalg.norm(XYa[PP[:, 0]] - XYa[PP[:, 1]], axis=1); PP = PP[dd > lo]
                aa, bb = rr[PP[:, 0]], rr[PP[:, 1]]; RHO.append(round(float(np.corrcoef(np.r_[aa, bb], np.r_[bb, aa])[0, 1]), 4))
            P0b = with_values(mu, s2, 0.0); P0b.bands = None; P0b.rho = 0.0; P0b.set_bands(EDGES, [max(0.0, x) for x in RHO])
        for K in KS:
            if TRUTHK:
                if K not in GS: out[f"{f}|{seed}|{K}"] = None; continue
                g = GS[K]; T = {p for p, v in Rt.items() if v >= g}   # the K highest true scores
            else:
                if len(sc) < K: out[f"{f}|{seed}|{K}"] = None; continue
                g = sc[K - 1]; T = set(mine(Pt, g)[0])             # true HUCPs at the same normalized threshold
            pool = [p for p, v in R0all.items() if v >= g]
            As = [pattern_atoms(P0, p) for p in pool]
            M = np.array([mu[x].sum() for x in As]); V = np.array([s2[x].sum() for x in As])
            if BOOT: V = V + np.array([UB[:, x].sum(1).var(ddof=1) for x in As])
            if BBM: M = np.array([UB[:, x].sum(1).mean() for x in As])     # bootstrap-averaged posterior mean
            if MCC: V = V + np.array([(np.bincount(F[x[x < n]], minlength=len(fs)) ** 2 * CVAR[:len(fs)]).sum() for x in As])
            tr = np.array([p in T for p in pool]); prob = norm.cdf((M - g * Z) / np.sqrt(V))
            res = dict(nT=len(T), n0=len(pool))
            Mt = np.array([ut[x].sum() for x in As]); ze = (Mt * Z / Zt - M) / np.sqrt(V)
            res["zerr"] = [round(float(ze.mean()), 2), round(float(ze.std()), 2), round(float(Zt / Z), 3)]
            top = np.argsort(-M)[:max(1, len(M) // 4)]; res["zerr_top"] = [round(float(ze[top].mean()), 2), round(float(ze[top].std()), 2)]
            if DIAG:
                cT = np.array([(momc_true(np.exp(lt[F == j])) if MOMC else np.log(np.exp(lt[F == j]).mean()) if MEANC else np.median(lt[F == j])) for j in range(len(fs))])
                th_hat = mu + CEN[F]                                         # posterior mean log-rate
                ev = np.array([(lt[x] - th_hat[x]).sum() for x in As]); ec = np.array([(CEN[F][x] - cT[F][x]).sum() for x in As])
                en = (Mt * Z / Zt - Mt)
                res["vshare_top"] = [round(float((V[top] - np.array([s2[x].sum() for x in [As[i] for i in top]])).mean() / V[top].mean()), 3), round(float(np.mean([len(As[i]) for i in top])), 1)]; res["diag_top"] = [round(float((ev[top] / np.sqrt(V[top])).mean()), 2), round(float((ec[top] / np.sqrt(V[top])).mean()), 2), round(float((en[top] / np.sqrt(V[top])).mean()), 2)]
            if SPF:                                                   # distance-banded variance estimated from the posterior means
                Vb = V + np.array([P0b._banded_extra(x[x < n]) for x in As]); res["lb_band"] = None
                res["rho_band"] = RHO
            def fd(sel): sel = np.asarray(sel, bool); k = int(sel.sum()); return [k, round(float(1 - tr[sel].mean()) if k else 0.0, 3), round(float(tr[sel].sum() / max(1, len(T))), 3)]
            res["point"] = fd(np.ones(len(pool)))
            if SPF: res["lb_band"] = fd(M - 1.645 * np.sqrt(Vb) >= g * Z)
            for z in (1.645, 2.45): res[f"lb{z}"] = fd(M - z * np.sqrt(V) >= g * Z)
            o = np.argsort(-prob, kind="stable"); cm = np.cumsum(1 - prob[o]) / np.arange(1, len(o) + 1)
            if CAL:
                res["lbcal"] = fd(M - ZC[K] * np.sqrt(V) >= g * Z); dl = DC[K]
                k = int(np.max(np.nonzero(cm <= dl)[0]) + 1) if (cm <= dl).any() else 0
                sel = np.zeros(len(pool), bool); sel[o[:k]] = True; res["bfdrcal"] = fd(sel)
            for dl in (0.05, 0.1):
                k = int(np.max(np.nonzero(cm <= dl)[0]) + 1) if (cm <= dl).any() else 0
                sel = np.zeros(len(pool), bool); sel[o[:k]] = True; res[f"bfdr{dl}"] = fd(sel)
            for fm in (0.25, 0.5):
                m = max(1, int(round(fm * len(pool))))
                res[f"eq{fm}"] = [round(float(tr[np.argsort(-M, kind="stable")[:m]].mean()), 3), round(float(tr[o[:m]].mean()), 3)]
            out[f"{f}|{seed}|{K}"] = res; print(city, f, seed, K, json.dumps(res), flush=True)
json.dump(out, open(f"pilot/{os.environ.get('OUTD', 'main')}/semisyn_{city}_{TRUTH}_{METHOD}{'_zpost' if ZPOST else ''}{'_cboot' if CBOOT else '_boot' if BOOT else ''}{'_mcc' if MCC else ''}{'_cal' if CAL else ''}{'_bbm' if BBM else ''}{'_meanc' if MEANC else ''}{'_momc' if MOMC else ''}{'_bcc' if BCC else ''}{'_spf' if SPF else ''}.json", "w"), indent=1)
