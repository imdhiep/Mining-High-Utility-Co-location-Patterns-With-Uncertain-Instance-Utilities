"""Center estimation study: per category, error of the estimated center relative to its bootstrap sd, for three
centers (NPMLE median, mean-rate, moment).  No mining."""
import sys, os, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], '1', sys.argv[2] if len(sys.argv) > 2 else 'lognormal']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'semisyn_fsq.py')).read()
src = src.replace("base = Problem(F, XY, np.zeros(n), np.ones(n), DIST, 0.0, feat_names=fs)", "base = None")
src = src.replace('print("layout", city, "n", n, "k", len(fs), "cliques", len(base.cliques), flush=True)', '')
src = src[:src.index("out = {}\nfor f in FR:")]
exec(compile(src, 'semisyn', 'exec'))
from scipy.special import gammaln
def npmle(x, Wexp, J=300, iters=300, w0=None):
    lo = np.log((x.min() + 0.5) / Wexp) - 2; hi = np.log((x.max() + 0.5) / Wexp) + 1
    th = np.linspace(lo, hi, J); ll = x[:, None] * th[None, :] - Wexp * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
    L = np.exp(ll - ll.max(1, keepdims=True)); w = np.full(J, 1.0 / J)
    for _ in range(iters):
        post = L * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
    return th[np.searchsorted(np.cumsum(w), 0.5)]
def cents(x, Wexp, it=300):
    return np.array([npmle(x, Wexp, iters=it), np.log(x.sum() / (len(x) * Wexp)), momc(x, Wexp)])
Wf = W * float(os.environ.get('FF', '1.0'))
for seed in range(3):
    rng = np.random.default_rng(1000 + seed); lt = draw_truth(rng); cnt = rng.poisson(np.exp(lt) * Wf).astype(float)
    rb = np.random.default_rng(5 + seed)
    for j in range(len(fs)):
        idx = F == j; x = cnt[idx]; lam = np.exp(lt[idx])
        tru = np.array([np.median(lt[idx]), np.log(lam.mean()), momc_true(lam)])
        est = cents(x, Wf)
        bs = np.array([cents(x[rb.integers(0, len(x), len(x))], Wf, it=100) for _ in range(20)])
        sd = bs.std(0, ddof=1); bb = bs.mean(0) - est
        print(city, seed, j, len(x), " ".join(f"{e-t:+.3f}/{s:.3f}" for e, t, s in zip(est, tru, sd)), " pos=" + " ".join(f"{(lt[idx]-t>0).mean():.2f}" for t in tru), flush=True)
