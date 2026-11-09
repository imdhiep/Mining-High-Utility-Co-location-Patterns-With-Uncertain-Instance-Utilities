import numpy as np, sys
sys.path.insert(0, '.')
from ceb import moment_center, _grid, _em
from scipy.special import gammaln
from scipy.stats import poisson
def ctrue(l): return 2 * np.log(l.mean()) - 0.5 * np.log((l ** 2).mean())
def npmle_c(x, E, t=0, J=300, iters=500):
    th = np.linspace(np.log((x.min() + 0.5) / E) - 3, np.log((x.max() + 0.5) / E) + 1, J)
    ll = x[:, None] * th[None, :] - E * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
    if t > 0: ll = ll - np.log(np.maximum(poisson.sf(t - 1, E * np.exp(th)), 1e-300))[None, :]
    L = np.exp(ll - ll.max(1, keepdims=True)); w = _em(L, np.full(J, 1.0 / J), iters)
    lam = np.exp(th); return 2 * np.log(w @ lam) - 0.5 * np.log(w @ lam ** 2), w, th
rng = np.random.default_rng(1)
for E, sd, t in [(1.0, 1.2, 0), (0.3, 1.2, 0), (1.0, 1.5, 1), (0.3, 1.5, 1), (5.0, 1.0, 11)]:
    e1, e2 = [], []
    for r in range(10):
        lt = rng.normal(0.5, sd, 6000); x = rng.poisson(np.exp(lt) * E).astype(float); keep = x >= t; x = x[keep]; lam = np.exp(lt[keep])
        c0 = ctrue(lam); e1.append(moment_center(x, E) - c0 if t == 0 else np.nan); e2.append(npmle_c(x, E, t)[0] - c0)
    print(f"E={E} sd={sd} t={t} n={len(x)}  factorial err {np.nanmean(e1):+.3f}±{np.nanstd(e1):.3f}   npmle-plugin err {np.mean(e2):+.3f}±{np.std(e2):.3f}")
