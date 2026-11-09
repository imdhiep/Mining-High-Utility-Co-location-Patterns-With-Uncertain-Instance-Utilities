"""Conditional (finite-population) error of the moment center: sd of c_hat over count redraws with the rates fixed,
versus venue-bootstrap sd and count-bootstrap sd (x* ~ Poisson(x)) and the delta-method sd."""
import sys, os, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], '1', sys.argv[2] if len(sys.argv) > 2 else 'lognormal']
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'semisyn_fsq.py')).read()
src = src.replace("base = Problem(F, XY, np.zeros(n), np.ones(n), DIST, 0.0, feat_names=fs)", "base = None")
src = src.replace('print("layout", city, "n", n, "k", len(fs), "cliques", len(base.cliques), flush=True)', '')
src = src[:src.index("out = {}\nfor f in FR:")]
exec(compile(src, 'semisyn', 'exec'))
def delta_sd(x, E):
    n_ = len(x); m = x.mean() / E; s = (x * (x - 1)).mean() / E ** 2
    f2 = x * (x - 1); f3 = x * (x - 1) * (x - 2)          # unbiased for mu^2, mu^3 (mu = lambda E)
    vm = x.sum() / (n_ * E) ** 2; cms = 2 * f2.sum() / (n_ ** 2 * E ** 3); vs = (4 * f3 + 2 * f2).sum() / (n_ ** 2 * E ** 4)
    var = 4 * vm / m ** 2 - 2 * cms / (m * s) + vs / (4 * s * s)
    return float(np.sqrt(max(var, 0)))
Wf = W
rng = np.random.default_rng(1000); lt = draw_truth(rng); lam = np.exp(lt)
for j in range(len(fs)):
    idx = F == j; L = lam[idx]; tru = momc_true(L)
    reps = np.array([momc(rng.poisson(L * Wf).astype(float), Wf) for _ in range(200)]); cond = reps.std(); bias = reps.mean() - tru
    x = rng.poisson(L * Wf).astype(float)
    vb = np.std([momc(x[rng.integers(0, len(x), len(x))], Wf) for _ in range(200)])
    cb = np.std([momc(rng.poisson(x).astype(float), Wf) for _ in range(200)])
    print(city, j, len(L), f"cond={cond:.4f} bias={bias:+.4f} venueboot={vb:.4f} countboot={cb:.4f} delta={delta_sd(x, Wf):.4f}", flush=True)
