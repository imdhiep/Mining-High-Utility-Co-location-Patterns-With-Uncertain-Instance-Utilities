import numpy as np, sys, os
sys.path.insert(0, '.')
from ceb import *
rng = np.random.default_rng(0)
F = np.repeat([0, 1], [800, 400]); lt = np.where(F == 0, rng.normal(0, 1.2, 1200), rng.normal(1, 0.8, 1200)); E = 3.0
lam = np.exp(lt)
def ctrue(l): m = l.mean(); s = (l ** 2).mean(); return 2 * np.log(m) - 0.5 * np.log(s)
# conditional sd of center vs delta method
reps = [moment_center(rng.poisson(lam[F == 0] * E), E) for _ in range(300)]
x = rng.poisson(lam * E).astype(float)
print("center err", moment_center(x[F == 0], E) - ctrue(lam[F == 0]), "cond sd", np.std(reps), "delta", center_sd(x[F == 0], E), "bias", np.mean(reps) - ctrue(lam[F == 0]))
mu, s2, ep, st = ceb_posterior(x, E, F)
U = cboot(x, E, st, B=30)
atoms = np.nonzero(F == 0)[0][:200]
# conditional variance of the bootstrap part: redraw counts with lam fixed, recompute posterior means for fixed x? (center part dominates)
print("V posterior", s2[atoms].sum(), "boot part", U[:, atoms].sum(1).var(ddof=1), "n*center var", (200 * center_sd(x[F == 0], E)) ** 2)
sel, p = bfdr_select(np.array([5, 1, -1, 3.0]), np.array([1, 1, 1, 1.0]), 0.0, 0.1); print(sel, p.round(3))
