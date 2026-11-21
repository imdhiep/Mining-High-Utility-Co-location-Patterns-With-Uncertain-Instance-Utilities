"""Semi-synthetic check-ins with known utilities (Section VI-A). For one layout and one setting:
  true log-rates  law "lognormal" (normal), "gamma" (log of a Gamma rate with the same log-mean and log-sd) or
                  "mixture" (30% of the instances 1.5 standard deviations higher); scale multiplied by `sd`; spatial
                  correlation over `ell` metres (Gaussian random field, squared-exponential kernel, random Fourier
                  features) when ell > 0, with the same marginal law
  counts          Poisson(rate * E * c), or negative binomial with overdispersion phi (Gamma-Poisson, variance
                  mean + phi mean^2) when phi > 0
  truncation t    instances with fewer than t records are removed (the real Foursquare pipeline keeps only venues with
                  a check-in, t = 1)
  true utility    true log-rate minus the moment center of the true rates of the feature, over all its instances
                  (also those removed by the truncation: the center of Definition 4 is that of the whole population)
  frac            share of the instances kept (drawn once per seed)"""
import numpy as np
from scipy.special import digamma

FOURIER_FEATURES = 500


def moment_center_of_rates(rates):
    m = rates.mean(); v = rates.var()
    return float(np.log(m) - 0.5 * np.log1p(v / (m * m)))


def spatial_field(xy, ell, rng):
    """zero-mean, unit-variance Gaussian field with kernel exp(-r^2 / (2 ell^2)) at the given points"""
    w = rng.normal(0.0, 1.0 / ell, size=(FOURIER_FEATURES, 2)); b = rng.uniform(0, 2 * np.pi, FOURIER_FEATURES)
    return np.sqrt(2.0 / FOURIER_FEATURES) * np.cos(xy @ w.T + b).sum(1)


def draw(layout, loc, scale, setting, seed):
    """returns dict(F, xy, log_rate, utility, counts, exposure) for the kept instances"""
    rng = np.random.default_rng(1000 + seed)
    F0 = layout.feature_index(); xy0 = layout.xy
    keep = np.sort(rng.choice(len(F0), int(round(setting["frac"] * len(F0))), replace=False)) if setting["frac"] < 1 else np.arange(len(F0))
    F, xy = F0[keep], xy0[keep]; n = len(F)
    s = setting["sd"] * scale[F]
    g = spatial_field(xy, setting["ell"], rng) if setting["ell"] > 0 else rng.normal(size=n)
    if setting["law"] == "gamma":
        from scipy.stats import gamma as gamma_law, norm
        k = 1.0 / np.maximum(s ** 2, 1e-3)
        lt = np.log(gamma_law.ppf(norm.cdf(g), k)) - digamma(k) + loc[F]
    elif setting["law"] == "mixture":
        high = rng.random(n) < 0.3
        lt = loc[F] - 0.45 * s + high * 1.5 * s + 0.7 * s * g
    else:
        lt = loc[F] + s * g
    exposure = layout.exposure * setting["c"]
    mean_count = np.exp(lt) * exposure
    if setting["phi"] > 0:
        mean_count = mean_count * rng.gamma(1.0 / setting["phi"], setting["phi"], size=n)
    counts = rng.poisson(mean_count).astype(float)
    centers = np.array([moment_center_of_rates(np.exp(lt[F == j])) if (F == j).any() else 0.0 for j in range(int(F0.max()) + 1)])
    kept = counts >= setting["t"]
    F, xy, lt, counts = F[kept], xy[kept], lt[kept], counts[kept]
    return dict(F=F, xy=xy, log_rate=lt, utility=lt - centers[F], counts=counts, exposure=exposure, seed=seed)
