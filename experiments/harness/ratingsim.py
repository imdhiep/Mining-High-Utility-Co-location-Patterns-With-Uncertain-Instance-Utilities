"""Semi-synthetic star ratings with known utilities on a Yelp layout (Section VI-B.2, check of the rating model (4)).

The layout gives every business its feature, location, training check-ins a(o) and training ratings (number n_o, sum
S_o, sum of squares SS_o). From the real ratings of the layout:
  sigma_s^2  pooled within-business variance (businesses with at least two ratings);
  m_f, tau_f^2  review-weighted mean rating of feature f, and the variance of its business means minus their mean
             sampling variance (floor TAU_FLOOR), as in the rating model.
For one seed:
  theta_o    ~ N(m_f, tau_f^2), the latent quality of every business;
  ratings    n_o of them (the real number, times c and rounded), each round(N(theta_o, sigma_s^2)) clipped to 1..5 stars:
             integer and bounded, unlike the normal ratings the model assumes;
  true mean  r_o = E[star | theta_o], the mean rating of the business over infinitely many reviews (exact, from the
             normal probabilities of the five bins);
  true utility  ln(1 + a(o)) (r_o - 3) / 2, the utility (4) with the true mean rating."""
import numpy as np
from scipy.stats import norm

TAU_FLOOR = 0.05
NEUTRAL = 3.0
CUTS = np.array([1.5, 2.5, 3.5, 4.5])


def fit(F, n, S, SS):
    """sigma_s^2, m_f and tau_f^2 of the real ratings"""
    many = n >= 2
    sigma2 = float((SS[many] - S[many] ** 2 / n[many]).sum() / (n[many] - 1).sum())
    k = int(F.max()) + 1
    m = np.zeros(k); tau2 = np.full(k, TAU_FLOOR)
    for j in range(k):
        rated = (F == j) & (n >= 1)
        if not rated.any():
            m[j] = NEUTRAL; continue
        m[j] = S[rated].sum() / n[rated].sum()
        means = S[rated] / n[rated]
        if rated.sum() > 1:
            tau2[j] = max(means.var(ddof=1) - (sigma2 / n[rated]).mean(), TAU_FLOOR)
    return sigma2, m, tau2


def expected_star(theta, sigma):
    """E[clip(round(N(theta, sigma^2)), 1, 5)]"""
    cdf = norm.cdf((CUTS[None, :] - theta[:, None]) / sigma)            # P(star <= 1), ..., P(star <= 4)
    return 5.0 - cdf.sum(1)


def draw(F, xy, a, n, S, SS, setting, seed):
    """returns dict(F, xy, utility, stars, counts, exposure, seed); counts = training check-ins (the activity weight)"""
    rng = np.random.default_rng(1000 + seed)
    sigma2, m, tau2 = fit(F, n, S, SS)
    sigma = np.sqrt(sigma2)
    theta = rng.normal(m[F], np.sqrt(tau2[F]))
    reviews = np.rint(n * setting["c"]).astype(int)
    stars = [np.clip(np.rint(rng.normal(t, sigma, size=r)), 1, 5).astype(int).tolist() for t, r in zip(theta, reviews)]
    w = np.log1p(np.maximum(a, 0.0))
    utility = w * (expected_star(theta, sigma) - NEUTRAL) / 2
    return dict(F=F, xy=xy, utility=utility, stars=stars, counts=np.maximum(a, 0.0), exposure=1.0, seed=seed,
                sigma2=sigma2)
