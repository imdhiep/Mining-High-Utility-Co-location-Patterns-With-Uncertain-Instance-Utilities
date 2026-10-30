"""EviGen (synthetic): instances with latent utility q_i, evidence volume n_i (heavy-tailed), noisy evidence
s_i ~ N(q_i, sig^2/n_i); Normal-Normal posterior shrinks towards the feature prior mean (empirical Bayes, known tau).
Returns instances, latent truth q, and posterior (mean, var) for three estimators:
  'post'  : posterior mean / variance (used with z>0 by UHUCM)
  'point' : posterior mean, variance 0 (point-estimate HUCPM on shrunk means)
  'raw'   : raw sample mean s_i, variance 0 (no shrinkage, 'FE' / naive)
"""
import math, random

def evigen(n, k, side, seed, tau=1.0, sig=3.0, neg_frac=0.35, vol_sigma=1.3, vol_med=4.0):
    rnd = random.Random(seed)
    feats = [f"F{j}" for j in range(k)]
    m_f = {f: (rnd.gauss(-0.6, 0.4) if rnd.random() < neg_frac else rnd.gauss(0.8, 0.4)) for f in feats}
    inst, q, vol, raw, post_m, post_v = [], {}, {}, {}, {}, {}
    for i in range(n):
        f = feats[rnd.randrange(k)]; iid = f"o{i}"
        x, y = rnd.uniform(0, side), rnd.uniform(0, side)
        inst.append((iid, f, x, y))
        q[iid] = rnd.gauss(m_f[f], tau)
        vol[iid] = max(1, int(round(vol_med * math.exp(rnd.gauss(0, vol_sigma)))))
        s = rnd.gauss(q[iid], sig / math.sqrt(vol[iid]))
        raw[iid] = s
        prec = 1 / tau**2 + vol[iid] / sig**2
        post_v[iid] = 1 / prec
        post_m[iid] = (m_f[f] / tau**2 + vol[iid] * s / sig**2) / prec
    return dict(inst=inst, q=q, vol=vol, raw=raw, post_m=post_m, post_v=post_v, m_f=m_f)
