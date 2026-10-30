"""Synthetic experiments (EviGen): E1 exactness at scale, E2 calibration incl. misspecification, scalability, clique stress.
Usage: python synth.py <exp> [args]   exp in {e1, e2, scale_n, scale_k, stress}; writes JSON to out_synth/.
All randomness is seeded; seeds are recorded."""
import sys, os, json, time, math, random, statistics as st, collections
import numpy as np
from scipy.stats import norm
import fastmine, uhucm

OUT = "out_synth"; os.makedirs(OUT, exist_ok=True)


def evigen2(n, k, seed, d=1.0, cluster=True, tau=1.0, sig=3.0, neg_frac=0.35, vol_sigma=1.3, vol_med=4.0,
            noise="normal", prior_shift=0.0, sigma_err=1.0, spatial=0.0, ell=5.0, density=0.6, estimate=False,
            activity=False, lam0=2.0):
    """Instances in a Thomas-like cluster process (parents uniform, children within 1.5 d), features uniform.
    Latent q_i = m_f + tau*(sqrt(1-spatial) e_i + sqrt(spatial) GRF(x_i)); evidence volume log-normal;
    observed mean s_i = q_i + sig/sqrt(n_i) * eps, eps ~ N(0,1) or t3/sqrt(3).
    Posterior: Normal-Normal with prior mean m_f + prior_shift*tau, prior sd tau, noise sd sig*sigma_err."""
    rnd = np.random.default_rng(seed)
    side = math.sqrt(n / density) * d
    m_f = np.where(rnd.random(k) < neg_frac, rnd.normal(-0.6, 0.4, k), rnd.normal(0.8, 0.4, k))
    if cluster:
        npar = max(1, n // 8); par = rnd.uniform(0, side, (npar, 2))
        which = rnd.integers(0, npar, n); ang = rnd.uniform(0, 2 * math.pi, n); rad = 1.5 * d * np.sqrt(rnd.random(n))
        xy = par[which] + np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    else:
        xy = rnd.uniform(0, side, (n, 2))
    feat = rnd.integers(0, k, n)
    e = rnd.normal(0, 1, n)
    if spatial > 0:
        # smooth random field: sum of random Fourier features with length scale ell*d
        W = rnd.normal(0, 1 / (ell * d), (64, 2)); b = rnd.uniform(0, 2 * math.pi, 64)
        grf = np.sqrt(2 / 64) * np.cos(xy @ W.T + b).sum(1); grf = (grf - grf.mean()) / (grf.std() + 1e-12)
        e = math.sqrt(1 - spatial) * e + math.sqrt(spatial) * grf
    q = m_f[feat] + tau * e
    vol = np.maximum(1, np.round(vol_med * np.exp(rnd.normal(0, vol_sigma, n)))).astype(int)
    eps = rnd.normal(0, 1, n) if noise == "normal" else rnd.standard_t(3, n) / math.sqrt(3)
    s = q + sig / np.sqrt(vol) * eps
    if activity:
        # Def. 4 setting: latent log-rate l_o = ln(lam0) + m_f/2 + tau*e (log-normal heterogeneity, so the Gamma prior of
        # Def. 4 is misspecified), counts a_o ~ Poisson(exp(l_o)), exposure W = 1; posterior Gamma(alpha_f + a, beta_f + 1)
        # with alpha_f, beta_f by moments per feature, utility = posterior mean of ln(lam) minus its feature median;
        # truth = l_o minus the feature median of l.
        lrate = math.log(lam0) + 0.5 * m_f[feat] + tau * e
        cnt = rnd.poisson(np.exp(lrate))
        from scipy.special import digamma, polygamma
        pm = np.zeros(n); pv = np.zeros(n); qt = np.zeros(n); raw = np.zeros(n)
        for f in range(k):
            idx = feat == f
            if idx.sum() == 0: continue
            x = cnt[idx].astype(float); mm = x.mean(); vv = x.var(ddof=1) if idx.sum() > 1 else mm
            al = mm * mm / (vv - mm) if vv > mm and mm > 0 else 1e6; be = al / max(mm, 1e-9)
            pmf = digamma(al + x) - math.log(be + 1.0)
            pm[idx] = pmf - np.median(pmf); pv[idx] = polygamma(1, al + x)
            qt[idx] = lrate[idx] - np.median(lrate[idx]); lr = np.log(x + 0.5); raw[idx] = lr - np.median(lr)
        return dict(feat=feat, xy=xy, q=qt, s=raw, pm=pm, pv=pv, vol=cnt, k=k, d=d, m0=np.zeros(n))
    sg = sig * sigma_err; m0 = m_f[feat] + prior_shift * tau; tau_i = np.full(n, tau)
    if estimate:
        # hyperparameters re-estimated from the simulated evidence by the method of moments used on Yelp:
        # per-instance sample variances of the n_i observations give a pooled sigma; per feature, the evidence-weighted
        # mean gives the prior mean and Var(s) - sigma^2 E[1/n] (floored) gives the prior variance.
        dfree = np.maximum(vol - 1, 0)
        sv = np.where(dfree > 0, sig ** 2 * rnd.chisquare(np.maximum(dfree, 1)) / np.maximum(dfree, 1), 0.0)
        sg = math.sqrt(float((dfree * sv).sum() / max(1, dfree.sum())))
        for f in range(k):
            idx = feat == f
            if idx.sum() < 2: continue
            m0[idx] = float((vol[idx] * s[idx]).sum() / vol[idx].sum())
            tau_i[idx] = math.sqrt(max(0.05, float(s[idx].var(ddof=1)) - sg ** 2 * float((1.0 / vol[idx]).mean())))
    prec = 1 / tau_i ** 2 + vol / sg ** 2
    pm = (m0 / tau_i ** 2 + vol * s / sg ** 2) / prec; pv = 1 / prec
    return dict(feat=feat, xy=xy, q=q, s=s, pm=pm, pv=pv, vol=vol, k=k, d=d, m0=m0)


def problem(G, mu, s2, z):
    return fastmine.Problem(G["feat"], G["xy"], mu, s2, G["d"], z)


# ------------------------------------------------------------------ E1
def e1(nrep=2000):
    """Random small instances; UHUCM (every bound kind x 5 pruning combinations) vs reference brute force."""
    combos = [(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 1)]
    kinds = ("two", "exact", "chord", "mean")
    runs = mism = nonempty = 0; t0 = time.time()
    for seed in range(nrep):
        rnd = random.Random(seed)
        n, k = rnd.randint(8, 30), rnd.randint(4, 6)
        G = evigen2(n, k, seed, d=1.0, density=rnd.choice([0.5, 1.0, 2.0]))
        z = rnd.choice([0.0, 1.0, 1.645, 3.0]); rho = rnd.choice([0.0, 0.3, 0.6]); beta = rnd.choice([0.0, 1.0])
        ids = [f"o{i}" for i in range(n)]
        inst = [(ids[i], f"F{G['feat'][i]}", float(G['xy'][i, 0]), float(G['xy'][i, 1])) for i in range(n)]
        D0 = uhucm.CData(inst, dict(zip(ids, G["pm"])), dict(zip(ids, G["pv"])), {}, {}, 1.0, 0.0, z, rho)
        iota, ivar, pairs = {}, {}, {}
        if beta > 0:
            for i in range(n):
                for j in D0.nb[ids[i]]:
                    jj = ids.index(j)
                    if i < jj and rnd.random() < 0.6:
                        v, w = rnd.uniform(-1, 1), rnd.uniform(0, 0.3)
                        iota[frozenset((ids[i], j))] = v; ivar[frozenset((ids[i], j))] = w; pairs[(i, jj)] = (v, w)
        D = uhucm.CData(inst, dict(zip(ids, G["pm"])), dict(zip(ids, G["pv"])), iota, ivar, 1.0, beta, z, rho)
        if D.Z <= 0: continue
        mu = rnd.choice([0.0, 0.02, 0.05, 0.1, 0.2])
        bf, allv = uhucm.brute_force(D, mu); want = {tuple(sorted(P)) for P in bf}
        names = sorted({f for _, f, _, _ in inst})
        Pb = fastmine.Problem([names.index(f) for _, f, _, _ in inst], G["xy"], G["pm"], G["pv"], 1.0, z, rho, pairs, beta, names)
        for use in combos:
            for kind in kinds:
                r = fastmine.mine(Pb, mu, use, kind)[0]; runs += 1
                mism += {tuple(sorted(P)) for P in r} != want
        nonempty += bool(want)
    res = dict(instances=nrep, runs=runs, mismatches=mism, nonempty=nonempty, seconds=round(time.time() - t0, 1))
    print("E1", res); json.dump(res, open(f"{OUT}/e1.json", "w"))


# ------------------------------------------------------------------ E2
def truth_sets(G, P0, R0all):
    q = G["q"]; out = {}
    for p in R0all:
        A = fastmine.pattern_atoms(P0, p); out[p] = float(q[A].sum())
    return out


def e2(nseed=100, n=400, k=8, knobs=None):
    """FDP / detection rate of each method w.r.t. latent truth: a pattern is a true HUCP iff sum of latent q
    over its atoms >= tau_abs. tau_abs = 70th percentile of the true values of all patterns with RI != empty."""
    knobs = knobs or {"correct": {}, "t3": dict(noise="t3"), "prior+0.3": dict(prior_shift=0.3), "prior-0.3": dict(prior_shift=-0.3),
                      "sigma x0.7": dict(sigma_err=0.7), "sigma x1.3": dict(sigma_err=1.3), "spatial": dict(spatial=0.5),
                      "estimated": dict(estimate=True)}
    deltas = [0.01, 0.05, 0.1, 0.2]
    methods = {f"normal d={dl}": norm.ppf(1 - dl) for dl in deltas}
    methods.update({"subG d=0.05": math.sqrt(2 * math.log(1 / 0.05)), "Cantelli d=0.05": math.sqrt((1 - 0.05) / 0.05)})
    res = {}
    for kn, kw in knobs.items():
        acc = {m: [] for m in list(methods) + ["z=0 (shrunk)", "raw", "FE"]}
        for seed in range(nseed):
            G = evigen2(n, k, 1000 + seed, **kw)
            P0 = problem(G, G["pm"], G["pv"], 0.0)
            if P0.Z <= 0 or len(P0.cliques) == 0: continue
            R0all = fastmine.mine(P0, -1e9)[0]
            if len(R0all) < 20: continue
            tv = truth_sets(G, P0, R0all)
            tau = float(np.quantile(list(tv.values()), 0.7)); ntrue = sum(v >= tau for v in tv.values())
            def ev(mu, s2, z):
                Pb = problem(G, mu, s2, z)
                if Pb.Z <= 0: return None
                R = fastmine.mine(Pb, tau / Pb.Z)[0]
                fd = sum(tv[p] < tau for p in R)
                return (len(R), fd / len(R) if R else 0.0, sum(tv[p] >= tau for p in R) / max(1, ntrue))
            for m, z in methods.items(): acc[m].append(ev(G["pm"], G["pv"], z))
            acc["z=0 (shrunk)"].append(ev(G["pm"], G["pv"], 0.0))
            acc["raw"].append(ev(G["s"], np.zeros(n), 0.0))
            acc["FE"].append(ev(G["pm"] - 1.645 * np.sqrt(G["pv"]), np.zeros(n), 0.0))
        out = {}
        for m, L in acc.items():
            L = [x for x in L if x]
            out[m] = dict(seeds=len(L), reported=round(st.mean(x[0] for x in L), 2), FDP=round(st.mean(x[1] for x in L), 4),
                          FDP_sd=round(st.stdev(x[1] for x in L), 4), FDP_p90=round(float(np.quantile([x[1] for x in L], 0.9)), 4),
                          power=round(st.mean(x[2] for x in L), 4), power_sd=round(st.stdev(x[2] for x in L), 4))
        res[kn] = out
        print("E2", kn, json.dumps(out), flush=True)
    json.dump(dict(n=n, k=k, nseed=nseed, quantile=0.7, results=res), open(f"{OUT}/e2.json", "w"), indent=1)


# ------------------------------------------------------------------ scalability
def topk_mu(Pb, K):
    m = 0.05
    while True:
        R = fastmine.mine(Pb, m)[0]
        if len(R) >= K or m < 1e-9: return sorted(R.values(), reverse=True)[min(K, len(R)) - 1] if R else m
        m /= 2


def run_scale(n, k, seed, K=300, noprune=True, deadline=600):
    G = evigen2(n, k, seed)
    t = time.time(); P0 = problem(G, G["pm"], G["pv"], 0.0); tb = time.time() - t
    muK = topk_mu(P0, K)
    t = time.time(); Pz = problem(G, G["pm"], G["pv"], 1.645); tb2 = time.time() - t
    out = dict(n=n, k=k, seed=seed, cliques=len(P0.cliques), omega=max(map(len, P0.cliques)) if P0.cliques else 0,
               build_s=round(tb2, 2), mu=muK)
    for name, use in (("uhucm", (1, 1, 1, 1)), ("noprune", (0, 0, 0, 0))):
        if name == "noprune" and not noprune: continue
        try:
            t = time.time(); R, s, _ = fastmine.mine(Pz, muK, use, "two", deadline=deadline)
            out[name] = dict(time=round(time.time() - t, 2), evaluated=s["evaluated"], R=len(R))
        except fastmine.Timeout:
            out[name] = dict(time=f">{deadline}")
    print("SCALE", json.dumps(out), flush=True)
    return out


def scale_n():
    res = [run_scale(n, 10, 7) for n in (1000, 3000, 10000, 30000, 100000)]
    json.dump(res, open(f"{OUT}/scale_n.json", "w"), indent=1)


def scale_k():
    res = [run_scale(10000, k, 7) for k in (5, 10, 15, 20, 25, 30)]
    json.dump(res, open(f"{OUT}/scale_k.json", "w"), indent=1)


def stress():
    """Dense blocks: m_c instances of each of k features inside a disk of diameter d -> C ~ m_c^k cliques."""
    res = []
    for k in (4, 6, 8):
        for mc in (2, 3, 4, 5):
            if mc ** k > 400000: continue
            rnd = np.random.default_rng(k * 10 + mc)
            feat = np.repeat(np.arange(k), mc); m = len(feat)
            ang = rnd.uniform(0, 2 * math.pi, m); rad = 0.49 * np.sqrt(rnd.random(m))
            xy = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
            mu = rnd.normal(0.5, 1, m); s2 = rnd.uniform(0.01, 0.5, m)
            t = time.time(); Pb = fastmine.Problem(feat, xy, mu, s2, 1.0, 1.645); tb = time.time() - t
            t = time.time(); R, s, _ = fastmine.mine(Pb, 0.05); tm = time.time() - t
            out = dict(k=k, mc=mc, cliques=len(Pb.cliques), expected=mc ** k, build_s=round(tb, 2), mine_s=round(tm, 2), evaluated=s["evaluated"])
            print("STRESS", json.dumps(out), flush=True); res.append(out)
    json.dump(res, open(f"{OUT}/stress.json", "w"), indent=1)


# ------------------------------------------------------------------ E2b (revision): further calibration settings
def rho_hat(G, Pb):
    r = G["pm"] - G["m0"]; a, b = [], []
    for i, S in enumerate(Pb.nb):
        for j in S:
            if i < j: a.append(r[i]); b.append(r[j])
    a, b = np.array(a + b), np.array(b + a)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else 0.0


def e2b(nseed=100):
    """Revision settings. Each entry: (generator kwargs, n, k, truth quantile). Methods: normal z at four deltas,
    sub-Gaussian and Cantelli at 0.05, point utility, and (spatial settings) the robust measure at the estimated rho."""
    settings = {"activity (Def. 4)": (dict(activity=True), 400, 8, 0.7),
                "spatial 0.1": (dict(spatial=0.1), 400, 8, 0.7),
                "spatial 0.5": (dict(spatial=0.5), 400, 8, 0.7),
                "n=1000, k=10": ({}, 1000, 10, 0.7),
                "truth 10%": ({}, 400, 8, 0.9)}
    deltas = [0.01, 0.05, 0.1, 0.2]
    res = {}
    for name, (kw, n, k, qtl) in settings.items():
        acc = collections.defaultdict(list); rh = []
        for seed in range(nseed):
            G = evigen2(n, k, 1000 + seed, **kw)
            P0 = problem(G, G["pm"], G["pv"], 0.0)
            if P0.Z <= 0 or len(P0.cliques) == 0: continue
            R0all = fastmine.mine(P0, -1e9)[0]
            if len(R0all) < 20: continue
            tv = truth_sets(G, P0, R0all)
            tau = float(np.quantile(list(tv.values()), qtl)); ntrue = sum(v >= tau for v in tv.values())
            def ev(mu, s2, z, rho=0.0):
                Pb = fastmine.Problem(G["feat"], G["xy"], mu, s2, G["d"], z, rho)
                if Pb.Z <= 0: return None
                R = fastmine.mine(Pb, tau / Pb.Z)[0]
                return (len(R), sum(tv[p] < tau for p in R) / len(R) if R else 0.0, sum(tv[p] >= tau for p in R) / max(1, ntrue))
            for dl in deltas: acc[f"normal d={dl}"].append(ev(G["pm"], G["pv"], norm.ppf(1 - dl)))
            acc["subG d=0.05"].append(ev(G["pm"], G["pv"], math.sqrt(2 * math.log(20))))
            acc["Cantelli d=0.05"].append(ev(G["pm"], G["pv"], math.sqrt(19)))
            acc["z=0 (shrunk)"].append(ev(G["pm"], G["pv"], 0.0))
            if kw.get("spatial"):
                r = max(0.0, rho_hat(G, P0)); rh.append(r)
                acc["robust rho_hat d=0.05"].append(ev(G["pm"], G["pv"], norm.ppf(0.95), r))
        out = {"rho_hat_mean": round(st.mean(rh), 4)} if rh else {}
        for m, L in acc.items():
            L = [x for x in L if x]
            out[m] = dict(seeds=len(L), reported=round(st.mean(x[0] for x in L), 2), FDP=round(st.mean(x[1] for x in L), 4),
                          FDP_sd=round(st.stdev(x[1] for x in L), 4), FDP_p90=round(float(np.quantile([x[1] for x in L], 0.9)), 4),
                          power=round(st.mean(x[2] for x in L), 4), power_sd=round(st.stdev(x[2] for x in L), 4))
        res[name] = out
        print("E2b", name, json.dumps(out), flush=True)
        json.dump(dict(nseed=nseed, results=res), open(f"{OUT}/e2b.json", "w"), indent=1)


def e2c(nseed=100):
    """Activity posterior (Def. 4) combined with spatially correlated latent log-rates: FDP and detection at delta = 0.05
    for the normal (1.645) and sub-Gaussian (2.45) choices, with rho = 0 and with the robust measure at the estimated rho."""
    res = {}
    for frac in (0.1, 0.25, 0.5):
        acc = collections.defaultdict(list); rh = []
        for seed in range(nseed):
            G = evigen2(400, 8, 1000 + seed, activity=True, spatial=frac)
            P0 = problem(G, G["pm"], G["pv"], 0.0)
            if P0.Z <= 0 or len(P0.cliques) == 0: continue
            R0all = fastmine.mine(P0, -1e9)[0]
            if len(R0all) < 20: continue
            tv = truth_sets(G, P0, R0all)
            tau = float(np.quantile(list(tv.values()), 0.7)); ntrue = sum(v >= tau for v in tv.values())
            def ev(z, rho=0.0):
                Pb = fastmine.Problem(G["feat"], G["xy"], G["pm"], G["pv"], G["d"], z, rho)
                if Pb.Z <= 0: return None
                R = fastmine.mine(Pb, tau / Pb.Z)[0]
                return (len(R), sum(tv[p] < tau for p in R) / len(R) if R else 0.0, sum(tv[p] >= tau for p in R) / max(1, ntrue))
            r = max(0.0, rho_hat(G, P0)); rh.append(r)
            for zn, z in (("normal", norm.ppf(0.95)), ("subG", math.sqrt(2 * math.log(20)))):
                acc[f"{zn} rho=0"].append(ev(z)); acc[f"{zn} robust rho_hat"].append(ev(z, r))
            acc["z=0"].append(ev(0.0))
        out = {"rho_hat_mean": round(st.mean(rh), 4), "rho_hat_sd": round(st.stdev(rh), 4)}
        for m, L in acc.items():
            L = [x for x in L if x]
            out[m] = dict(seeds=len(L), FDP=round(st.mean(x[1] for x in L), 4), FDP_sd=round(st.stdev(x[1] for x in L), 4),
                          power=round(st.mean(x[2] for x in L), 4))
        res[f"activity+spatial {frac}"] = out
        print("E2c", frac, json.dumps(out), flush=True)
        json.dump(dict(nseed=nseed, results=res), open(f"{OUT}/e2c.json", "w"), indent=1)


if __name__ == "__main__":
    exp = sys.argv[1]
    {"e1": e1, "e2": e2, "e2b": e2b, "e2c": e2c, "scale_n": scale_n, "scale_k": scale_k, "stress": stress}[exp]()
