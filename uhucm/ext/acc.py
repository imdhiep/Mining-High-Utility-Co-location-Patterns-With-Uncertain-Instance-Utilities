"""Accuracy of UHUCPM, CQ-HUCPM and PSO-MUCP on semi-synthetic check-ins with known utilities (Section VI-B).

Layout: the venues, features and locations of a real city (Foursquare NYC / TKY, or a Yelp city). True log-rates are drawn
per feature from a law fitted to the real counts, counts are drawn from Poisson(rate * E * c) over the real exposure, and
the true utility is the log-rate minus the moment center of the drawn rates. The true patterns are the K patterns with the
highest true score, and gamma is the K-th largest true score; every method is judged at this gamma.
  uhucpm  lower bound at z* = 1.645 with V_T (B = 20) and the expected normalizer, posterior of Definition (count utility)
  cq      the authors' CQ-HUCPM at z = 0 with the posterior means as instance utilities. Its answer set equals the point
          answer of the same search (exact miner); the point set is used for every seed and the authors' code is run on
          the first NCHK seeds of each setting, recording whether the sets agree
  pso     the authors' PSO-MUCP at z = 0 with one utility per feature (the mean posterior mean of the feature), run on
          every seed; its pruning is not exact, so its own output is used. When no feature has a positive utility, no
          pattern can reach gamma and its answer is empty (the program is not run, as it would divide by zero)
All three use the same normalizer Z, so a pattern is reported by each when its total reaches gamma * Z.
Usage:
  python acc.py LAYOUT VARY VALUES SEEDS        LAYOUT in NYC, TKY, Philadelphia, New_Orleans, Tampa
     VARY in K, d, k, frac, c, sd, law          VALUES comma-separated, SEEDS = number of seeds
  defaults K = 100, d = 50, k = 20, frac = 1, c = 1, sd = 1, law = lognormal; env KS (thresholds when VARY is not K,
  default 100), NCHK (default 3), YELP_DIR, REVIEW_DIR
  law: lognormal (normal log-rates), gamma (log of a Gamma rate with the same log-mean and log-sd) or mixture (two
  components, 30% of the venues 1.5 standard deviations higher), as in the calibration of the paper
-> out_acc/acc_<LAYOUT>_<VARY>.json, written after every seed"""
import sys, os, json, time, functools, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from scipy.special import digamma, gammaln
import paths
from fastmine import Problem, mine, pattern_atoms

ZS = 1.645; B = 20; NCHK = int(os.environ.get("NCHK", "3"))
ALPHA = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)] + [chr(0x4E00 + i) for i in range(200)]
CQ = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "CQ-HUCPM.py"); PSO = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "PSO-MUCP.py")
OUT = "out_acc"; os.makedirs(OUT, exist_ok=True)
layout, vary, vals, NS = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
DEF = dict(K=100, d=50.0, k=20, frac=1.0, c=1.0, sd=1.0, law="lognormal")
cast = dict(K=int, d=float, k=int, frac=float, c=float, sd=float, law=str)
vals = [cast[vary](v) for v in vals]
_cache = {}


def load_layout(k):
    """features (0..k-1, by decreasing number of venues), locations, real counts and exposure of the layout"""
    if k in _cache: return _cache[k]
    if layout in ("NYC", "TKY"):
        import fsq_loader
        rows = fsq_loader.load(paths.fsq(layout), layout, nfeat=k)[0]; W = float(rows[0]["W"])
    else:
        import yelp_full
        city = layout.replace("_", " ")
        rows = yelp_full.build(os.environ["YELP_DIR"], [city], split="2019-01-01", test_end="2020-01-01", log=lambda s: None,
                               review_dir=os.environ.get("REVIEW_DIR"))[city]["rows"]
        cnt = {}
        for r in rows: cnt[r["feat"]] = cnt.get(r["feat"], 0) + 1
        top = sorted(cnt, key=lambda f: (-cnt[f], f))[:k]; rows = [r for r in rows if r["feat"] in top]
        W = 1.0                                                   # counts over the training period, exposure 1
    cnt = {}
    for r in rows: cnt[r["feat"]] = cnt.get(r["feat"], 0) + 1
    order = sorted(cnt, key=lambda f: (-cnt[f], f)); fi = {f: j for j, f in enumerate(order)}
    F = np.array([fi[r["feat"]] for r in rows]); XY = np.array([(r["x"], r["y"]) for r in rows], float)
    a = np.array([r["a"] for r in rows], float)
    # law of the true log-rates: Gamma-Poisson posterior means of the real counts (moments per feature), over the
    # venues with at least one record (all venues on Foursquare; on Yelp the businesses without check-ins would give
    # log-rates near -1/A for a strongly overdispersed feature and an unbounded spread)
    lr = np.full(len(a), np.nan)
    for j in range(len(order)):
        x = a[F == j]; m, v = x.mean(), x.var(ddof=1)
        A = m * m / (v - m) if v > m else 1e6; be = A * W / max(m, 1e-9)
        lr[F == j] = np.where(x >= 1, digamma(A + x) - np.log(be + W), np.nan)
    loc = np.array([np.nanmean(lr[F == j]) for j in range(len(order))]); scl = np.array([np.nanstd(lr[F == j]) for j in range(len(order))])
    _cache[k] = (F, XY, W, loc, scl, order); return _cache[k]


def momc(x, E):
    m = x.mean() / E; v = max((x * (x - 1)).mean() / E ** 2 - m * m + m / (len(x) * E), 0.0)
    return float(np.log(m) - 0.5 * np.log1p(v / (m * m)))


def momc_true(lam):
    m = lam.mean(); v = lam.var(); return float(np.log(m) - 0.5 * np.log1p(v / (m * m)))


def posterior(cnt, F, E, J=300, iters=300):
    """NPMLE prior per feature on a grid of log-rates, moment center, posterior mean / variance and E[u+]"""
    n = len(cnt); m = np.zeros(n); v = np.zeros(n); ep = np.zeros(n); G = {}
    for j in np.unique(F):
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        lo = np.log((x.min() + 0.5) / E) - 2; hi = np.log((x.max() + 0.5) / E) + 1
        th = np.linspace(lo, hi, J); ll = x[:, None] * th[None, :] - E * np.exp(th)[None, :] - gammaln(x + 1)[:, None]
        L = np.exp(ll - ll.max(1, keepdims=True)); w = np.full(J, 1.0 / J)
        for _ in range(iters):
            post = L * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
        post = L * w; post /= post.sum(1, keepdims=True); c = momc(x, E)
        mm = post @ th; m[idx] = mm - c; v[idx] = np.maximum(post @ th ** 2 - mm ** 2, 1e-12)
        ep[idx] = post @ np.clip(th - c, 0, None); G[j] = (th, L, w)
    return m, v, float(ep.sum()), G


def boot(cnt, F, E, G, rng, iters=40):
    """V_T: counts redrawn from Poisson(x), venues fixed; prior and center estimated again; posterior means of the observed x"""
    U = np.zeros((B, len(cnt)))
    for j, (th, L, w0) in G.items():
        idx = np.nonzero(F == j)[0]; x = cnt[idx]
        for b in range(B):
            xb = rng.poisson(x).astype(float)
            llb = xb[:, None] * th[None, :] - E * np.exp(th)[None, :]; Lb = np.exp(llb - llb.max(1, keepdims=True)); w = w0.copy()
            for _ in range(iters):
                post = Lb * w; post /= post.sum(1, keepdims=True); w = np.maximum(post.mean(0), 1e-300)
            post = L * w; post /= post.sum(1, keepdims=True)
            U[b, idx] = post @ th - (momc(xb, E) if xb.sum() > 0 else 0.0)
    return U


def with_values(base, mu, s2, Z=None):
    P = Problem.__new__(Problem); P.__dict__.update(base.__dict__)
    amu = np.asarray(mu, float); as2 = np.asarray(s2, float); P.amu, P.as2, P.z, P.UB = amu, as2, 0.0, None
    P.Z = float(np.clip(amu, 0, None).sum()) if Z is None else float(Z)
    posi = [i for i in range(len(amu)) if amu[i] > 0]
    cmp = lambda i, j: int(as2[i] * amu[j] > as2[j] * amu[i]) - int(as2[i] * amu[j] < as2[j] * amu[i])
    posi.sort(key=functools.cmp_to_key(cmp))
    P.rank = -np.ones(len(amu), dtype=np.int64); P.rank[posi] = np.arange(len(posi)); P.smu = amu[posi]; P.ss2 = as2[posi]
    return P


def topk(P, K, start=0.05):
    m = start
    while True:
        R = mine(P, m)[0]
        if len(R) >= K or m < 1e-12: return R, sorted(R.values(), reverse=True)
        m /= 2


def ext(cmd):
    p = subprocess.run([sys.executable, "-I"] + cmd, capture_output=True, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr[-2000:])
    r = json.loads(p.stdout.strip().splitlines()[-1]); return set(r["patterns"]), r["seconds"]


def key(p): return "".join(sorted(p))


res = dict(layout=layout, vary=vary, values=vals, defaults=DEF, z=ZS, B=B, nchk=NCHK, runs={})
path = os.path.join(OUT, f"acc_{layout}_{vary}.json")
settings = [dict(DEF, **{vary: v}) for v in vals] if vary != "K" else [dict(DEF)]
Ks = vals if vary == "K" else [int(x) for x in os.environ.get("KS", str(DEF["K"])).split(",")]
for st in settings:
    F0, XY0, W, loc, scl, order = load_layout(st["k"])
    tagv = st[vary] if vary != "K" else "all"
    base_full = None if st["frac"] < 1 else Problem(F0, XY0, np.zeros(len(F0)), np.ones(len(F0)), st["d"], 0.0, feat_names=ALPHA[:len(order)])
    for seed in range(NS):
        t0 = time.time(); rng = np.random.default_rng(1000 + seed)
        sel = np.sort(rng.choice(len(F0), int(round(st["frac"] * len(F0))), replace=False)) if st["frac"] < 1 else np.arange(len(F0))
        F, XY = F0[sel], XY0[sel]; n = len(F)
        base = base_full if base_full is not None else Problem(F, XY, np.zeros(n), np.ones(n), st["d"], 0.0, feat_names=ALPHA[:len(order)])
        s_ = st["sd"] * scl[F]
        if st["law"] == "gamma":
            kk = 1.0 / np.maximum(s_ ** 2, 1e-3); lt = np.log(rng.gamma(kk, 1.0)) - digamma(kk) + loc[F]
        elif st["law"] == "mixture":
            hi_ = rng.random(n) < 0.3; lt = rng.normal(loc[F] - 0.45 * s_ + hi_ * 1.5 * s_, 0.7 * s_)
        else:
            lt = loc[F] + s_ * rng.normal(size=n)
        ut = lt - np.array([momc_true(np.exp(lt[F == j])) if (F == j).any() else 0.0 for j in range(len(order))])[F]
        E = W * st["c"]; cnt = rng.poisson(np.exp(lt) * E).astype(float)
        mu, s2, Z, G = posterior(cnt, F, E); UB = boot(cnt, F, E, G, np.random.default_rng(seed))
        Pt = with_values(base, ut, np.zeros(n)); Rt, sct = topk(Pt, max(Ks))
        P0 = with_values(base, mu, s2, Z)
        gs = {K: sct[K - 1] for K in Ks if len(sct) >= K}
        if not gs:
            print(layout, vary, tagv, seed, "fewer true patterns than K:", len(sct), flush=True)
            res["runs"][f"{st[vary] if vary != 'K' else 'all'}|{seed}"] = dict(skipped=True, n_true_patterns=len(sct)); continue
        R0 = mine(P0, min(gs.values()))[0]
        fv = {j: float(mu[F == j].mean()) for j in np.unique(F)}; mu_fl = np.array([fv[j] for j in F])
        Pf = with_values(base, mu_fl, np.zeros(n), Z)
        # files for the authors' programs (instance ids: feature letter + running number)
        tag = f"{layout}_{vary}_{tagv}_{seed}"; csvp = os.path.abspath(os.path.join(OUT, tag + ".csv")); ids = []; cc = {}
        with open(csvp, "w", encoding="utf-8") as f:
            f.write("Feature,Instance,LocX,LocY,Checkin\n")
            for j, (x, y) in zip(F.tolist(), XY):
                cc[j] = cc.get(j, 0) + 1; ids.append(f"{ALPHA[j]}{cc[j]}"); f.write(f"{ALPHA[j]},{cc[j]},{x:.4f},{y:.4f},0\n")
        uj = os.path.abspath(os.path.join(OUT, tag + "_u.json")); json.dump({i: float(x) for i, x in zip(ids, mu)}, open(uj, "w"))
        Upos = float(np.clip(mu, 0, None).sum()); Ufl = float(np.clip(mu_fl, 0, None).sum())
        util_fl = repr({ALPHA[j]: v for j, v in fv.items()})
        for K, g in gs.items():
            T = {key(p) for p, v in Rt.items() if v >= g}
            pool = [p for p, v in R0.items() if v >= g]                       # point answer = CQ-HUCPM answer
            As = [pattern_atoms(P0, p) for p in pool]
            M = np.array([mu[x].sum() for x in As]); V = np.array([s2[x].sum() + UB[:, x].sum(1).var(ddof=1) for x in As])
            S_u = {key(p) for p, m_, v_ in zip(pool, M, V) if m_ - ZS * np.sqrt(v_) >= g * Z}
            S_cq = {key(p) for p in pool}
            if Ufl > 0:
                S_pso, t_pso = ext([os.path.join(HERE, "run_pso.py"), PSO, csvp, util_fl, repr(float(g * Z / Ufl * (1 - 1e-9))), str(st["d"])])
            else:                                                 # no feature has a positive utility: no pattern reaches gamma
                S_pso, t_pso = set(), None
            S_fl = {key(p) for p, v in mine(Pf, g * (1 - 1e-12))[0].items()}  # exact feature-level answer, for reference
            chk = None
            if seed < NCHK:
                S_cq_ext, t_cq = ext([os.path.join(HERE, "run_cq.py"), CQ, csvp, uj, repr(float(g * Z / Upos * (1 - 1e-9))), str(st["d"])])
                chk = dict(equal=S_cq_ext == S_cq, n_ext=len(S_cq_ext), seconds=t_cq)

            def fd(S): k_ = len(S); return dict(n=k_, fdp=round(1 - len(S & T) / k_, 4) if k_ else 0.0, det=round(len(S & T) / len(T), 4))
            rec = dict(nT=len(T), gamma=float(g), uhucpm=fd(S_u), cq=fd(S_cq), pso=fd(S_pso), pso_exact_fl=fd(S_fl),
                       pso_missed=len(S_fl - S_pso), cq_check=chk, pso_seconds=t_pso, n=n, sec=round(time.time() - t0, 1))
            rk = f"{K}|{seed}" if vary == "K" else (f"{st[vary]}|{seed}" if len(Ks) == 1 else f"{st[vary]}|{K}|{seed}")
            res["runs"][rk] = rec
            print(layout, vary, st[vary] if vary != "K" else K, seed, json.dumps({m: rec[m] for m in ("uhucpm", "cq", "pso")}), chk, flush=True)
        for fn in (csvp, uj): os.remove(fn)
        json.dump(res, open(path, "w"), indent=1)
print("DONE", flush=True)
