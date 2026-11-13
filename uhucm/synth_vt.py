"""Rating setting of EviGen with re-estimated hyperparameters: lower bound with the posterior variance only versus the
total variance of Definition 5, whose bootstrap redraws every instance's evidence with the instances kept
(observed mean ~ N(s_o, sigma^2/n_o), within-instance variance ~ sigma^2 chi2_{n_o-1}/(n_o-1)), re-estimates the
hyperparameters by the same moments and recomputes the posterior means of the observed evidence.
Writes out_synth/e2_vt.json.  Usage: python synth_vt.py [nseed]"""
import sys, math, json, numpy as np
from scipy.stats import norm
import fastmine
from synth import evigen2, problem, truth_sets, OUT


def estimate(s, vol, sv, feat, k):
    dfree = np.maximum(vol - 1, 0); sg = math.sqrt(float((dfree * sv).sum() / max(1, dfree.sum())))
    m0 = np.zeros(len(s)); tau = np.ones(len(s))
    for f in range(k):
        idx = feat == f
        if idx.sum() < 2: continue
        m0[idx] = float((vol[idx] * s[idx]).sum() / vol[idx].sum())
        tau[idx] = math.sqrt(max(0.05, float(s[idx].var(ddof=1)) - sg ** 2 * float((1.0 / vol[idx]).mean())))
    return sg, m0, tau


def post(s, vol, sg, m0, tau):
    prec = 1 / tau ** 2 + vol / sg ** 2
    return (m0 / tau ** 2 + vol * s / sg ** 2) / prec, 1 / prec


def boot(G, B=int(__import__("os").environ.get("BV", "20")), seed=0):
    rng = np.random.default_rng(seed); s, vol, feat, k = G["s"], G["vol"], G["feat"], G["k"]
    sg = G["sg"]; dfree = np.maximum(vol - 1, 0); U = np.zeros((B, len(s)))
    for b in range(B):
        sb = s + sg / np.sqrt(vol) * rng.normal(size=len(s))
        svb = np.where(dfree > 0, sg ** 2 * rng.chisquare(np.maximum(dfree, 1)) / np.maximum(dfree, 1), 0.0)
        sgb, m0b, taub = estimate(sb, vol, svb, feat, k)
        U[b] = post(s, vol, sgb, m0b, taub)[0]
    return U


nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
deltas = [0.01, 0.05, 0.1, 0.2]; acc = {f"{v} d={d}": [] for v in ("V", "VT") for d in deltas}
for seed in range(nseed):
    G = evigen2(400, 8, 1000 + seed, estimate=True)
    P0 = problem(G, G["pm"], G["pv"], 0.0)
    if P0.Z <= 0 or len(P0.cliques) == 0: continue
    R0all = fastmine.mine(P0, -1e9)[0]
    if len(R0all) < 20: continue
    tv = truth_sets(G, P0, R0all); tau = float(np.quantile(list(tv.values()), 0.7)); ntrue = sum(v >= tau for v in tv.values())
    U = boot(G, seed=seed)
    for d in deltas:
        for v in ("V", "VT"):
            Pb = problem(G, G["pm"], G["pv"], norm.ppf(1 - d))
            if v == "VT": Pb.set_boot(U)
            R = fastmine.mine(Pb, tau / Pb.Z)[0]
            acc[f"{v} d={d}"].append((len(R), sum(tv[p] < tau for p in R) / len(R) if R else 0.0, sum(tv[p] >= tau for p in R) / max(1, ntrue)))
out = {m: dict(seeds=len(L), FDP=round(float(np.mean([x[1] for x in L])), 4), FDP_sd=round(float(np.std([x[1] for x in L], ddof=1)), 4),
               power=round(float(np.mean([x[2] for x in L])), 4)) for m, L in acc.items()}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{OUT}/e2_vt_B{__import__('os').environ.get('BV', '20')}.json", "w"), indent=1)
