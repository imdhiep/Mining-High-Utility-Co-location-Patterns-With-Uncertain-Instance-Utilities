"""Revision extras: (1) ablation on EviGen with negative-only features, where FUB and the non-positive tier act;
(2) kappa_iota sensitivity of the interaction atoms on Foursquare. Writes out_synth/rev_extra.json."""
import json, time, functools, sys, tempfile
import numpy as np
import synth, fastmine, fsq_loader, paths
from suite import Suite, jac, ZMAIN

res = {"ablation": [], "kappa": {}}
for seed in (7, 8, 9):
    G = synth.evigen2(10000, 12, seed, neg_frac=0.5, tau=0.3)
    Pz = synth.problem(G, G["pm"], G["pv"], 1.645); P0 = synth.problem(G, G["pm"], G["pv"], 0.0)
    negonly = int(sum(1 for f in range(12) if (G["pm"][G["feat"] == f] <= 0).all()))
    for K in (30, 300):
        muK = synth.topk_mu(P0, K); row = dict(seed=seed, K=K, negonly=negonly)
        for name, use in (("full", (1, 1, 1, 1)), ("noFUB", (0, 1, 1, 1)), ("noPCB", (1, 0, 1, 1)), ("noSB", (1, 1, 0, 1)),
                          ("noNP", (1, 1, 1, 0)), ("noprune", (0, 0, 0, 0))):
            t = time.time(); R, s, _ = fastmine.mine(Pz, muK, use, "exact"); row[name] = dict(nodes=s["evaluated"], time=round(time.time() - t, 3), R=len(R))
        print("ABL", json.dumps(row), flush=True); res["ablation"].append(row)
for c in ("NYC", "TKY"):
    rows, users, NU, info = fsq_loader.load(paths.fsq(c), c)
    S = Suite("x", rows, 50.0, tempfile.gettempdir(), log=lambda s: None)
    P0 = S.mk(); R0all, sc = S.topk(P0, 100); A0 = {p for p, v in R0all.items() if v >= sc[99]}
    out = {}
    for kap in (2.5, 5.0, 10.0):
        pairs = fsq_loader.interactions(rows, users, NU, P0.nb, kappa_i=kap)
        Pb0 = S.mk(pairs=pairs, beta=1.0); Pb1 = S.mk(z=ZMAIN, pairs=pairs, beta=1.0)
        Rb0, scb = S.topk(Pb0, 100); mK = scb[99]; B0 = {p for p, v in Rb0.items() if v >= mK}; B1 = set(fastmine.mine(Pb1, mK)[0])
        out[str(kap)] = dict(R0=len(B0), Rz=len(B1), J_vs_beta0=jac(B0, A0), kept=round(len(B1) / max(1, len(B0)), 3))
    print("KAP", c, json.dumps(out), flush=True); res["kappa"][c] = out
json.dump(res, open("out_synth/rev_extra.json", "w"), indent=1)
