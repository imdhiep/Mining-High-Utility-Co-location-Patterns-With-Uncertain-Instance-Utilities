"""Effect of the second term of the total variance V_T on real check-in data: for D4/D5 with the compound empirical
Bayes model, the same posterior means, the same expected normalizer and z = 1.645, the answer set with V_T is compared
with the answer set when the pattern variance is the sum of the instance variances only. Reported per K: sizes, the
point-utility HUCPs rejected only because of the second term, their median number of participating instances, and
the test-period hold rates of the patterns kept with V_T, rejected only by the second term, and rejected by both.
Usage: python run_vt_only.py CITY   -> out_vt_only/vt_only_<CITY>.json"""
import sys, os, json, statistics as st
import numpy as np
import fsq_loader, paths
from suite import Suite
from fastmine import mine, pattern_atoms

city = sys.argv[1]; Z_ = 1.645
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, model="ceb")
UB, Zhat = info.pop("UB"), info.pop("Zhat")
S = Suite("VTONLY_" + city, rows, 50.0, "out_vt_only", zmain=Z_, UB=UB, Zhat=Zhat, log=lambda s: print(s, flush=True))
P0 = S.mk()                                        # z = 0 with the expected normalizer
PT = S.mk(z=Z_)                                    # V_T: instance variances plus the bootstrap term
PP = S.mk(mu=S.mu, s2=S.s2, z=Z_); PP.Z = float(Zhat)   # instance variances only, same means and normalizer
R0all, sc = Suite.topk(P0, 1000)
has_t = ~np.isnan(S.mut); Zt = float(np.clip(S.mut[has_t], 0, None).sum())
out = {}
for K in (100, 300, 1000):
    if K > len(sc): continue
    g = sc[K - 1]
    R0 = {p for p, v in R0all.items() if v >= g}
    RT = set(mine(PT, g)[0]); RP = set(mine(PP, g)[0])
    A = {p: pattern_atoms(P0, p) for p in R0}
    te = lambda p: float(np.nansum(S.mut[A[p]][has_t[A[p]]]) / Zt) >= g
    hold = lambda Ps: round(float(np.mean([te(p) for p in Ps])), 4) if Ps else None
    only_vt = (RP - RT) & R0; both = R0 - RP
    out[K] = dict(gamma=g, R0=len(R0), R_T=len(RT), R_P=len(RP), subset=RT <= RP,
                  only_vt=len(only_vt), only_vt_frac=round(len(only_vt) / len(R0), 4),
                  atoms_only_vt=float(st.median(len(A[p]) for p in only_vt)) if only_vt else None,
                  atoms_kept=float(st.median(len(A[p]) for p in RT & R0)) if RT & R0 else None,
                  hold_kept=hold(RT & R0), hold_only_vt=hold(only_vt), hold_both=hold(both))
    print(city, K, json.dumps(out[K]), flush=True)
os.makedirs("out_vt_only", exist_ok=True)
json.dump(dict(city=city, z=Z_, results=out), open(f"out_vt_only/vt_only_{city}.json", "w"), indent=1)
print("DONE", flush=True)
