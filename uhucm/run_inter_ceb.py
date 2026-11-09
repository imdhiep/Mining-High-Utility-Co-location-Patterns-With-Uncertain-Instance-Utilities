"""Interaction atoms (E7) and sensitivity of the compound EB model (bootstrap size B, grid size). Usage: python run_inter_ceb.py CITY"""
import sys, json, numpy as np
import fsq_loader, paths
from suite import Suite, jac
from fastmine import mine
c = sys.argv[1]
rows, users, NU, info = fsq_loader.load(paths.fsq(c), c, model="ceb"); UB, Zh = info.pop("UB"), info.pop("Zhat")
S = Suite(f"CEB_{c}_inter", rows, 50.0, "out_suite_ceb", users=users, NU=NU, interactions=fsq_loader.interactions,
          resid=[r["mu"] for r in rows], log=lambda s: print(s, flush=True), zmain=1.645, UB=UB, Zhat=Zh)
S.quality(Ks=(100, 300)); S.interaction(); S.save()
# sensitivity: B = 50 bootstrap draws (another seed) and a 600-point grid, answer sets at K = 300 versus the default
muK = S.sc[299]; base = set(mine(S.mk(z=1.645), muK)[0]); out = {}
for tag, kw in (("B50", dict(B=50, seed=2)), ("seed3", dict(B=20, seed=3))):
    r2, _, _, i2 = fsq_loader.load(paths.fsq(c), c, model="ceb", **kw)
    S2 = Suite("tmp", r2, 50.0, "out_suite_ceb", log=lambda s: None, zmain=1.645, UB=i2["UB"], Zhat=i2["Zhat"])
    R = set(mine(S2.mk(z=1.645), muK)[0]); out[tag] = dict(J=jac(base, R), n=len(R), n0=len(base))
    print("SENS", c, tag, out[tag], flush=True)
json.dump(out, open(f"out_suite_ceb/sens_{c}.json", "w"), indent=1)
