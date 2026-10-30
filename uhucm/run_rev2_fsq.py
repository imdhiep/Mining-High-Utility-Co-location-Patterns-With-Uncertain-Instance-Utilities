"""Foursquare (D4/D5) with the calibrated sub-Gaussian choice z = 2.45 for the activity posterior: quality at d = 50 m
(kept/rejected, hold-out with matched pairs, filters with score-matched comparison, FE), interactions, sensitivity and
dependence; quality at d = 30 and 70 m; and quality at z = 1.645 for comparison.
Usage: python run_rev2_fsq.py CITY"""
import sys, functools
import fsq_loader, paths
from suite import Suite
city = sys.argv[1]; out = "out_rev2"
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city)
kw = dict(recompute=functools.partial(fsq_loader.recompute, rows),
          variants=[("alpha_x0.5", dict(alpha_mult=0.5)), ("alpha_x2", dict(alpha_mult=2.0)), ("quasi", dict(quasi=True))],
          users=users, NU=NU, interactions=fsq_loader.interactions, resid=[r["mu"] for r in rows], log=lambda s: print(s, flush=True))
S = Suite(f"FSQ_{city}_z2.45", rows, 50.0, out, zmain=2.45, **kw)
S.quality(); S.interaction(); S.sensitivity(); S.dependence(); S.save()
S = Suite(f"FSQ_{city}_z1.645", rows, 50.0, out, zmain=1.645, **kw); S.quality(); S.save()
for d in (30.0, 70.0):
    S = Suite(f"FSQ_{city}_z2.45_d{int(d)}", rows, d, out, zmain=2.45, **kw); S.quality(); S.save()
print("DONE", flush=True)
