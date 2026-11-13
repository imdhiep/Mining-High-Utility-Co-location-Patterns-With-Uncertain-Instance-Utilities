"""Repeated timing on Foursquare with the compound EB posteriors and total variance (10 runs per variant, z = 1.645):
exact bound, mean bound and NoPrune at K in {30, 100, 300, 1000} for k = 20, 40, 60, 80 features, the ablation at k = 20,
K = 300, and d = 70 m at k = 20 (NoPrune timed once per configuration for k >= 40).
Usage: OMP_NUM_THREADS=1 python run_timing_ceb.py CITY [k ...]"""
import sys
import fsq_loader, paths
from suite import Suite
import os
city = sys.argv[1]; out = "out_timing_ceb"; ks = [int(x) for x in sys.argv[2:]] or [20, 40, 60, 80]
NPR = int(os.environ["NP_REPS"]) if os.environ.get("NP_REPS") else None
for nf in ks:
    rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, nfeat=nf, model="ceb"); UB, Zh = info.pop("UB"), info.pop("Zhat")
    S = Suite(f"CEB_{city}_k{nf}", rows, 50.0, out, zmain=1.645, log=lambda s: print(s, flush=True), UB=UB, Zhat=Zh)
    S.timing(Ks=(30, 100, 300, 1000), reps=10, abl_K=300 if nf == 20 else -1, np_once=nf > 20, np_reps=NPR); S.save()
    if nf == 20:
        S = Suite(f"CEB_{city}_k20_d70", rows, 70.0, out, zmain=1.645, log=lambda s: print(s, flush=True), UB=UB, Zhat=Zh)
        S.timing(Ks=(30, 300), reps=10, abl_K=-1); S.save()
print("DONE", flush=True)
