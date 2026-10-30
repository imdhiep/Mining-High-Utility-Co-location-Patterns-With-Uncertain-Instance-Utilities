"""Repeated timing on Foursquare (10 runs per variant, z = 1.645 as on the Yelp data): exact bound, mean bound and
NoPrune at K in {30, 100, 300, 1000, 3000}, the ablation at K = 300, and (Tokyo) the dense setting d = 70 m.
Usage: OMP_NUM_THREADS=1 python run_timing_fsq.py CITY"""
import sys
import fsq_loader, paths
from suite import Suite
city = sys.argv[1]; out = "out_timing"
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city)
S = Suite(f"FSQ_{city}_timing", rows, 50.0, out, zmain=1.645, log=lambda s: print(s, flush=True))
S.timing(Ks=(30, 100, 300, 1000, 3000), reps=10); S.save()
if city == "TKY":
    S = Suite(f"FSQ_{city}_timing_d70", rows, 70.0, out, zmain=1.645, log=lambda s: print(s, flush=True))
    S.timing(Ks=(30, 300), reps=10); S.save()
print("DONE", flush=True)
