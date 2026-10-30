"""Repeated timing on Foursquare (10 runs per variant, z = 1.645 as on the Yelp data) with the binary-search suffix
bound: exact bound, mean bound and NoPrune at K in {30, 100, 300, 1000} for k = 20, 40, 60 and 80 features (the most
frequent venue categories), the bound/strategy ablation (including the linear suffix scan) at k = 20 and K = 300, and
the dense setting d = 70 m at k = 20. Usage: OMP_NUM_THREADS=1 python run_timing_k.py CITY"""
import sys
import fsq_loader, paths
from suite import Suite
city = sys.argv[1]; out = "out_timing2"
for nf in (20, 40, 60, 80):
    rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, nfeat=nf)
    S = Suite(f"FSQ_{city}_k{nf}", rows, 50.0, out, zmain=1.645, log=lambda s: print(s, flush=True))
    S.timing(Ks=(30, 100, 300, 1000), reps=10, abl_K=300 if nf == 20 else -1); S.save()
    if nf == 20:
        S = Suite(f"FSQ_{city}_k20_d70", rows, 70.0, out, zmain=1.645, log=lambda s: print(s, flush=True))
        S.timing(Ks=(30, 300), reps=10, abl_K=-1); S.save()
print("DONE", flush=True)
