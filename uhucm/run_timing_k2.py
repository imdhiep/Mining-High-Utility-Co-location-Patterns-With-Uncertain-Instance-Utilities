"""Same as run_timing_k.py for a chosen list of feature counts (k >= 40, no ablation).
Usage: OMP_NUM_THREADS=1 python run_timing_k2.py CITY K1 [K2 ...]"""
import sys
import fsq_loader, paths
from suite import Suite
city = sys.argv[1]; out = "out_timing2"
for nf in map(int, sys.argv[2:]):
    rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, nfeat=nf)
    S = Suite(f"FSQ_{city}_k{nf}", rows, 50.0, out, zmain=1.645, log=lambda s: print(s, flush=True))
    S.timing(Ks=(30, 100, 300, 1000), reps=10, abl_K=-1, np_once=True); S.save()
print("DONE", flush=True)
