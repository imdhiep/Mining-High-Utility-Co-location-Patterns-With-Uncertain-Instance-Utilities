"""Timing with one protocol for every variant and configuration: 5 interleaved runs of the mining step (after the
clique build) for UHUCPM, the mean bound and NoPrune at every K, z = 1.645, total variance. At k = 20 also the
ablation at K = 300 and d = 70 m (K = 30, 300). Usage: OMP_NUM_THREADS=1 python run_timing_fair.py CITY k [K ...]
-> out_timing_fair/suite_FAIR_<CITY>_k<k>[_K...].json"""
import sys, os
import fsq_loader, paths
from suite import Suite
city, nf = sys.argv[1], int(sys.argv[2]); Ks = tuple(int(x) for x in sys.argv[3:]) or (30, 100, 300, 1000)
out = "out_timing_fair"; os.makedirs(out, exist_ok=True); tag = "" if not sys.argv[3:] else "_K" + "_".join(sys.argv[3:])
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, nfeat=nf, model="ceb"); UB, Zh = info.pop("UB"), info.pop("Zhat")
S = Suite(f"FAIR_{city}_k{nf}{tag}", rows, 50.0, out, zmain=1.645, log=lambda s: print(s, flush=True), UB=UB, Zhat=Zh)
S.timing(Ks=Ks, reps=5, abl_K=300 if nf == 20 else -1, np_once=False); S.save()
if nf == 20 and not sys.argv[3:]:
    S = Suite(f"FAIR_{city}_k20_d70", rows, 70.0, out, zmain=1.645, log=lambda s: print(s, flush=True), UB=UB, Zhat=Zh)
    S.timing(Ks=(30, 300), reps=5, abl_K=-1, np_once=False); S.save()
print("DONE", flush=True)
