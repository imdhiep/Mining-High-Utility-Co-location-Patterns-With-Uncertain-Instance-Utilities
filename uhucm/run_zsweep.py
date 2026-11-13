"""z-sweep of the pruning bounds (review item A1): exact bound (UHUCPM), mean bound and NoPrune at K = 300 on
Foursquare with the compound EB posteriors and total variance, for z in {1.645, 2.326, 3.09, 4.36}.
5 interleaved runs for UHUCPM and the mean bound, 1 for NoPrune (its node count does not depend on the bounds).
Usage: OMP_NUM_THREADS=1 python run_zsweep.py CITY k [k ...]   -> out_zsweep/zsweep_CITY_k*.json"""
import sys, os, json
import fsq_loader, paths
from suite import Suite
city = sys.argv[1]; ks = [int(x) for x in sys.argv[2:]] or [20, 60]; out = "out_zsweep"; os.makedirs(out, exist_ok=True)
for nf in ks:
    rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, nfeat=nf, model="ceb"); UB, Zh = info.pop("UB"), info.pop("Zhat")
    res = {}
    for z in (1.645, 2.326, 3.09, 4.36):
        S = Suite(f"Z_{city}_k{nf}_z{z}", rows, 50.0, out, zmain=z, log=lambda s: print(s, flush=True), UB=UB, Zhat=Zh)
        res[str(z)] = S.timing(Ks=(300,), reps=5, z=z, abl_K=-1, np_reps=1)
    json.dump(res, open(os.path.join(out, f"zsweep_{city}_k{nf}.json"), "w"), indent=1, default=str)
print("DONE", flush=True)
