"""One UHUCPM run in its own process, reading the same instance file as the authors' miners plus a .npz with the
posterior means, variances and (optionally) the bootstrap matrix of V_T and the normalizer.
Usage: python run_uhucpm.py DATA_CSV NPZ THRESHOLD D Z VARIANT
  THRESHOLD  normalized score threshold (score = (M - z sqrt(V)) / Z with the normalizer of the .npz)
  VARIANT    exact (UHUCPM), mean (the mean bound in place of the exact subtree bound) or none (no pruning)
-> one JSON line with n, patterns and the inner time"""
import sys, os, csv, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fastmine

t = time.time()
data, npz, thr, d, z, variant = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
R = list(csv.DictReader(open(data, encoding="utf-8")))
names = sorted({r["Feature"] for r in R}); fi = {f: j for j, f in enumerate(names)}
feat = [fi[r["Feature"]] for r in R]; xy = [(float(r["LocX"]), float(r["LocY"])) for r in R]
A = np.load(npz)
Pb = fastmine.Problem(feat, xy, A["mu"], A["s2"], d, z, feat_names=names)
if "UB" in A.files: Pb.set_boot(A["UB"], float(A["Z"]))
elif "Z" in A.files: Pb.Z = float(A["Z"])
res, st, _ = fastmine.mine(Pb, thr, (0, 0, 0, 0), "exact") if variant == "none" else fastmine.mine(Pb, thr, (1, 1, 1, 1), variant)
print(json.dumps(dict(mode="uhucpm_" + variant, n=len(res), patterns=sorted("".join(sorted(p)) for p in res),
                      seconds=round(time.time() - t, 2), cliques=len(Pb.cliques), evaluated=st["evaluated"])))
