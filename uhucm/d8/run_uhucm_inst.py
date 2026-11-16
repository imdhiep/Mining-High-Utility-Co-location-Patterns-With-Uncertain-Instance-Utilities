"""One UHUCPM run at z = 0 with instance-level utilities, read from the same files as the authors' CQ-HUCPM and
PSO-MUCP runners (run_cq_fair.py, run_pso_fair.py), in its own process, so that all of them are timed in the same way.
Usage: python run_uhucm_inst.py DATA_CSV UTIL_JSON MINUTIL D   -> one JSON line with n, patterns and the inner time.
MINUTIL is relative to the total positive utility of the instances, the normalizer given to all three programs."""
import sys, os, csv, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fastmine

t = time.time()
data, ujson, mu, d = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
U = json.load(open(ujson))
R = list(csv.DictReader(open(data)))
names = sorted({r["Feature"] for r in R}); fi = {f: j for j, f in enumerate(names)}
feat = [fi[r["Feature"]] for r in R]; xy = [(float(r["LocX"]), float(r["LocY"])) for r in R]
mu_i = [float(U[r["Feature"] + r["Instance"]]) for r in R]
Pb = fastmine.Problem(feat, xy, mu_i, [0.0] * len(R), d, 0.0, feat_names=names)
norm = sum(max(x, 0.0) for x in mu_i)
res, st, _ = fastmine.mine(Pb, mu * norm / Pb.Z)
print(json.dumps(dict(mode="uhucpm", n=len(res), patterns=sorted("".join(sorted(p)) for p in res),
                      seconds=round(time.time() - t, 2), cliques=len(Pb.cliques), evaluated=st["evaluated"])))
