"""Kaggle driver (Yelp): quality at d = 50 m for every listed city (hold-out with matched-pair counts, filters with
score-matched comparison), then repeated timing (10 runs per variant) on the first three cities, at d = 50 m and,
on the first city, at d = 70 m. Prints compact lines with a CRC so that transcribed lines can be checked.
Usage: python kexp5.py ROOT OUT SPLIT TEST_END "City1;City2;..." """
import sys, os, json, zlib, platform, multiprocessing as mp
import numpy as np
import yelp_full
from suite import Suite


def emit(s):
    print(f"{s}|ck={zlib.crc32(s.encode()) & 0xffff:04x}", flush=True)


def r6(x):
    return "na" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x))


def qline(tag, q):
    p2 = q.get("pair2x2", {})
    F = []
    for k, v in q.get("filters", {}).items():
        b = v.get("pair2x2", {})
        F.append(f"{k}=" + ",".join(r6(v.get(x)) for x in ("J", "n_only_f", "hits_only_f", "n_only_z", "hits_only_z", "train_only_f", "train_only_z"))
                 + "," + ",".join(r6(b.get(x)) for x in ("both", "only_z", "only_f", "neither")))
    emit(f"Q|{tag}|{q['K']}|{q['R0']},{q['R1.645']},{q['R2.45']},{q['R4.36']}"
         f"|{r6(q.get('hits_kept'))}/{r6(q.get('n_kept'))},{r6(q.get('hits_rej'))}/{r6(q.get('n_rej'))},{r6(q.get('hits_matched'))}/{r6(q.get('n_matched'))}"
         f"|{r6(q.get('perm_p'))}|{r6(q.get('FE_miss'))}|" + ",".join(r6(p2.get(x)) for x in ("both", "only_matched", "only_rej", "neither", "distinct_kept", "max_gap"))
         + f"|{r6(q.get('size_kept'))}/{r6(q.get('size_rej'))}|{r6(q.get('atoms_kept'))}/{r6(q.get('atoms_rej'))}|{r6(q.get('med_ev_kept'))}/{r6(q.get('med_ev_rej'))}"
         f"|{r6(q.get('J_raw'))}|" + ";".join(F))


def run_quality(args):
    out, city, data = args
    tag = city.replace(" ", "_")
    S = Suite(f"{tag}_d50", data["rows"], 50.0, out, log=lambda s: None)
    Q = S.quality(); st = S.rec["stats"]
    emit(f"S|{tag}|{st['n']},{st['features']},{st['cliques']},{st['omega']},{r6(st['ev_median'])},{st['insig_1645']},{st['neg']},{st['test_cov']},"
         f"{r6(data['sigma_s2'])},{r6(min(h['kappa'] for h in data['hyper'].values()))},{r6(max(h['kappa'] for h in data['hyper'].values()))}")
    for q in Q: qline(tag, q)
    S.save()
    return tag


def run_timing(out, city, data, ds, Ks):
    tag = city.replace(" ", "_")
    S = Suite(f"{tag}_timing", data["rows"], 50.0, out, log=lambda s: None)
    T = S.timing(Ks=Ks, reps=10, ds=ds)
    for d, res in T.items():
        for K, rec in res.items():
            if K == "cliques": continue
            emit(f"T|{tag}|{int(d)}|{K}|{res['cliques']}|" + ";".join(
                f"{n}={np.mean(r['times']):.4f},{np.std(r['times'], ddof=1):.4f},{r['evaluated']}" for n, r in rec.items()))
    S.save()


if __name__ == "__main__":
    root, out, split, tend = sys.argv[1:5]; cities = sys.argv[5].split(";")
    os.makedirs(out, exist_ok=True)
    import scipy, pandas, pyproj
    emit(f"V|python {platform.python_version()}|numpy {np.__version__}|scipy {scipy.__version__}|pyproj {pyproj.__version__}|cpus {os.cpu_count()}")
    data = yelp_full.build(root, cities, split=split, test_end=tend, log=lambda s: None)
    emit("C|" + ";".join(f"{c.replace(' ', '_')}={len(data[c]['rows']) if c in data else 0}" for c in cities))
    jobs = [(out, c, data[c]) for c in cities if c in data and len(data[c]["rows"]) >= 1000]
    with mp.get_context("fork").Pool(4) as pool:
        print(pool.map(run_quality, jobs), flush=True)
    for i, c in enumerate(cities[:3]):
        if c in data: run_timing(out, c, data[c], (50.0,), (30, 100, 300, 1000, 3000))
    if cities[0] in data: run_timing(out, cities[0] + "_d70", data[cities[0]], (70.0,), (30, 300))
    print("DONE5", flush=True)
