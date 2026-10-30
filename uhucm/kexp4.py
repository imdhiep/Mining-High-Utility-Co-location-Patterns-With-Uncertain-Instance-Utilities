"""Kaggle driver for the revision (Yelp, D1-D3): quality at d = 50 m with filter baselines and hit counts, the same with
s0 = city median, and quality at d = 30 and 70 m. Prints compact 'Y|' lines and writes JSON.
Usage: python kexp4.py ROOT OUT SPLIT TEST_END "City1;City2;..." """
import sys, os, json, math, copy, multiprocessing as mp
import yelp_full
from suite import Suite

def compact(tag, rows):
    for q in rows:
        f = q.get("filters", {})
        fs = ";".join(f"{k}:{v['J']}/{v['hold']}/{v['hold_only_f']}/{v['hold_only_z']}/{v['n_only_f']}/{v['n_only_z']}" for k, v in f.items())
        print(f"Y|{tag}|K={q['K']}|R={q['R0']},{q['R1.645']},{q['R2.45']},{q['R4.36']}|kept={q.get('hits_kept')}/{q.get('n_kept')}"
              f"|rej={q.get('hits_rej')}/{q.get('n_rej')}|mat={q.get('hits_matched')}/{q.get('n_matched')}|p={q.get('perm_p')}"
              f"|rd={q.get('regr_diff')}|fe={q.get('FE_miss')}|sz={q.get('size_kept')}/{q.get('size_rej')}|at={q.get('atoms_kept')}/{q.get('atoms_rej')}"
              f"|ev={q.get('med_ev_kept')}/{q.get('med_ev_rej')}|F={fs}", flush=True)

def s0_rows(rows, s0):
    out = []
    for r in rows:
        r = dict(r); g = math.log1p(r["a"]); gt = math.log1p(r["a_test"]) if r["a_test"] >= 0 else g
        th = r["theta"]; r["mu"] = g * (th - s0) / 2
        r["raw"] = (g * (r["S"] / r["n_r"] - s0) / 2) if r["n_r"] else 0.0
        if r["n_test"]: r["mu_test"] = r["mu_test"] - gt * (s0 - 3.0) / 2
        out.append(r)
    return out

def run(args):
    out, city, rows, med = args
    tag = city.replace(" ", "_")
    L = lambda s: None
    res = {}
    S = Suite(f"{tag}_d50", rows, 50.0, out, log=L); res["d50"] = S.quality(); compact(f"{tag}_d50", res["d50"])
    S = Suite(f"{tag}_s0med", s0_rows(rows, med), 50.0, out, log=L); res["s0med"] = S.quality(); compact(f"{tag}_s0med", res["s0med"])
    for d in (30.0, 70.0):
        S = Suite(f"{tag}_d{int(d)}", rows, d, out, log=L); res[f"d{int(d)}"] = S.quality(); compact(f"{tag}_d{int(d)}", res[f"d{int(d)}"])
    json.dump(res, open(os.path.join(out, f"rev_{tag}.json"), "w"), default=str)
    return tag

if __name__ == "__main__":
    root, out, split, tend = sys.argv[1:5]; cities = sys.argv[5].split(";")
    os.makedirs(out, exist_ok=True)
    data = yelp_full.build(root, cities, split=split, test_end=tend)
    for c in cities:
        if c in data: print(f"Y|{c}|median_stars={data[c]['median_stars']:.4f}|n={len(data[c]['rows'])}", flush=True)
    jobs = [(out, c, data[c]["rows"], data[c]["median_stars"]) for c in cities if c in data]
    with mp.get_context("fork").Pool(min(3, len(jobs))) as pool:
        print(pool.map(run, jobs), flush=True)
    print("DONE4", flush=True)
