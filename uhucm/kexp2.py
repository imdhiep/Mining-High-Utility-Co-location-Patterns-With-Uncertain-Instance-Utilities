"""Kaggle driver for the current Yelp Open Dataset (and the Challenge-era release): full experiment suite per city.
Usage: python kexp2.py ROOT OUT d SPLIT TEST_END CITY1;CITY2;...  [quality_only]"""
import sys, os, functools, multiprocessing as mp, numpy as np
import yelp_full
from suite import Suite


def run(args):
    out, d, city, data, qonly = args
    rows = data["rows"]
    log = open(os.path.join(out, f"log_{city.replace(' ', '_')}.txt"), "w")
    def L(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    rec = functools.partial(yelp_full.recompute, rows, data["hyper"], data["sigma_s2"])
    med = data["median_stars"]
    variants = [("kappa_x0.25", dict(kappa_mult=0.25)), ("kappa_x0.5", dict(kappa_mult=0.5)), ("kappa_x2", dict(kappa_mult=2.0)),
                ("kappa_x4", dict(kappa_mult=4.0)), ("sigma_x0.7", dict(sigma_mult=0.7)), ("sigma_x1.3", dict(sigma_mult=1.3)),
                ("g_const", dict(g_kind="const")), ("s0_median", dict(s0=med))]
    S = Suite(city.replace(" ", "_"), rows, d, out, recompute=rec, variants=variants, users=data["users"], NU=data["NU"],
              interactions=yelp_full.interactions, resid=[r["theta"] - data["hyper"][r["feat"]]["mu_f"] for r in rows], log=L)
    L(f"R|{city}|hyper|sigma_s2={data['sigma_s2']:.4f} NU={data['NU']} median_stars={med:.3f}")
    yelp_full.save(rows, os.path.join(out, f"atoms_{city.replace(' ', '_')}.csv"))
    S.quality(); S.interaction()
    if not qonly:
        S.sensitivity(); S.dependence(); S.efficiency()
    S.save(); log.close()


if __name__ == "__main__":
    root, out, d, split, tend = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
    cities = sys.argv[6].split(";"); qonly = len(sys.argv) > 7
    os.makedirs(out, exist_ok=True)
    data = yelp_full.build(root, cities, split=split, test_end=tend)
    jobs = [(out, d, c, data[c], qonly) for c in cities if c in data]
    with mp.get_context("fork").Pool(min(4, len(jobs))) as pool:
        pool.map(run, jobs)
    print("DONE", flush=True)
