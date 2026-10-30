"""Kaggle driver (Yelp): quality at d = 50 m for every listed city with caliper matching (|training-score
difference| <= 0.05 of the threshold) for kept vs rejected patterns and for the filter comparison; writes
<OUT>/suite_<city>_d50.json. Usage: python kexp6.py ROOT OUT SPLIT TEST_END "City1;City2;..." """
import sys, os, multiprocessing as mp
import yelp_full
from suite import Suite


def run(args):
    out, city, data = args
    S = Suite(f"{city.replace(' ', '_')}_d50", data["rows"], 50.0, out, log=lambda s: None)
    S.quality(); S.save(); return city


if __name__ == "__main__":
    root, out, split, tend = sys.argv[1:5]; cities = sys.argv[5].split(";")
    os.makedirs(out, exist_ok=True)
    data = yelp_full.build(root, cities, split=split, test_end=tend, log=lambda s: None)
    with mp.get_context("fork").Pool(4) as pool:
        print(pool.map(run, [(out, c, data[c]) for c in cities if c in data]), flush=True)
    print("DONE6", flush=True)
