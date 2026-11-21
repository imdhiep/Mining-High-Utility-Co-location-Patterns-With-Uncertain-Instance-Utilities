"""Instance files of the real datasets (layouts of the semi-synthetic data and real data of Section VI-D), written by
the C++ loaders into data/layouts/, with data/layouts/index.json giving the file and exposure of every (city, k).
  Foursquare TSMC2014 (New York, Tokyo): cpp/bin/fsq_prep; exposure = weeks of the training period.
  Yelp Open Dataset (Philadelphia, Tampa, New Orleans, ...): cpp/bin/yelp_reviews then cpp/bin/yelp_prep;
  check-ins before 2019-01-01 are the training period, exposure 1.
Usage: python prepare_layouts.py --fsq-dir DIR [--yelp-dir DIR] [--yelp-cities "City|City"] [--ks 10,20,30,40]"""
import argparse, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "..", "cpp", "bin"); OUT = os.path.join(ROOT, "data", "layouts")
FSQ = {"New_York": ("NYC", 18), "Tokyo": ("TKY", 54)}
YELP_SPLIT, YELP_TEST_END = "2019-01-01", "2020-01-01"


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fsq-dir", required=True); ap.add_argument("--yelp-dir")
    ap.add_argument("--yelp-cities", default="Philadelphia|Tampa|New Orleans"); ap.add_argument("--ks", default="10,20,30,40")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "index.json")
    index = json.load(open(path)) if os.path.exists(path) else {}
    for k in (int(x) for x in a.ks.split(",")):
        for name, (code, zone) in FSQ.items():
            f = os.path.join(OUT, f"{name}_k{k}.csv")
            r = run([os.path.join(BIN, "fsq_prep"), os.path.join(a.fsq_dir, f"dataset_TSMC2014_{code}.txt"), zone, k, f])
            index[f"{name}_k{k}"] = dict(file=os.path.relpath(f, ROOT), exposure=r["Wtr"], test_exposure=r["Wte"], n=r["n"])
            print(name, k, r, flush=True)
        if a.yelp_dir:
            reviews = os.path.join(OUT, "yelp_reviews")
            os.makedirs(reviews, exist_ok=True)
            business = os.path.join(a.yelp_dir, "yelp_academic_dataset_business.json")
            if not all(os.path.exists(os.path.join(reviews, f"reviews_{c.replace(' ', '_')}.csv")) for c in a.yelp_cities.split("|")):
                print(run([os.path.join(BIN, "yelp_reviews"), business, os.path.join(a.yelp_dir, "yelp_academic_dataset_review.json"),
                           a.yelp_cities, reviews]), flush=True)
            out_dir = os.path.join(OUT, f"yelp_k{k}"); os.makedirs(out_dir, exist_ok=True)
            r = run([os.path.join(BIN, "yelp_prep"), business, os.path.join(a.yelp_dir, "yelp_academic_dataset_checkin.json"), reviews,
                     a.yelp_cities, YELP_SPLIT, YELP_TEST_END, out_dir, "--nfeat", k, "--B", 0])
            for c in r["cities"]:
                name = c["city"].replace(" ", "_")
                index[f"{name}_k{k}"] = dict(file=os.path.relpath(os.path.join(out_dir, name + ".csv"), ROOT), exposure=1.0, n=c["n"])
                print(name, k, c, flush=True)
    json.dump(index, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
