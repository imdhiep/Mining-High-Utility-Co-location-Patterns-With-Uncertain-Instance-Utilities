"""Inputs of the hold-rate experiment (Section VI-D.1) in data/hold/index.json.
  Foursquare (New York, Tokyo): the 20-feature layouts of data/layouts (prepare_layouts.py), with both exposures.
  Yelp cities: cpp/bin/yelp_reviews, then cpp/bin/yelp_prep (all features, B = 20, seed 0): instance file with the
  test utilities and the training posterior file. Training period before 2019-01-01, test period 2019.
Usage: python prepare_hold.py --yelp-dir DIR --cities "City|City|..." """
import argparse, json, os, subprocess

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "..", "cpp", "bin"); OUT = os.path.join(ROOT, "data", "hold")
SPLIT, TEST_END, BOOTSTRAP = "2019-01-01", "2020-01-01", 20


def run(cmd):
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yelp-dir", required=True); ap.add_argument("--cities", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "reviews"), exist_ok=True); os.makedirs(os.path.join(OUT, "yelp"), exist_ok=True)
    layouts = json.load(open(os.path.join(ROOT, "data", "layouts", "index.json")))
    index = {}
    for name in ("New_York", "Tokyo"):
        e = layouts[f"{name}_k20"]
        index[name] = dict(source="foursquare", layout=e["file"], exposure=e["exposure"], test_exposure=e["test_exposure"])
    business = os.path.join(a.yelp_dir, "yelp_academic_dataset_business.json")
    print(run([os.path.join(BIN, "yelp_reviews"), business, os.path.join(a.yelp_dir, "yelp_academic_dataset_review.json"), a.cities,
               os.path.join(OUT, "reviews")]), flush=True)
    r = run([os.path.join(BIN, "yelp_prep"), business, os.path.join(a.yelp_dir, "yelp_academic_dataset_checkin.json"),
             os.path.join(OUT, "reviews"), a.cities, SPLIT, TEST_END, os.path.join(OUT, "yelp"), "--B", BOOTSTRAP, "--seed", 0])
    for c in r["cities"]:
        name = c["city"].replace(" ", "_")
        index[name] = dict(source="yelp", csv=os.path.relpath(os.path.join(OUT, "yelp", name + ".csv"), ROOT),
                           post=os.path.relpath(os.path.join(OUT, "yelp", name + ".post"), ROOT), n=c["n"])
        print(c, flush=True)
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
