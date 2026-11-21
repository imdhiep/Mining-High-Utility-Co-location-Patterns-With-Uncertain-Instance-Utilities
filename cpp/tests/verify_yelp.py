"""C++ Yelp loader and rating posterior (bin/yelp_prep) against uhucm/yelp_full.py (build, vt_boot, expected_pos).
Uses the local business.json and check-in file of the Challenge-era release (check-ins without dates) or a mock check-in
file with dates, and synthetic review files (random users, stars and dates) for the chosen cities.
Usage: python verify_yelp.py BUSINESS_JSON CHECKIN_JSON "City|City" [--dated]"""
import csv, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "uhucm"))
import yelp_full  # noqa: E402

bjson, cjson, cities = sys.argv[1], sys.argv[2], sys.argv[3].split("|"); dated = "--dated" in sys.argv
rng = np.random.default_rng(5)
with tempfile.TemporaryDirectory() as tmp:
    root = os.path.join(tmp, "root"); os.makedirs(root)
    os.symlink(bjson, os.path.join(root, "yelp_academic_dataset_business.json"))
    ids = []
    with open(bjson) as f:
        for l in f:
            b = json.loads(l)
            if b.get("city") in cities: ids.append((b["business_id"], b["city"]))
    if dated:                                       # mock dated check-ins for half of the businesses
        with open(os.path.join(root, "yelp_academic_dataset_checkin.json"), "w") as f:
            for b, _ in ids:
                if rng.random() < 0.5: continue
                k = int(rng.integers(1, 40)); ds = []
                for _ in range(k):
                    y = int(rng.integers(2012, 2021)); m = int(rng.integers(1, 13)); d = int(rng.integers(1, 29))
                    ds.append(f"{y:04d}-{m:02d}-{d:02d} {int(rng.integers(0,24)):02d}:00:00")
                f.write(json.dumps({"business_id": b, "date": ", ".join(ds)}) + "\n")
    else:
        os.symlink(cjson, os.path.join(root, "yelp_academic_dataset_checkin.json"))
    rdir = os.path.join(tmp, "rev"); os.makedirs(rdir)
    for c in cities:
        with open(os.path.join(rdir, f"reviews_{c.replace(' ', '_')}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["review_id", "user_id", "business_id", "stars", "date"])
            for b, cc in ids:
                if cc != c: continue
                for j in range(int(rng.poisson(rng.choice([0.5, 3, 12])))):
                    y = int(rng.integers(2012, 2021))
                    w.writerow([f"r{b}{j}", f"u{int(rng.integers(0, 3000))}", b, float(rng.integers(1, 6)), f"{y:04d}-{int(rng.integers(1,13)):02d}-15 10:00:00"])
    split, test_end = "2019-01-01", "2020-01-01"
    out = yelp_full.build(root, cities, split=split, test_end=test_end, log=lambda s: None, review_dir=rdir)
    od = os.path.join(tmp, "out"); os.makedirs(od)
    r = subprocess.run([os.path.join(HERE, "..", "bin", "yelp_prep"), os.path.join(root, "yelp_academic_dataset_business.json"),
                        os.path.join(root, "yelp_academic_dataset_checkin.json"), rdir, "|".join(cities), split, test_end, od,
                        "--B", "400", "--seed", "1"], capture_output=True, text=True, check=True)
    summ = json.loads(r.stdout)
    for c in cities:
        rows = out[c]["rows"]
        C = list(csv.DictReader(open(os.path.join(od, c.replace(" ", "_") + ".csv"))))
        same_ids = [x["id"] for x in rows] == [x["business_id"] for x in C]
        f = lambda key, ck: float(np.nanmax(np.abs(np.array([float(x[key]) for x in rows]) - np.array([float(y[ck]) for y in C])))) if rows else 0.0
        mu = np.array([x["mu"] for x in rows]); s2 = np.array([x["s2"] for x in rows])
        Zpy = yelp_full.expected_pos(mu, s2)
        with open(os.path.join(od, c.replace(" ", "_") + ".post"), "rb") as fh:
            n, B = np.fromfile(fh, np.int64, 2); Zc = np.fromfile(fh, float, 1)[0]; np.fromfile(fh, float, 2 * n); CU = np.fromfile(fh, float, B * n).reshape(B, n)
        PU = yelp_full.vt_boot(rows, B=400, seed=1)
        F = np.array([x["feat"] for x in rows]); rr = []
        for _ in range(300):
            ft = rng.choice(np.unique(F)); idx = np.flatnonzero(F == ft); g = rng.choice(idx, min(len(idx), int(rng.integers(3, 80))), replace=False)
            rr.append(CU[:, g].sum(1).var(ddof=1) / max(PU[:, g].sum(1).var(ddof=1), 1e-300))
        print(json.dumps(dict(city=c, n=len(rows), dated=dated, ids_same=same_ids, feat_same=[x["feat"] for x in rows] == [y["feat"] for y in C],
                              max_dx=f("x", "LocX"), max_dy=f("y", "LocY"), max_mu=f("mu", "mu"), max_s2=f("s2", "s2"),
                              max_mu_test=f("mu_test", "mu_test"), a_same=[x["a"] for x in rows] == [int(float(y["a"])) for y in C],
                              Z_py=Zpy, Z_cpp=float(Zc), boot_VT_ratio_q05_50_95=[round(float(np.quantile(rr, q)), 3) for q in (0.05, 0.5, 0.95)])))
