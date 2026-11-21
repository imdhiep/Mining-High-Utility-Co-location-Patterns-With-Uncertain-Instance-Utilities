"""Checks layouts against a reference made from the original Foursquare files: every column except the coordinates
must be identical (same venues, features, counts, order), and the coordinates must agree within TOLERANCE metres
(the CSV copy on Kaggle may carry latitudes and longitudes rounded in the last bit).
Usage: python check_layouts.py LAYOUT_DIR REFERENCE_JSON [--write]   (--write: create the reference from LAYOUT_DIR)"""
import csv, glob, hashlib, json, os, sys

TOLERANCE = 1e-3


def summary(path):
    h = hashlib.md5(); sx = sy = 0.0; n = 0
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f); header = next(r)
        ix, iy = header.index("LocX"), header.index("LocY")
        for row in r:
            h.update(("|".join(v for i, v in enumerate(row) if i not in (ix, iy)) + "\n").encode())
            sx += float(row[ix]); sy += float(row[iy]); n += 1
    return dict(other_columns_md5=h.hexdigest(), rows=n, sum_x=sx, sum_y=sy)


def main(layout_dir, reference, write):
    files = sorted(glob.glob(os.path.join(layout_dir, "New_York_k*.csv")) + glob.glob(os.path.join(layout_dir, "Tokyo_k*.csv")))
    got = {os.path.basename(p): summary(p) for p in files}
    if write:
        json.dump(got, open(reference, "w"), indent=1); print("reference written:", len(got), "files"); return 0
    ref = json.load(open(reference)); bad = 0
    for name, r in ref.items():
        g = got.get(name)
        ok = g is not None and g["other_columns_md5"] == r["other_columns_md5"] and g["rows"] == r["rows"] and \
            abs(g["sum_x"] - r["sum_x"]) <= TOLERANCE * r["rows"] and abs(g["sum_y"] - r["sum_y"]) <= TOLERANCE * r["rows"]
        bad += not ok
        print(name, "OK" if ok else "DIFFERENT")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], "--write" in sys.argv))
