"""Foursquare TSMC2014 as published on Kaggle (chetanism/foursquare-nyc-and-tokyo-checkin-dataset: CSV with a header,
UTF-8) -> the original tab-separated, latin-1 files read by cpp/bin/fsq_prep. The layouts built from the result are
checked against checksums of the layouts built from the original files (kaggle/layout_checksums.txt).
Usage: python fsq_from_csv.py CSV_DIR OUT_DIR"""
import csv, glob, os, sys

COLUMNS = ["userId", "venueId", "venueCategoryId", "venueCategory", "latitude", "longitude", "timezoneOffset", "utcTimestamp"]


def main(src, out):
    os.makedirs(out, exist_ok=True)
    for code in ("NYC", "TKY"):
        path = glob.glob(os.path.join(src, "**", f"dataset_TSMC2014_{code}.csv"), recursive=True)[0]
        with open(path, newline="", encoding="utf-8") as f, open(os.path.join(out, f"dataset_TSMC2014_{code}.txt"), "w",
                                                                   encoding="latin-1", newline="\n") as g:
            rows = 0
            for r in csv.DictReader(f):
                g.write("\t".join(r[c] for c in COLUMNS) + "\n"); rows += 1
        print(code, rows, "check-ins")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
