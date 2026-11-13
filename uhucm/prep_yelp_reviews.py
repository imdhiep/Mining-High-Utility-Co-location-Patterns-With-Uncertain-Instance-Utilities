"""Slim the Yelp review/tip files for selected cities so they can be moved (< 400 MB each).

Run on Windows from D:\\datamining (standard library only, Python 3.8+):
    python prep_yelp_reviews.py
Optional sentiment (VADER, slower; needs: pip install vaderSentiment):
    python prep_yelp_reviews.py --sentiment

Output in D:\\datamining\\slim\\ :
    reviews_<City>.csv : review_id, user_id, business_id, stars, date[, sentiment]   (no review text is kept)
    tips_<City>.csv    : user_id, business_id, date[, sentiment]
    prep_log.txt       : counts per city
"""
import argparse, csv, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CITIES = ["Las Vegas", "Toronto", "Phoenix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment", action="store_true", help="add VADER compound score in [-1,1]")
    ap.add_argument("--cities", default=",".join(CITIES))
    ap.add_argument("--archive", default=os.path.join(HERE, "archive"), help="folder with yelp_academic_dataset_*.json")
    ap.add_argument("--out", default=os.path.join(HERE, "slim"))
    ap.add_argument("--no-tips", action="store_true")
    A = ap.parse_args(); ARCH, OUT = A.archive, A.out
    cities = [c.strip() for c in A.cities.split(",")]
    os.makedirs(OUT, exist_ok=True)
    sia = None
    if A.sentiment:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()

    t0 = time.time()
    biz_city = {}
    with open(os.path.join(ARCH, "yelp_academic_dataset_business.json"), encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b.get("city") in cities:
                biz_city[b["business_id"]] = b["city"]
    print(f"businesses kept: {len(biz_city)}", flush=True)

    def writer(kind, city, cols):
        fh = open(os.path.join(OUT, f"{kind}_{city.replace(' ', '_')}.csv"), "w", newline="", encoding="utf-8")
        w = csv.writer(fh); w.writerow(cols); return fh, w

    log = []
    for kind, fname, cols in [
        ("reviews", "yelp_academic_dataset_review.json", ["review_id", "user_id", "business_id", "stars", "date"]),
        ("tips", "yelp_academic_dataset_tip.json", ["user_id", "business_id", "date"]),
    ][:1 if A.no_tips else 2]:
        c2 = cols + (["sentiment"] if sia else [])
        W = {c: writer(kind, c, c2) for c in cities}
        cnt = {c: 0 for c in cities}; n = 0
        with open(os.path.join(ARCH, fname), encoding="utf-8") as f:
            for line in f:
                n += 1
                if n % 500000 == 0:
                    print(f"  {kind}: {n:,} lines read, {time.time()-t0:.0f}s", flush=True)
                r = json.loads(line)
                city = biz_city.get(r.get("business_id"))
                if city is None:
                    continue
                row = [r.get(k, "") for k in cols]
                if sia:
                    row.append(round(sia.polarity_scores(r.get("text", ""))["compound"], 4))
                W[city][1].writerow(row); cnt[city] += 1
        for c in cities:
            W[c][0].close()
        log.append(f"{kind}: {n} lines read; kept " + ", ".join(f"{c}={cnt[c]}" for c in cities))
        print(log[-1], flush=True)
    with open(os.path.join(OUT, "prep_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + f"\nsentiment={'VADER' if sia else 'none'}\nseconds={time.time()-t0:.0f}\n")
    print("done ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
