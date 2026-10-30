"""Yelp (Challenge-era release) -> atom tables using per-review evidence (runs on Kaggle).

Evidence actually used (Def. 4 with alpha = 1, rating channel only; text sentiment is not used in this run):
  business.json : coordinates, categories -> curated feature table, review_count (only for reporting)
  checkin.json  : a(o) = total check-ins (this release has weekday-hour counts without dates, so a(o) is not split by time)
  review.json   : per-review stars, date, user_id
Temporal split: reviews dated < SPLIT are training evidence, >= SPLIT are test evidence (E7 only).
Hyperparameters per city and feature, from training reviews only:
  sigma_s^2 = pooled within-business variance of stars;
  mu_f      = review-weighted mean stars;  tau_f^2 = Var(business means) - sigma_s^2 * mean(1/n)  (method of moments, floored);
  kappa_r   = sigma_s^2 / tau_f^2.
Interaction evidence (Def. 5 proxy): U(o) = set of training reviewers of o; for neighbouring pairs
  cv = |U(o) & U(o')|, e = |U(o)| |U(o')| / N_U, iota = (cv-e)/(cv+e+kappa_i), Var(iota) = cv [(2e+kappa_i)/(cv+e+kappa_i)^2]^2.
"""
import json, math, collections, csv, os, time

FEATURES = [
    ("Coffee & Tea", "Coffee"), ("Fast Food", "FastFood"), ("Pizza", "Pizza"), ("Mexican", "Mexican"),
    ("Chinese", "Chinese"), ("Japanese", "Japanese"), ("Italian", "Italian"), ("Burgers", "Burgers"),
    ("Breakfast & Brunch", "Breakfast"), ("Desserts", "Desserts"), ("Bars", "Bars"), ("Hotels", "Hotels"),
    ("Nail Salons", "Nails"), ("Hair Salons", "Hair"), ("Fitness & Instruction", "Fitness"),
    ("Auto Repair", "AutoRepair"), ("Dentists", "Dentists"), ("Grocery", "Grocery"),
    ("Banks & Credit Unions", "Banks"), ("Fashion", "Fashion"),
]
UTM = {"Las Vegas": 32611, "Phoenix": 32612, "Toronto": 32617, "Charlotte": 32617, "Pittsburgh": 32617,
       "Calgary": 32611, "Scottsdale": 32612, "Mesa": 32612}


def feature_of(cats):
    s = {c.strip() for c in (cats or "").split(",")}
    for key, name in FEATURES:
        if key in s:
            return name
    return None


def build(root, cities, split="2017-01-01", test_end="9999-12-31", s0=3.0, tau2_floor=0.05, log=print):
    import pyproj
    t0 = time.time()
    biz = {}
    with open(os.path.join(root, "yelp_academic_dataset_business.json")) as f:
        for l in f:
            b = json.loads(l)
            if b.get("city") not in cities or b["business_id"] in biz or b.get("latitude") is None:
                continue
            ft = feature_of(b.get("categories"))
            if ft is None:
                continue
            biz[b["business_id"]] = dict(city=b["city"], feat=ft, lat=b["latitude"], lon=b["longitude"],
                                         review_count=int(b["review_count"]))
    ck = collections.Counter(); ck_te = collections.Counter(); dated = False
    with open(os.path.join(root, "yelp_academic_dataset_checkin.json")) as f:
        for l in f:
            r = json.loads(l)
            if r["business_id"] not in biz:
                continue
            if "date" in r:                      # current release: comma-separated timestamps
                dated = True
                for t in r["date"].split(","):
                    t = t.strip()[:10]
                    if t < split: ck[r["business_id"]] += 1
                    elif t < test_end: ck_te[r["business_id"]] += 1
            else:                                # Challenge-era release: weekday-hour counts, no dates
                ck[r["business_id"]] += sum(r.get("time", {}).values())
    tr = collections.defaultdict(list); te = collections.defaultdict(list); users = collections.defaultdict(set)
    dmin, dmax, nrev = "9999", "0000", 0
    with open(os.path.join(root, "yelp_academic_dataset_review.json")) as f:
        for l in f:
            r = json.loads(l)
            b = r["business_id"]
            if b not in biz:
                continue
            nrev += 1; dt = r["date"][:10]; dmin = min(dmin, dt); dmax = max(dmax, dt)
            if dt < split:
                tr[b].append(float(r["stars"])); users[b].add(r["user_id"])
            elif dt < test_end:
                te[b].append(float(r["stars"]))
    log(f"[yelp] businesses={len(biz)} reviews kept={nrev} dates {dmin}..{dmax} split={split} test_end={test_end} dated_checkins={dated} ({time.time()-t0:.0f}s)")
    out = {}
    for city in cities:
        ids = [b for b in biz if biz[b]["city"] == city]
        if not ids:
            continue
        lons = sorted(biz[b]["lon"] for b in ids); zone = int((lons[len(lons) // 2] + 180) // 6) + 1
        tf = pyproj.Transformer.from_crs(4326, UTM.get(city, 32600 + zone), always_xy=True)
        # pooled within-business variance -> sigma_s^2
        num = sum(sum((s - sum(tr[b]) / len(tr[b])) ** 2 for s in tr[b]) for b in ids if len(tr[b]) >= 2)
        den = sum(len(tr[b]) - 1 for b in ids if len(tr[b]) >= 2)
        s2s = num / den
        hyper = {}
        byf = collections.defaultdict(list)
        for b in ids:
            if tr[b]:
                byf[biz[b]["feat"]].append(b)
        for ft, L in byf.items():
            m = sum(sum(tr[b]) for b in L) / sum(len(tr[b]) for b in L)
            means = [sum(tr[b]) / len(tr[b]) for b in L]
            mb = sum(means) / len(means)
            v = sum((x - mb) ** 2 for x in means) / max(1, len(means) - 1)
            tau2 = max(tau2_floor, v - s2s * sum(1 / len(tr[b]) for b in L) / len(L))
            hyper[ft] = dict(mu_f=m, tau2=tau2, kappa=s2s / tau2, n=len(L))
        rows = []
        for b in ids:
            ft = biz[b]["feat"]
            if ft not in hyper:
                continue
            h = hyper[ft]; n = len(tr[b]); S = sum(tr[b])
            th = (S + h["kappa"] * h["mu_f"]) / (n + h["kappa"]); vth = s2s / (n + h["kappa"])
            g = math.log1p(ck[b]); gt = math.log1p(ck_te[b]) if dated else g
            x, y = tf.transform(biz[b]["lon"], biz[b]["lat"])
            nt = len(te[b])
            rows.append(dict(id=b, feat=ft, x=x, y=y, a=ck[b], a_test=(ck_te[b] if dated else -1), n_r=n, S=S, n_test=nt,
                             mu=g * (th - s0) / 2, s2=g * g * vth / 4,
                             raw=(g * (S / n - s0) / 2) if n else 0.0,
                             mu_test=(gt * (sum(te[b]) / nt - s0) / 2) if nt else float("nan"),
                             theta=th))
        NU = len(set().union(*(users[b] for b in ids)))
        out[city] = dict(rows=rows, hyper=hyper, sigma_s2=s2s, users={r["id"]: users[r["id"]] for r in rows}, NU=NU, s0=s0,
                         median_stars=float(sorted(r["theta"] for r in rows)[len(rows) // 2]))
        log(f"[yelp] {city}: n={len(rows)} sigma_s^2={s2s:.3f} N_U={NU} "
            f"n_r median={sorted(r['n_r'] for r in rows)[len(rows)//2]} with test reviews={sum(r['n_test']>0 for r in rows)}")
    return out


def interactions(rows, users, NU, nb, kappa_i=5.0):
    """nb: list of neighbour sets over row indices -> {(i,j): (iota, var)}"""
    U = [users[r["id"]] for r in rows]
    pairs = {}
    for i, S in enumerate(nb):
        for j in S:
            if i < j:
                cv = len(U[i] & U[j]); e = len(U[i]) * len(U[j]) / NU
                den = cv + e + kappa_i
                pairs[(i, j)] = ((cv - e) / den, cv * ((2 * e + kappa_i) / den ** 2) ** 2)
    return pairs


def save(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def recompute(rows, hyper, sigma_s2, kappa_mult=1.0, sigma_mult=1.0, g_kind="log", s0=3.0):
    """Posterior means/variances under a changed hyperparameter (E8). Uses the stored sufficient statistics."""
    mu, s2 = [], []
    ss = sigma_s2 * sigma_mult ** 2
    for r in rows:
        h = hyper[r["feat"]]; kap = h["kappa"] * kappa_mult * sigma_mult ** 2   # kappa = sigma^2/tau^2
        n, S = r["n_r"], r["S"]
        th = (S + kap * h["mu_f"]) / (n + kap); vth = ss / (n + kap)
        g = math.log1p(r["a"]) if g_kind == "log" else 1.0
        mu.append(g * (th - s0) / 2); s2.append(g * g * vth / 4)
    return mu, s2
