"""Foursquare TSMC2014 (NYC / Tokyo) -> atom table (Def. 4, Gamma-Poisson relative-performance utility) + interaction pairs.

Fields used: user id, venue id, category name, latitude, longitude, UTC time (+ offset).
Training period: 2012-04-03 .. 2012-11-30 (local time); test period: 2012-12-01 .. 2013-02-16.
Features: the 20 most frequent public venue categories by number of venues (private/generic categories excluded).
Instance utility (training): u = ln(lam_o) - c_f with lam_o | a ~ Gamma(alpha_f + a, beta_f + W) (Poisson counts with a
Gamma prior per category, alpha_f and beta_f by moments); mean digamma(alpha_f + a) - ln(beta_f + W), variance trigamma(alpha_f + a);
c_f = median posterior mean of the category. Test utility: same posterior with test counts and W_test.
phi_w (within-venue weekly dispersion, pooled moments) is kept for the quasi-Poisson sensitivity variant.
Interactions: U(o) = training visitors; cv = |U(o) & U(o')| (Delta = whole training period), e = |U||U'|/N_U.
"""
import math, collections, csv, sys
import pandas as pd, numpy as np, pyproj
from scipy.special import digamma, polygamma

EXCLUDE = {"Home (private)", "Residential Building (Apartment / Condo)", "Building", "Road", "Bridge", "Office",
           "Government Building", "College Academic Building", "Other Great Outdoors", "Miscellaneous Shop",
           "Restaurant", "General Entertainment"}
UTM = {"NYC": 32618, "TKY": 32654}
TRAIN_END = pd.Timestamp("2012-12-01")


def load(path, city, nfeat=20):
    df = pd.read_csv(path, sep="\t", header=None, encoding="latin-1",
                     names=["u", "v", "c", "cn", "lat", "lon", "tz", "t"])
    df["ts"] = pd.to_datetime(df.t, format="%a %b %d %H:%M:%S +0000 %Y") + pd.to_timedelta(df.tz, unit="m")
    df = df[~df.cn.isin(EXCLUDE)]
    top = df.groupby("cn").v.nunique().sort_values(ascending=False).head(nfeat).index
    df = df[df.cn.isin(top)]
    # one category per venue (most frequent label)
    vcat = df.groupby("v").cn.agg(lambda s: s.value_counts().index[0])
    tr = df[df.ts < TRAIN_END]; te = df[df.ts >= TRAIN_END]
    t0, t1, t2 = df.ts.min(), TRAIN_END, df.ts.max()
    Wtr = (t1 - t0).days / 7.0; Wte = (t2 - t1).days / 7.0
    a = tr.groupby("v").size(); ate = te.groupby("v").size()
    venues = a.index                              # instances: venues with >= 1 training check-in
    # weekly counts for phi
    tr = tr.assign(week=((tr.ts - t0).dt.days // 7))
    wk = tr.groupby(["v", "week"]).size().unstack(fill_value=0)
    nweeks = int(wk.columns.max()) + 1
    wk = wk.reindex(columns=range(nweeks), fill_value=0)
    m = wk.mean(1); var = wk.var(1, ddof=1)
    phi = {}
    for f in top:
        idx = [v for v in venues if vcat[v] == f]
        num = float((m[idx] ** 2).sum()); den = float(np.clip(var[idx] - m[idx], 0, None).sum())
        phi_w = num / den if den > 0 else float("inf")
        phi[f] = nweeks * phi_w
    coords = df.groupby("v")[["lat", "lon"]].median()
    tf = pyproj.Transformer.from_crs(4326, UTM[city], always_xy=True)
    # Gamma-Poisson model (Def. 4): a | lam ~ Poisson(lam * W), lam ~ Gamma(alpha_f, beta_f) per category;
    # alpha_f, beta_f by the method of moments on the training counts of the category's venues.
    alpha, beta = {}, {}
    for f in top:
        x = a[[v for v in venues if vcat[v] == f]].astype(float)
        m_, v_ = float(x.mean()), float(x.var(ddof=1))
        alpha[f] = m_ * m_ / (v_ - m_) if v_ > m_ else 1e6
        beta[f] = alpha[f] * Wtr / m_                       # prior mean weekly rate m_/Wtr
    def post(f, cnt, W):
        return float(digamma(alpha[f] + cnt) - math.log(beta[f] + W)), float(polygamma(1, alpha[f] + cnt))
    pm = {v: post(vcat[v], float(a[v]), Wtr) for v in venues}
    pt = {v: post(vcat[v], float(ate.get(v, 0)), Wte) for v in venues}
    lraw = {v: math.log(float(a[v]) / Wtr) for v in venues}
    med = {f: float(np.median([pm[v][0] for v in venues if vcat[v] == f])) for f in top}
    medte = {f: float(np.median([pt[v][0] for v in venues if vcat[v] == f])) for f in top}
    medraw = {f: float(np.median([lraw[v] for v in venues if vcat[v] == f])) for f in top}
    rows = []
    for v in venues:
        f = vcat[v]; av = float(a[v])
        x, y = tf.transform(coords.lon[v], coords.lat[v])
        rows.append(dict(id=v, feat=f.replace(" ", "_").replace("/", "_").replace("&", "and"), x=x, y=y,
                         a=av, n_r=av, mu=pm[v][0] - med[f], s2=pm[v][1], raw=lraw[v] - medraw[f],
                         mu_test=pt[v][0] - medte[f], a_test=float(ate.get(v, 0)), phi=phi[f], med=med[f],
                         alpha=alpha[f], beta=beta[f], W=Wtr, phi_w=phi[f] / nweeks))
    users = tr.groupby("v").u.apply(set).to_dict()
    NU = tr.u.nunique()
    info = dict(city=city, n=len(rows), Wtr=round(Wtr, 1), Wte=round(Wte, 1), NU=NU, nweeks=nweeks,
                phi={k: (round(v, 1) if math.isfinite(v) else "inf") for k, v in phi.items()},
                alpha={k: round(v, 3) for k, v in alpha.items()}, beta={k: round(v, 3) for k, v in beta.items()})
    return rows, users, NU, info


def interactions(rows, users, NU, nb, kappa_i=5.0):
    U = [users.get(r["id"], set()) for r in rows]
    out = {}
    for i, S in enumerate(nb):
        for j in S:
            if i < j:
                cv = len(U[i] & U[j]); e = len(U[i]) * len(U[j]) / NU; den = cv + e + kappa_i
                out[(i, j)] = ((cv - e) / den, cv * ((2 * e + kappa_i) / den ** 2) ** 2)
    return out


if __name__ == "__main__":
    rows, users, NU, info = load(sys.argv[1], sys.argv[2])
    print(info)
    with open(sys.argv[3], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def recompute(rows, alpha_mult=1.0, quasi=False):
    """E8 variants of Def. 4: prior strength alpha_f scaled (beta_f scaled alike, so the prior mean rate is kept), or
    quasi-Poisson variance inflation 1 + rate/phi_w with phi_w the within-venue weekly dispersion."""
    post = [(float(digamma(r["alpha"] * alpha_mult + r["a"]) - math.log(r["beta"] * alpha_mult + r["W"])),
             float(polygamma(1, r["alpha"] * alpha_mult + r["a"]))) for r in rows]
    byf = collections.defaultdict(list)
    for r, (m, _) in zip(rows, post): byf[r["feat"]].append(m)
    med = {f: float(np.median(v)) for f, v in byf.items()}
    mu, s2 = [], []
    for r, (m, v) in zip(rows, post):
        infl = (1.0 + (r["a"] / r["W"]) / r["phi_w"]) if (quasi and math.isfinite(r["phi_w"]) and r["phi_w"] > 0) else 1.0
        mu.append(m - med[r["feat"]]); s2.append(v * infl)
    return mu, s2
