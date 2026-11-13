"""Yelp with the same pipeline as Foursquare, on the same machine: rating posterior (eq. rating), total variance V_T
with B = 20 bootstrap draws of the estimated hyperparameters (yelp_full.vt_boot), expected normalizer, z = 1.645.
Training ratings before SPLIT, test ratings from SPLIT to TEST_END.
Usage: python run_yelp_fair.py ROOT REVIEW_DIR CITY d [sens]
  ROOT        folder with yelp_academic_dataset_business.json and yelp_academic_dataset_checkin.json
  REVIEW_DIR  folder with reviews_<City>.csv written by prep_yelp_reviews.py
  sens        also the sensitivity of the answer set to kappa_f, sigma_s and w_o (V_T recomputed under each variant)
-> out_yelp/suite_YELP_<City>[_d<d>].json"""
import sys, os, functools
import numpy as np
import yelp_full
from suite import Suite

SPLIT, TEST_END, B = "2019-01-01", "2020-01-01", 20
root, rdir, city, d = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); sens = "sens" in sys.argv[5:]
data = yelp_full.build(root, [city], split=SPLIT, test_end=TEST_END, log=lambda s: print(s, flush=True), review_dir=rdir)[city]
rows = data["rows"]
UB = yelp_full.vt_boot(rows, B=B, seed=0); Zhat = yelp_full.expected_pos([r["mu"] for r in rows], [r["s2"] for r in rows])


def variant(**kw):
    mu, s2 = yelp_full.recompute(rows, data["hyper"], data["sigma_s2"], **kw)
    U = yelp_full.vt_boot(rows, B=B, seed=0, **{k: v for k, v in kw.items() if k in ("kappa_mult", "sigma_mult", "g_kind")})
    return mu, s2, U, yelp_full.expected_pos(mu, s2)


variants = [("kappa_x0.25", dict(kappa_mult=0.25)), ("kappa_x0.5", dict(kappa_mult=0.5)), ("kappa_x2", dict(kappa_mult=2.0)),
            ("kappa_x4", dict(kappa_mult=4.0)), ("sigma_x0.7", dict(sigma_mult=0.7)), ("sigma_x1.3", dict(sigma_mult=1.3)),
            ("g_const", dict(g_kind="const"))]
tag = "YELP_" + city.replace(" ", "_") + ("" if d == 50.0 else f"_d{int(d)}")
S = Suite(tag, rows, d, "out_yelp", recompute=variant, variants=variants, users=data["users"], NU=data["NU"],
          interactions=yelp_full.interactions, resid=[r["theta"] - data["hyper"][r["feat"]]["mu_f"] for r in rows],
          log=lambda s: print(s, flush=True), zmain=1.645, UB=UB, Zhat=Zhat)
S.rec["hyper"] = dict(sigma_s2=data["sigma_s2"], kappa=[h["kappa"] for h in data["hyper"].values()], B=B, split=SPLIT, test_end=TEST_END)
S.quality()
if sens: S.sensitivity()
S.save(); print("DONE", flush=True)
