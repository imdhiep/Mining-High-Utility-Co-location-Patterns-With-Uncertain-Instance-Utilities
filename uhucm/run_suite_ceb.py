"""Suite on Foursquare with the compound empirical Bayes model (Def. 4), total variance and expected normalizer.
Usage: python run_suite_ceb.py TXT CITY OUT [parts] [d]   parts: comma list of quality,dependence,efficiency,timing; d in metres (default 50)"""
import sys, numpy as np
import fsq_loader
from suite import Suite
path, city, out = sys.argv[1], sys.argv[2], sys.argv[3]; parts = (sys.argv[4] if len(sys.argv) > 4 else "quality").split(","); d = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
rows, users, NU, info = fsq_loader.load(path, city, model="ceb")
UB, Zhat = info.pop("UB"), info.pop("Zhat")
print("## info", {k: v for k, v in info.items() if k in ("city", "n", "Wtr", "Wte", "NU")}, "Zhat", round(Zhat, 1), flush=True)
S = Suite("CEB_" + city + ("" if d == 50.0 else f"_d{int(d)}"), rows, d, out, users=users, NU=NU, interactions=fsq_loader.interactions,
          resid=[r["mu"] for r in rows], log=lambda s: print(s, flush=True), zmain=1.645, UB=UB, Zhat=Zhat)
for p in parts: getattr(S, p)()
S.save(); print("DONE", flush=True)
