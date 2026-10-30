"""Quality experiments (E0, E3, E7) on Foursquare at other distance thresholds. Usage: python run_fsq_d.py TXT CITY OUT d"""
import sys, functools
import fsq_loader
from suite import Suite
path, city, out, d = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
rows, users, NU, info = fsq_loader.load(path, city)
S = Suite(f"FSQ_{city}_d{int(d)}", rows, d, out, recompute=functools.partial(fsq_loader.recompute, rows),
          users=users, NU=NU, interactions=fsq_loader.interactions, resid=[r["mu"] for r in rows], log=lambda s: print(s, flush=True))
S.quality(); S.save(); print("DONE", flush=True)
