"""Run the full experiment suite on Foursquare TSMC2014 (NYC or TKY). Usage: python run_suite_fsq.py TXT CITY OUT [d]"""
import sys, functools, numpy as np
import fsq_loader
from suite import Suite
path, city, out = sys.argv[1], sys.argv[2], sys.argv[3]; d = float(sys.argv[4]) if len(sys.argv) > 4 else 50.0
rows, users, NU, info = fsq_loader.load(path, city)
print("## info", info, flush=True)
S = Suite("FSQ_" + city, rows, d, out,
          recompute=functools.partial(fsq_loader.recompute, rows),
          variants=[("alpha_x0.5", dict(alpha_mult=0.5)), ("alpha_x2", dict(alpha_mult=2.0)), ("quasi", dict(quasi=True))],
          users=users, NU=NU, interactions=fsq_loader.interactions, resid=[r["mu"] for r in rows],
          log=lambda s: print(s, flush=True))
S.quality(); S.interaction(); S.sensitivity(); S.dependence(); S.efficiency(); S.save()
print("DONE", flush=True)
