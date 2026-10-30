"""Re-run the repository's CQ-HUCPM (memoized runner d8/run_cq_memo.py, output unchanged) at the K = 300 point threshold
and compare with R_0. Usage: python cq_rerun.py CITY  -> prints an X|FSQ_CITY|CQ|K=300|{...} line."""
import os, sys, json, time, subprocess
import numpy as np, fsq_loader, paths
from suite import Suite
from fastmine import mine, pattern_atoms
from ext_baselines import _csv, H, L
c = sys.argv[1]; tag = f"FSQ_{c}"; out = "out_baseline"; d = 50.0; K = 300
src = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "CQ-HUCPM.py")
rows, u, NU, info = fsq_loader.load(paths.fsq(c), c)
S = Suite(tag, rows, d, out, log=lambda s: None); P0 = S.mk(); R0all, sc = S.topk(P0, K)
muK = sc[K - 1]; R0 = {p for p, s in R0all.items() if s >= muK}
csvp, lab, ids = _csv(S, out, tag)
U_all_i = float(S.mu.sum()); up = os.path.join(out, f"{tag}_u_point.json"); json.dump({i: float(x) for i, x in zip(ids, S.mu)}, open(up, "w"))
cmd = [sys.executable, "-I", os.path.join(H, "d8", "run_cq_memo.py"), src, csvp, up, repr(float(muK * P0.Z / U_all_i * (1 - 1e-6))), str(d)]
o = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[-1])
pc = {tuple(sorted(S.names[L.index(x)] for x in s)) for s in o["patterns"]}
r = dict(U_all=U_all_i, n=o["n"], seconds=o["seconds"], runner="memo", missing=len(R0 - pc), extra=len(pc - R0),
         extra_below=sum(1 for p in pc - R0 if P0.score(pattern_atoms(P0, p)) < muK - 1e-12))
print(f"X|{tag}|CQ|K={K}|" + json.dumps(r), flush=True)
