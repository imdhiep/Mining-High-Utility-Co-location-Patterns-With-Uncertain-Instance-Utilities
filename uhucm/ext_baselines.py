"""External baselines on any atom table (rows with feat, x, y, mu, s2, mu_test), shared by Foursquare and Yelp runs.
- feature-level HUCPM with the authors' PSO-MUCP code (v(f) = mean posterior mean of f), top-K by feature-level score;
- instance-level CQ-HUCPM (repository implementation) with posterior means, at the K-th point-utility threshold;
- peak memory (tracemalloc) of building and mining for UHUCM and NoPrune at the K = 300 threshold."""
import os, sys, csv, json, time, string, subprocess
import numpy as np
from fastmine import mine, pattern_atoms
from suite import Suite
import memtrack
H = os.path.dirname(os.path.abspath(__file__))
L = string.ascii_uppercase


def _csv(S, out, tag):
    lab = {j: L[j] for j in range(len(S.names))}; p = os.path.join(out, f"{tag}_authors.csv"); ids = []; cnt = {}
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Feature", "Instance", "LocX", "LocY", "Checkin"])
        for j, (x, y) in zip(S.F, S.XY):
            cnt[j] = cnt.get(j, 0) + 1; w.writerow([lab[j], cnt[j], round(x, 3), round(y, 3), 0]); ids.append(f"{lab[j]}{cnt[j]}")
    return p, lab, ids


def run_all(rows, tag, out, src_dir, d=50.0, Ks=(100, 300), log=print, do_authors=True):
    S = Suite(tag, rows, d, out, log=lambda s: None)
    P0 = S.mk(); Pz = S.mk(z=1.645); R0all, sc = S.topk(P0, max(Ks))
    has_t = ~np.isnan(S.mut); Zt = float(np.clip(S.mut[has_t], 0, None).sum())
    def te(p):
        A = pattern_atoms(P0, p); return float(np.nansum(S.mut[A][has_t[A]]) / Zt)
    jac = lambda a, b: round(len(a & b) / max(1, len(a | b)), 3)
    F = np.array(S.F); v = {j: float(S.mu[F == j].mean()) for j in range(len(S.names))}
    mu_fl = np.array([v[j] for j in S.F]); Pfl = S.mk(mu=mu_fl, s2=np.zeros_like(mu_fl)); Rfl_all, scfl = S.topk(Pfl, max(Ks))
    csvp, lab, ids = _csv(S, out, tag)
    util = {lab[j]: v[j] for j in v}; U_all = float(sum(v[j] * int((F == j).sum()) for j in v))
    res = dict(tag=tag, n=len(rows), U_all_fl=U_all, fl={}, cq={}, mem={})
    for K in Ks:
        muK = sc[K - 1]; R0 = {p for p, s in R0all.items() if s >= muK}; Rz = set(mine(Pz, muK)[0])
        mufl = scfl[K - 1]; Rfl = {p for p, s in Rfl_all.items() if s >= mufl}
        r = dict(J_fl_R0=jac(Rfl, R0), J_fl_Rz=jac(Rfl, Rz), fl_inst_hucp=round(float(np.mean([P0.score(pattern_atoms(P0, p)) >= muK for p in Rfl])), 3),
                 hold_fl=round(float(np.mean([te(p) >= muK for p in Rfl])), 3), hold_R0=round(float(np.mean([te(p) >= muK for p in R0])), 3),
                 hold_Rz=round(float(np.mean([te(p) >= muK for p in Rz])), 3))
        if do_authors and U_all > 0:
            cmd = [sys.executable, "-I", os.path.join(H, "d8", "run_authors.py"), os.path.join(src_dir, "PSO-MUCP.py"), csvp, repr(util),
                   repr(float(mufl * Pfl.Z / U_all * (1 - 1e-6))), str(d)]
            a = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[-1])
            pa = {tuple(sorted(S.names[L.index(c)] for c in s)) for s in a.get("patterns", [])}
            r.update(authors_n=a.get("n"), authors_seconds=a.get("seconds"), authors_missing=len(Rfl - pa), authors_extra=len(pa - Rfl),
                     authors_extra_below=sum(1 for p in pa - Rfl if Rfl_all.get(p, 0) < mufl))
        res["fl"][K] = r; log(f"X|{tag}|FL|K={K}|" + json.dumps(r))
    # CQ-HUCPM with instance utilities (point), at K = 300
    K = max(Ks); muK = sc[K - 1]; R0 = {p for p, s in R0all.items() if s >= muK}
    U_all_i = float(S.mu.sum()); up = os.path.join(out, f"{tag}_u_point.json"); json.dump({i: float(x) for i, x in zip(ids, S.mu)}, open(up, "w"))
    r = dict(U_all=U_all_i)
    if U_all_i > 0:
        cmd = [sys.executable, "-I", os.path.join(H, "d8", "run_cq.py"), os.path.join(src_dir, "CQ-HUCPM.py"), csvp, up,
               repr(float(muK * P0.Z / U_all_i * (1 - 1e-6))), str(d)]
        try:
            o = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=5400).stdout.strip().splitlines()[-1])
            pc = {tuple(sorted(S.names[L.index(c)] for c in s)) for s in o["patterns"]}
            t = time.time(); _ = mine(P0, muK); t_ours = time.time() - t
            r.update(n=o["n"], seconds=o["seconds"], ours_z0_seconds=round(t_ours, 2), missing=len(R0 - pc), extra=len(pc - R0),
                     extra_below=sum(1 for p in pc - R0 if P0.score(pattern_atoms(P0, p)) < muK - 1e-12))
        except subprocess.TimeoutExpired:
            r["timeout"] = 5400
    res["cq"] = r; log(f"X|{tag}|CQ|K={K}|" + json.dumps(r))
    # peak memory (tracemalloc) of build + mine at the K = 300 threshold
    for name, use in (("uhucm", (1, 1, 1, 1)), ("noprune", (0, 0, 0, 0))):
        memtrack.start(); t = time.time(); Pm = S.mk(z=1.645); R, st, _ = mine(Pm, muK, use, "two"); el = time.time() - t
        res["mem"][name] = dict(peak_mb=round(memtrack.stop_mb(), 1), seconds_traced=round(el, 2), evaluated=st["evaluated"]); del Pm
    log(f"X|{tag}|MEM|" + json.dumps(res["mem"]))
    json.dump(res, open(os.path.join(out, f"ext_{tag}.json"), "w"), indent=1, default=str)
    return res
