"""Running time of UHUCPM at z = 0 against existing HUCPM miners on D4 (NYC) and D5 (TKY), under the protocol of
time_d8_fair.py: the job has the machine to itself, each program runs as a separate Python process (started with -I)
that reads the same input files, and is timed from process start to exit (wall clock) together with its CPU time, so
data loading, neighbor search, clique enumeration and mining count the same way for every program. Each configuration
gets one untimed warm-up run of every program, then REPS timed rounds in rotated order. A program that exceeds
TIMEOUT seconds is recorded as a timeout and not run again for that configuration.
  * Instance-level utilities (the posterior means, as Point utility in the paper): UHUCPM against CQ-HUCPM.
  * Feature-level utilities (the mean posterior mean of each feature, as PSO-MUCP assigns one utility per feature):
    UHUCPM against PSO-MUCP. Both use the authors' code from the PSO-MUCP repository with its mining logic unchanged.
The threshold of each K is the K-th largest point score (with feature-level utilities few patterns have a positive
score, so the threshold is the smallest positive score when fewer than K patterns have one, and a threshold already
timed for a smaller K is not timed again), given to every program as a share of the total positive
utility of the instances (the sum of all utilities, which the authors' code divides by, is negative on D4 and D5).
UHUCPM reads the feature-level utilities from a second utility file, expanded to the instances.
The posteriors are computed once before timing and are not part of the times. The result sets of every run are compared.
Usage: python time_ext_fsq.py CITY [K ...]   (default K = 100 300 1000)  -> out_timing_ext/time_ext_<CITY>.json"""
import sys, os, json, time, resource, subprocess, string
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import fsq_loader, paths
from suite import Suite

city = sys.argv[1]; Ks = [int(x) for x in sys.argv[2:]] or [100, 300, 1000]
REPS = int(os.environ.get("REPS", "5")); TIMEOUT = float(os.environ.get("TIMEOUT", "3600")); d = 50.0
CQ = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "CQ-HUCPM.py"); PSO = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "PSO-MUCP.py")
out = "out_timing_ext"; os.makedirs(out, exist_ok=True)

rows, users, NU, info = fsq_loader.load(paths.fsq(city), city, model="ceb"); info.pop("UB"); info.pop("Zhat")
S = Suite(f"EXT_{city}", rows, d, out, log=lambda s: None)
L = string.ascii_uppercase
csvp = os.path.abspath(os.path.join(out, f"{city}.csv")); ids = []; cnt = {}
with open(csvp, "w") as f:
    f.write("Feature,Instance,LocX,LocY,Checkin\n")
    for j, (x, y) in zip(S.F, S.XY):
        cnt[j] = cnt.get(j, 0) + 1; ids.append(f"{L[j]}{cnt[j]}"); f.write(f"{L[j]},{cnt[j]},{x:.3f},{y:.3f},0\n")
# instance level: posterior means; feature level: mean posterior mean of each feature
P0 = S.mk(mu=S.mu, s2=np.zeros_like(S.mu)); U_all = float(np.clip(S.mu, 0, None).sum())
ujson = os.path.abspath(os.path.join(out, f"{city}_u.json")); json.dump({i: float(x) for i, x in zip(ids, S.mu)}, open(ujson, "w"))
F = np.array(S.F); v = {j: float(S.mu[F == j].mean()) for j in range(len(S.names))}
mu_fl = np.array([v[j] for j in S.F]); Pfl = S.mk(mu=mu_fl, s2=np.zeros_like(mu_fl)); U_fl = float(np.clip(mu_fl, 0, None).sum())
ufl = os.path.abspath(os.path.join(out, f"{city}_ufl.json")); json.dump({i: float(x) for i, x in zip(ids, mu_fl)}, open(ufl, "w"))
util_fl = {L[j]: v[j] for j in v}
R0, sc0 = S.topk(P0, max(Ks)); Rf, scf = S.topk(Pfl, max(Ks))
print("prep", city, "n", len(rows), "U_pos", U_all, "U_fl_pos", U_fl, "sum", float(S.mu.sum()), flush=True)


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime


def once(cmd):
    c0, t0 = child_cpu(), time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    wall, cpu = time.perf_counter() - t0, child_cpu() - c0
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    r = json.loads(p.stdout.strip().splitlines()[-1])
    return wall, cpu, r["seconds"], sorted(r["patterns"])


fl_done = {}
res = dict(city=city, n=len(rows), d=d, reps=REPS, timeout=TIMEOUT, U_all=U_all, U_fl=U_fl, util_fl=util_fl, K={})
for K in Ks:
    kf = min(K, sum(1 for x in scf if x > 0))
    m_i = float(sc0[K - 1] * P0.Z / U_all * (1 - 1e-9)); m_f = float(scf[kf - 1] * Pfl.Z / U_fl * (1 - 1e-9))
    py = [sys.executable, "-I"]
    progs = {"uhucpm_inst": py + [os.path.join(HERE, "run_uhucm_inst.py"), csvp, ujson, repr(m_i), str(d)],
             "cq": py + [os.path.join(HERE, "run_cq_fair.py"), CQ, csvp, ujson, repr(m_i), str(d)],
             "uhucpm_fl": py + [os.path.join(HERE, "run_uhucm_inst.py"), csvp, ufl, repr(m_f), str(d)],
             "psomucp": py + [os.path.join(HERE, "run_pso_fair.py"), PSO, csvp, repr(util_fl), repr(m_f), str(d)]}
    if kf in fl_done:                                              # same feature-level threshold as a smaller K
        for p in ("uhucpm_fl", "psomucp"): progs.pop(p)
    rec = {p: dict(times=[], cpu=[], inner=[], timeout=False) for p in progs}; sets = {p: None for p in progs}
    order = list(progs)
    for it in range(-1, REPS):                                         # it = -1: untimed warm-up of every program
        k = 0 if it < 0 else it % len(order)
        for p in order[k:] + order[:k]:
            if rec[p]["timeout"]: continue
            r = once(progs[p])
            if r is None:
                rec[p]["timeout"] = True; print(city, K, p, "timeout", flush=True); continue
            w, c, inner, pats = r
            if sets[p] is None: sets[p] = pats
            elif sets[p] != pats: rec[p]["unstable"] = True
            if it >= 0:
                rec[p]["times"].append(round(w, 3)); rec[p]["cpu"].append(round(c, 3)); rec[p]["inner"].append(inner)
            print(city, K, it, p, round(w, 2), flush=True)
    for p in progs: rec[p]["n"] = None if sets[p] is None else len(sets[p])
    cmp = {}
    for a, b in (("uhucpm_inst", "cq"), ("uhucpm_fl", "psomucp")):
        if sets[a] is not None and sets[b] is not None:
            A, B = set(sets[a]), set(sets[b]); cmp[f"{a}_vs_{b}"] = dict(same=A == B, missed_by_b=sorted(A - B), extra_in_b=sorted(B - A))
    if kf not in fl_done: fl_done[kf] = K
    res["K"][str(K)] = dict(minutil_inst=m_i, minutil_fl=m_f, K_fl=kf, fl_same_as_K=fl_done[kf] if fl_done[kf] != K else None,
                            programs=rec, compare=cmp)
    print(city, K, json.dumps({p: dict(n=r["n"], t=r["times"], to=r["timeout"]) for p, r in rec.items()}), json.dumps(cmp)[:400], flush=True)
    json.dump(res, open(os.path.join(out, f"time_ext_{city}.json"), "w"), indent=1)
print("DONE", flush=True)
