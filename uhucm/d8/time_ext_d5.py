"""Running time of UHUCPM at z = 0 against the authors' CQ-HUCPM and PSO-MUCP on D5 (TKY), for one K per Kaggle
session. It follows time_ext_fsq.py (the D4 protocol: separate -I processes reading the same files, timed from process
start to exit with their CPU time, the machine to itself, the result sets compared), with one change for CQ-HUCPM, whose
single run on D5 takes hours: CQ-HUCPM gets one timed run and no warm-up, because the start-up costs that a warm-up
removes (file cache, bytecode compilation) take about a second. The fast programs (UHUCPM at both utility levels and
PSO-MUCP) first get one untimed warm-up each and REPS timed rounds in rotated order, on the same machine; CQ-HUCPM then
runs once. A CQ-HUCPM run longer than CQ_TIMEOUT seconds (default 40000, 11.1 h, so that the session ends within
Kaggle's 12-hour limit and keeps its output) is stopped and recorded as a lower bound on its time.
Usage: python time_ext_d5.py K   -> out_timing_ext/time_ext_TKY.json"""
import sys, os, json, time, resource, subprocess, string
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import fsq_loader, paths
from suite import Suite

city = "TKY"; Ks = [int(sys.argv[1])]
REPS = int(os.environ.get("REPS", "5")); CQ_TIMEOUT = float(os.environ.get("CQ_TIMEOUT", "40000")); d = 50.0
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


def once(cmd, timeout=None):
    c0, t0 = child_cpu(), time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    wall, cpu = time.perf_counter() - t0, child_cpu() - c0
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    r = json.loads(p.stdout.strip().splitlines()[-1])
    return wall, cpu, r["seconds"], sorted(r["patterns"])


fl_done = {}
res = dict(city=city, n=len(rows), d=d, reps=REPS, cq_reps=1, cq_warmup=False, cq_timeout=CQ_TIMEOUT, U_all=U_all, U_fl=U_fl, util_fl=util_fl, K={})
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
    order = [p for p in progs if p != "cq"]
    for it in range(-1, REPS):                                         # fast programs: warm-up (it = -1), then REPS rounds
        k = 0 if it < 0 else it % len(order)
        for p in order[k:] + order[:k]:
            w, c, inner, pats = once(progs[p])
            if sets[p] is None: sets[p] = pats
            elif sets[p] != pats: rec[p]["unstable"] = True
            if it >= 0:
                rec[p]["times"].append(round(w, 3)); rec[p]["cpu"].append(round(c, 3)); rec[p]["inner"].append(inner)
            print(city, K, it, p, round(w, 2), flush=True)
    print(city, K, "cq start", time.strftime("%H:%M:%S"), flush=True)
    t0 = time.perf_counter(); r = once(progs["cq"], CQ_TIMEOUT)          # CQ-HUCPM: one timed run, no warm-up
    if r is None:
        rec["cq"]["timeout"] = True; rec["cq"]["at_least"] = round(time.perf_counter() - t0, 1)
        print(city, K, "cq timeout after", rec["cq"]["at_least"], flush=True)
    else:
        w, c, inner, pats = r; sets["cq"] = pats
        rec["cq"]["times"].append(round(w, 3)); rec["cq"]["cpu"].append(round(c, 3)); rec["cq"]["inner"].append(inner)
        print(city, K, "cq", round(w, 2), flush=True)
    for p in progs: rec[p]["n"] = None if sets[p] is None else len(sets[p])
    cmp = {}
    for a, b in (("uhucpm_inst", "cq"), ("uhucpm_fl", "psomucp")):
        if sets.get(a) is not None and sets.get(b) is not None:
            A, B = set(sets[a]), set(sets[b]); cmp[f"{a}_vs_{b}"] = dict(same=A == B, missed_by_b=sorted(A - B), extra_in_b=sorted(B - A))
    if kf not in fl_done: fl_done[kf] = K
    res["K"][str(K)] = dict(minutil_inst=m_i, minutil_fl=m_f, K_fl=kf, fl_same_as_K=fl_done[kf] if fl_done[kf] != K else None,
                            programs=rec, compare=cmp)
    print(city, K, json.dumps({p: dict(n=r["n"], t=r["times"], to=r["timeout"]) for p, r in rec.items()}), json.dumps(cmp)[:400], flush=True)
    json.dump(res, open(os.path.join(out, f"time_ext_{city}.json"), "w"), indent=1)
print("DONE", flush=True)
