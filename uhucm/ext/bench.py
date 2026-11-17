"""Running time of UHUCPM against the authors' CQ-HUCPM and PSO-MUCP for the efficiency experiments of Section VI-C,
under one protocol: every program runs as a separate Python process (started with -I) that reads the same instance file,
and is timed from process start to exit (wall clock, with its CPU time), so loading, neighbor search, clique enumeration
and mining count the same way for every program; the job has the machine to itself (fair/run.py, exclusive).
Programs, all at the same rank threshold gamma_K (the K-th largest point score):
  uhucpm     UHUCPM at z* = 1.645 with the total variance V_T and the expected normalizer (the paper's setting)
  uhucpm0    the same search at z = 0 (posterior means as exact utilities), the problem that CQ-HUCPM solves
  meanbound  UHUCPM with the mean bound in place of the exact subtree bound
  noprune    the same search at z* without any bound
  cq         the authors' CQ-HUCPM at z = 0 with the posterior means as instance utilities
  pso        the authors' PSO-MUCP at z = 0 with one utility per feature, the mean posterior mean of the feature
             (when fewer than K patterns have a positive feature-level score, its threshold is the smallest positive one)
The four UHUCPM programs get one untimed warm-up and REPS timed rounds in rotated order. CQ-HUCPM and PSO-MUCP run
once first; a first run shorter than QUICK seconds counts as their warm-up and REPS timed runs follow, a longer one is
their single measurement. A run longer than CAP seconds is stopped and the program is recorded as not finished for that
configuration (and is not run again at larger thresholds of the same configuration). The answer of uhucpm0 is compared
with that of CQ-HUCPM.
Usage:
  python bench.py LAYOUT k d frac K [K ...]   LAYOUT in NYC, TKY (Foursquare D4, D5) or Philadelphia, New_Orleans, Tampa
                                              (Yelp D1-D3, env YELP_DIR and REVIEW_DIR); k most frequent features,
                                              distance d (m), a random share frac of the instances (seed 7)
  python bench.py sim n K [K ...]             the simulator of the scalability experiment (k = 10, seed 7), n instances
-> out_bench/bench_<tag>.json, written after every K"""
import sys, os, json, time, resource, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import paths

REPS = int(os.environ.get("REPS", "5")); QUICK = float(os.environ.get("QUICK", "600")); CAP = float(os.environ.get("CAP", "7200"))
ZS = 1.645
CQ = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "CQ-HUCPM.py"); PSO = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "PSO-MUCP.py")
out = "out_bench"; os.makedirs(out, exist_ok=True)
ALPHA = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)] + [chr(0x4E00 + i) for i in range(200)]

src = sys.argv[1]
from fastmine import Problem, mine
if src == "sim":
    from synth import evigen2
    n = int(sys.argv[2]); Ks = [int(x) for x in sys.argv[3:]] or [300]
    G = evigen2(n, 10, 7); F, XY, mu, s2, d = np.asarray(G["feat"]), [tuple(p) for p in np.asarray(G["xy"])], np.asarray(G["pm"], float), np.asarray(G["pv"], float), float(G["d"])
    UB = None; Z = None; tag = f"sim_n{n}"
else:
    k, d, frac = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); Ks = [int(x) for x in sys.argv[5:]]
    if src in ("NYC", "TKY"):
        import fsq_loader
        from suite import Suite
        rows, users, NU, info = fsq_loader.load(paths.fsq(src), src, nfeat=k, model="ceb")
        S = Suite(f"B_{src}", rows, d, out, log=lambda s: None, UB=info.pop("UB"), Zhat=info.pop("Zhat"))
        F, XY, mu, s2, UB, Z = np.array(S.F), S.XY, S.mu, S.s2, np.asarray(S.UB, float), float(S.Zhat); EP = info["EP"]
    else:
        import yelp_full
        city = src.replace("_", " ")
        rows = yelp_full.build(os.environ["YELP_DIR"], [city], split="2019-01-01", test_end="2020-01-01", log=lambda s: None,
                               review_dir=os.environ.get("REVIEW_DIR"))[city]["rows"]
        cnt = {}
        for r in rows: cnt[r["feat"]] = cnt.get(r["feat"], 0) + 1
        top = set(sorted(cnt, key=lambda f: (-cnt[f], f))[:k]); rows = [r for r in rows if r["feat"] in top]
        names = sorted(top); F = np.array([names.index(r["feat"]) for r in rows]); XY = [(r["x"], r["y"]) for r in rows]
        mu = np.array([r["mu"] for r in rows], float); s2 = np.array([r["s2"] for r in rows], float)
        UB = yelp_full.vt_boot(rows, B=20); Z = yelp_full.expected_pos(mu, s2)
    if frac < 1:                                              # random share of the instances; V_T and Z of the kept ones
        keep = np.sort(np.random.default_rng(7).choice(len(F), int(round(frac * len(F))), replace=False))
        F, XY, mu, s2 = F[keep], [XY[i] for i in keep], mu[keep], s2[keep]
        if UB is not None:
            UB = UB[:, keep]
            Z = float(EP[keep].sum()) if src in ("NYC", "TKY") else yelp_full.expected_pos(mu, s2)
    tag = f"{src}_k{k}_d{int(d)}_f{int(round(100 * frac))}"


def mk(m, v, z, boot):
    return Problem(list(F), XY, m, v, d, z, feat_names=None)


def topk(Pb, K):
    m = 0.05
    while True:
        R = mine(Pb, m)[0]
        if len(R) >= K or m < 1e-12: return sorted(R.values(), reverse=True)
        m /= 2


P0 = mk(mu, np.zeros_like(mu), 0.0, False); Zn = P0.Z                                  # point scores, instance level
U_pos = float(np.clip(mu, 0, None).sum())
fv = {j: float(mu[F == j].mean()) for j in sorted(set(F.tolist()))}; mu_fl = np.array([fv[j] for j in F])
Pf = mk(mu_fl, np.zeros_like(mu_fl), 0.0, False); Zf = Pf.Z; U_fl = float(np.clip(mu_fl, 0, None).sum())
sc0 = topk(P0, max(Ks)); scf = topk(Pf, max(Ks))
csvp = os.path.abspath(os.path.join(out, f"{tag}.csv")); ids = []; cnt = {}
with open(csvp, "w", encoding="utf-8") as f:
    f.write("Feature,Instance,LocX,LocY,Checkin\n")
    for j, (x, y) in zip(F.tolist(), XY):
        cnt[j] = cnt.get(j, 0) + 1; ids.append(f"{ALPHA[j]}{cnt[j]}"); f.write(f"{ALPHA[j]},{cnt[j]},{x:.4f},{y:.4f},0\n")
uj = os.path.abspath(os.path.join(out, f"{tag}_u.json")); json.dump({i: float(x) for i, x in zip(ids, mu)}, open(uj, "w"))
npz = os.path.abspath(os.path.join(out, f"{tag}.npz"))
kw = dict(mu=mu, s2=s2) if UB is None else dict(mu=mu, s2=s2, UB=UB, Z=Z)
np.savez(npz, **kw)
util_fl = {ALPHA[j]: v for j, v in fv.items()}
print("prep", tag, "n", len(F), "k", len(fv), "cliques", len(P0.cliques), flush=True)


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime


def once(cmd, cap=None):
    c0, t0 = child_cpu(), time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
    except subprocess.TimeoutExpired:
        return None
    w, c = time.perf_counter() - t0, child_cpu() - c0
    if p.returncode != 0: raise RuntimeError(p.stderr[-2000:])
    r = json.loads(p.stdout.strip().splitlines()[-1])
    return round(w, 3), round(c, 3), r["seconds"], sorted(r["patterns"]), r.get("evaluated")


py = [sys.executable, "-I"]; res = dict(source=src, tag=tag, n=len(F), d=d, reps=REPS, quick=QUICK, cap=CAP, z=ZS, K={})
dead = set()                                                    # external programs that did not finish at a smaller K
for K in Ks:
    if K > len(sc0): continue
    m = sc0[K - 1]                                              # gamma_K on the normalizer Zn of P0
    kf = min(K, sum(1 for x in scf if x > 0)); mf = scf[kf - 1]
    thr_u = m * Zn / (Z if Z else Zn)                           # UHUCPM threshold on its own normalizer
    progs = {"uhucpm": py + [os.path.join(HERE, "run_uhucpm.py"), csvp, npz, repr(float(thr_u)), str(d), str(ZS), "exact"],
             "uhucpm0": py + [os.path.join(HERE, "run_uhucpm.py"), csvp, npz, repr(float(thr_u)), str(d), "0", "exact"],
             "meanbound": py + [os.path.join(HERE, "run_uhucpm.py"), csvp, npz, repr(float(thr_u)), str(d), str(ZS), "mean"],
             "noprune": py + [os.path.join(HERE, "run_uhucpm.py"), csvp, npz, repr(float(thr_u)), str(d), str(ZS), "none"],
             "cq": py + [os.path.join(HERE, "run_cq.py"), CQ, csvp, uj, repr(float(m * Zn / U_pos * (1 - 1e-9))), str(d)],
             "pso": py + [os.path.join(HERE, "run_pso.py"), PSO, csvp, repr(util_fl), repr(float(mf * Zf / U_fl * (1 - 1e-9))), str(d)]}
    rec = {p: dict(times=[], cpu=[], inner=[], finished=True) for p in progs}; sets = {}
    fast = ["uhucpm", "uhucpm0", "meanbound", "noprune"]
    for it in range(-1, REPS):
        k0 = 0 if it < 0 else it % len(fast)
        for p in fast[k0:] + fast[:k0]:
            w, c, inner, pats, ev = once(progs[p])
            sets.setdefault(p, pats); rec[p]["evaluated"] = ev
            if it >= 0: rec[p]["times"].append(w); rec[p]["cpu"].append(c); rec[p]["inner"].append(inner)
    for p in ("pso", "cq"):
        if p in dead: rec[p]["finished"] = False; rec[p]["skipped"] = True; continue
        r = once(progs[p], CAP)
        if r is None:
            rec[p]["finished"] = False; dead.add(p); print(tag, K, p, "not finished", flush=True); continue
        w, c, inner, pats, _ = r; sets[p] = pats
        if w >= QUICK:
            rec[p]["protocol"] = "single run"; rec[p]["times"].append(w); rec[p]["cpu"].append(c); rec[p]["inner"].append(inner)
        else:
            rec[p]["protocol"] = "warm-up + REPS"
            for _ in range(REPS):
                r = once(progs[p], CAP)
                if r is None: rec[p]["finished"] = False; dead.add(p); break
                w, c, inner, pats2, _ = r
                if pats2 != pats: rec[p]["unstable"] = True
                rec[p]["times"].append(w); rec[p]["cpu"].append(c); rec[p]["inner"].append(inner)
    for p in progs: rec[p]["n"] = len(sets[p]) if p in sets else None
    cmp = {}
    if "cq" in sets:                                            # z = 0 answer of UHUCPM's search against CQ-HUCPM's
        cmp["uhucpm0_equals_cq"] = set(sets["uhucpm0"]) == set(sets["cq"])
    cmp["uhucpm_equals_noprune"] = set(sets["uhucpm"]) == set(sets["noprune"])
    cmp["uhucpm_equals_meanbound"] = set(sets["uhucpm"]) == set(sets["meanbound"])
    res["K"][str(K)] = dict(gamma=float(m), K_fl=kf, programs=rec, compare=cmp)
    print(tag, K, json.dumps({p: (round(float(np.mean(r["times"])), 3) if r["times"] else None, r["finished"], r["n"]) for p, r in rec.items()}), flush=True)
    json.dump(res, open(os.path.join(out, f"bench_{tag}.json"), "w"), indent=1)
print("DONE", flush=True)
