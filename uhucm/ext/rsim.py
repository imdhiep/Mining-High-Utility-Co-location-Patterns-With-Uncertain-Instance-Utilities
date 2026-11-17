"""Rating simulator (EviGen, Section VI-B): FDP and detection of UHUCPM, CQ-HUCPM and PSO-MUCP.

The data, the posterior, the truth (a pattern is true when the summed latent quality of its participating instances
reaches tau, the 70th percentile over all patterns) and the eight settings (correct model, t3 noise, prior mean shifted
by +-0.3, noise scale x0.7 / x1.3, spatially correlated quality, estimated hyperparameters) are those of synth.py e2.
Every method reports a pattern when its total reaches tau:
  uhucpm  lower bound at z* = 1.645
  cq      the authors' CQ-HUCPM at z = 0 with the posterior means as instance utilities; the exact z = 0 answer is used
          for every seed and the authors' code is run on the first NCHK seeds of each setting, recording agreement
  pso     the authors' PSO-MUCP at z = 0 with one utility per feature (the mean posterior mean of the feature), run on
          every seed; empty when no feature has a positive utility
Usage: python rsim.py [NSEED]   (default 100; env NCHK, default 3) -> out_rsim/rsim.json"""
import sys, os, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import paths, fastmine
from synth import evigen2, problem, truth_sets

ZS = 1.645; NS = int(sys.argv[1]) if len(sys.argv) > 1 else 100; NCHK = int(os.environ.get("NCHK", "3")); n, k = 400, 8
CQ = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "CQ-HUCPM.py"); PSO = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "PSO-MUCP.py")
L = "ABCDEFGH"; OUT = "out_rsim"; os.makedirs(OUT, exist_ok=True)
KNOBS = {"correct": {}, "t3": dict(noise="t3"), "prior+0.3": dict(prior_shift=0.3), "prior-0.3": dict(prior_shift=-0.3),
         "sigma x0.7": dict(sigma_err=0.7), "sigma x1.3": dict(sigma_err=1.3), "spatial": dict(spatial=0.5),
         "estimated": dict(estimate=True)}


def ext(cmd):
    p = subprocess.run([sys.executable, "-I"] + cmd, capture_output=True, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr[-2000:])
    return {tuple(sorted(str(L.index(c)) for c in s)) for s in json.loads(p.stdout.strip().splitlines()[-1])["patterns"]}


res = dict(n=n, k=k, nseed=NS, z=ZS, quantile=0.7, runs={})
for kn, kw in KNOBS.items():
    for seed in range(NS):
        G = evigen2(n, k, 1000 + seed, **kw)                       # the same draws as synth.py e2
        P0 = problem(G, G["pm"], G["pv"], 0.0)
        if P0.Z <= 0 or len(P0.cliques) == 0: continue
        R0all = fastmine.mine(P0, -1e9)[0]
        if len(R0all) < 20: continue
        tv = truth_sets(G, P0, R0all)
        tau = float(np.quantile(list(tv.values()), 0.7)); T = {p for p, v in tv.items() if v >= tau}
        Pz = problem(G, G["pm"], G["pv"], ZS)
        S_u = set(fastmine.mine(Pz, tau / Pz.Z)[0]); S_cq = set(fastmine.mine(P0, tau / P0.Z)[0])
        F = np.asarray(G["feat"]); mu = np.asarray(G["pm"], float)
        fv = {j: float(mu[F == j].mean()) for j in np.unique(F)}; Ufl = float(sum(max(fv[j], 0.0) for j in F))
        csvp = os.path.abspath(os.path.join(OUT, "rsim.csv")); ids = []; cc = {}
        with open(csvp, "w", encoding="utf-8") as f:
            f.write("Feature,Instance,LocX,LocY,Checkin\n")
            for j, (x, y) in zip(F.tolist(), np.asarray(G["xy"])):
                cc[j] = cc.get(j, 0) + 1; ids.append(f"{L[j]}{cc[j]}"); f.write(f"{L[j]},{cc[j]},{x:.6f},{y:.6f},0\n")
        S_pso = ext([os.path.join(HERE, "run_pso.py"), PSO, csvp, repr({L[j]: v for j, v in fv.items()}),
                     repr(float(tau / Ufl * (1 - 1e-9))), str(float(G["d"]))]) if Ufl > 0 and tau > 0 else set()
        chk = None
        if seed < NCHK:
            uj = os.path.abspath(os.path.join(OUT, "rsim_u.json")); json.dump({i: float(x) for i, x in zip(ids, mu)}, open(uj, "w"))
            Upos = float(np.clip(mu, 0, None).sum())
            S_ext = ext([os.path.join(HERE, "run_cq.py"), CQ, csvp, uj, repr(float(tau / Upos * (1 - 1e-9))), str(float(G["d"]))])
            chk = dict(equal=S_ext == S_cq, n_ext=len(S_ext))

        def fd(S):
            m = len(S); return dict(n=m, fdp=round(1 - len(S & T) / m, 4) if m else 0.0, det=round(len(S & T) / max(1, len(T)), 4))
        res["runs"][f"{kn}|{seed}"] = dict(nT=len(T), tau=tau, uhucpm=fd(S_u), cq=fd(S_cq), pso=fd(S_pso), cq_check=chk)
    print(kn, {m: round(float(np.mean([r[m]["fdp"] for key, r in res["runs"].items() if key.startswith(kn + "|")])), 4)
               for m in ("uhucpm", "cq", "pso")}, flush=True)
    json.dump(res, open(os.path.join(OUT, "rsim.json"), "w"), indent=1)
print("DONE", flush=True)
