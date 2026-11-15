"""Scalability in the number of instances on EviGen (k = 10, seed 7, n = 10^3 ... 10^5, K = 300, z = 1.645), with the
timing protocol of the other experiments: the job has the machine to itself, one untimed warm-up, then 5 timed rounds
of the clique build and of the mining step in alternating order, garbage collector off inside the timed region, wall
clock and CPU time. The growth exponents are least-squares slopes of log mean time against log n.
Usage: python run_scale_fair.py   -> out_scale/scale_n.json"""
import gc, json, os, time
import numpy as np
import fastmine
from synth import evigen2, problem, topk_mu

REPS, NS, K = 5, (1000, 3000, 10000, 30000, 100000), 300


def timed(f):
    gc.collect(); gc.disable()
    t, c = time.perf_counter(), time.process_time()
    r = f()
    t, c = time.perf_counter() - t, time.process_time() - c
    gc.enable()
    return r, round(t, 4), round(c, 4)


res = {}
for n in NS:
    G = evigen2(n, 10, 7)
    muK = topk_mu(problem(G, G["pm"], G["pv"], 0.0), K)                # threshold of the K-th point score, untimed
    rec = {"build": dict(times=[], cpu=[]), "mine": dict(times=[], cpu=[])}
    Pz = None
    for it in range(-1, REPS):
        steps = ("build", "mine") if it % 2 == 0 or it < 0 or Pz is None else ("mine", "build")
        for s in steps:
            if s == "build":
                Pz, t, c = timed(lambda: problem(G, G["pm"], G["pv"], 1.645))
            else:
                (R, st, _), t, c = timed(lambda: fastmine.mine(Pz, muK, (1, 1, 1, 1), "two"))
                rec["mine"].update(evaluated=st["evaluated"], R=len(R))
            if it >= 0:
                rec[s]["times"].append(t); rec[s]["cpu"].append(c)
    rec.update(n=n, cliques=len(Pz.cliques), omega=max(map(len, Pz.cliques)) if Pz.cliques else 0, mu=muK)
    res[str(n)] = rec
    print("SCALE", n, json.dumps({s: (round(float(np.mean(rec[s]["times"])), 3)) for s in ("build", "mine")}), rec["cliques"], flush=True)
x = np.log([float(n) for n in NS])
expo = {s: round(float(np.polyfit(x, np.log([np.mean(res[str(n)][s]["times"]) for n in NS]), 1)[0]), 3) for s in ("build", "mine")}
os.makedirs("out_scale", exist_ok=True)
json.dump(dict(k=10, seed=7, K=K, reps=REPS, exponents=expo, timing=res), open("out_scale/scale_n.json", "w"), indent=1)
print("exponents", expo, "\nDONE", flush=True)
