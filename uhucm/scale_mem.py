"""Scalability with dispersion and peak memory. Each configuration runs in a fresh process (3 seeds for EviGen,
3 repetitions for D4/D5); peak resident memory is ru_maxrss of that process, reported together with the resident
memory after imports (baseline). Usage: python scale_mem.py [worker args]"""
import sys, os, json, time, resource, subprocess, statistics as st
import paths
H = os.path.dirname(os.path.abspath(__file__))

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def worker(kind, a, b, seed, method, mem=False):
    import numpy as np, gc, memtrack
    if kind == "syn":
        import synth, fastmine
        n, k = int(a), int(b); G = synth.evigen2(n, k, seed)
        P0 = synth.problem(G, G["pm"], G["pv"], 0.0); muK = synth.topk_mu(P0, 300); del P0; gc.collect()
        if mem: memtrack.start()
        t = time.time(); Pz = synth.problem(G, G["pm"], G["pv"], 1.645); tb = time.time() - t
    else:
        import fsq_loader, functools
        from suite import Suite
        rows, users, NU, info = fsq_loader.load(a, b)
        S = Suite("x", rows, 50.0, "/tmp", log=lambda s: None)
        P0 = S.mk(); R, sc = S.topk(P0, 300); muK = sc[299]; del P0, R; gc.collect()
        if mem: memtrack.start()
        t = time.time(); Pz = S.mk(z=1.645); tb = time.time() - t
    import fastmine
    use = (1, 1, 1, 1) if method == "uhucm" else (0, 0, 0, 0)
    t = time.time(); R, s, _ = fastmine.mine(Pz, muK, use, "two"); tm = time.time() - t
    print(json.dumps(dict(build_s=round(tb, 3), mine_s=round(tm, 3), evaluated=s["evaluated"], R=len(R), cliques=len(Pz.cliques),
                          peak_mb=(round(memtrack.stop_mb(), 1) if mem else None))))

def run(args):
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "worker"] + [str(x) for x in args], capture_output=True, text=True, cwd=H)
    return json.loads(out.stdout.strip().splitlines()[-1])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], len(sys.argv) > 7); sys.exit()
    res = []
    confs = [("syn", n, 10) for n in (1000, 3000, 10000, 30000, 100000)] + [("syn", 10000, k) for k in (5, 15, 20, 25, 30)]
    for kind, a, b in confs:
        for method in ("uhucm", "noprune"):
            reps = [run([kind, a, b, seed, method]) for seed in (7, 8, 9)]
            r = dict(kind=kind, n=a, k=b, method=method, reps=reps, mem=run([kind, a, b, 7, method, "mem"]))
            print(json.dumps(r), flush=True); res.append(r)
    for city in ("NYC", "TKY"):
        p = paths.fsq(city)
        for method in ("uhucm", "noprune"):
            reps = [run(["fsq", p, city, 0, method]) for _ in range(3)]
            r = dict(kind="fsq", n=city, k=None, method=method, reps=reps, mem=run(["fsq", p, city, 0, method, "mem"])); print(json.dumps(r), flush=True); res.append(r)
    json.dump(res, open(os.path.join(H, "out_synth", "scale_mem.json"), "w"), indent=1)
