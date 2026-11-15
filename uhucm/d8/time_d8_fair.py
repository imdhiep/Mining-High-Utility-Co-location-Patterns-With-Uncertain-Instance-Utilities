"""Timing of UHUCPM against the authors' PSO-MUCP code on the public PSO-MUCP data, with the timing protocol of the
other experiments: the job has the machine to itself, every configuration gets one untimed warm-up run of both
programs, then 5 timed rounds in alternating order. Each run is a separate Python process (run_uhucm_d8.py and
run_authors.py, both started with -I), timed from the parent from process start to exit (wall clock) together with the
CPU time of the child, so start-up, data loading, neighbor search and mining count the same way for both. The result
sets of every run are compared.
Configurations: the positive (degenerate) setting of run_d8_all.py, minimum utility 0.02, 0.05 and 0.1.
Usage: python time_d8_fair.py cal|lv   -> out_timing_d8/time_d8_<cal|lv>.json"""
import sys, os, json, time, resource, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import random
import paths

# utilities of the positive setting, as in run_d8_all.py (the authors' default for California, seeded draws for Las Vegas)
CAL = {'A': 2, 'B': 4, 'C': 8, 'D': 4, 'E': 1, 'F': 4, 'G': 5, 'H': 3, 'I': 9, 'J': 3, 'K': 5, 'L': 10, 'M': 8}
rnd = random.Random(2026)
LV = {f: rnd.randint(1, 10) for f in "ABCDEFGHIJ"}

REPS = 5
which = sys.argv[1]
name, D, util = {"cal": ("California_POI 13f.csv", 1300, CAL), "lv": ("LasVegas_POI 10f.csv", 100, LV)}[which]
data = os.path.join(paths.PSOMUCP_DIR, "DATA", name)
src = os.path.join(paths.PSOMUCP_DIR, "HUCPM", "PSO-MUCP.py")


def cmd(prog, mu):
    if prog == "uhucpm":
        return [sys.executable, "-I", os.path.join(HERE, "run_uhucm_d8.py"), data, repr(util), str(mu), str(D)]
    return [sys.executable, "-I", os.path.join(HERE, "run_authors.py"), src, data, repr(util), str(mu), str(D)]


def child_cpu():
    r = resource.getrusage(resource.RUSAGE_CHILDREN); return r.ru_utime + r.ru_stime


def once(prog, mu):
    c0, t0 = child_cpu(), time.perf_counter()
    out = subprocess.run(cmd(prog, mu), capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    wall, cpu = time.perf_counter() - t0, child_cpu() - c0
    r = json.loads(out)
    return wall, cpu, r["seconds"], sorted(r["patterns"])


res = {}
for mu in (0.02, 0.05, 0.1):
    rec = {p: dict(times=[], cpu=[], inner=[]) for p in ("uhucpm", "authors")}
    sets = {p: [] for p in rec}
    order = ["uhucpm", "authors"]
    for it in range(-1, REPS):                                         # it = -1: untimed warm-up of both programs
        for p in (order if it % 2 == 0 or it < 0 else order[::-1]):
            w, c, inner, pats = once(p, mu)
            sets[p].append(pats)
            if it >= 0:
                rec[p]["times"].append(round(w, 3)); rec[p]["cpu"].append(round(c, 3)); rec[p]["inner"].append(inner)
    allsets = {json.dumps(s) for L in sets.values() for s in L}
    for p in rec: rec[p]["n"] = len(sets[p][0])
    rec["same_sets"] = len(allsets) == 1
    res[str(mu)] = rec
    print(which, mu, json.dumps({p: (v if not isinstance(v, dict) else {a: b for a, b in v.items()}) for p, v in rec.items()}), flush=True)
os.makedirs("out_timing_d8", exist_ok=True)
json.dump(dict(dataset=name, D=D, util=util, reps=REPS, timing=res), open(f"out_timing_d8/time_d8_{which}.json", "w"), indent=1)
print("DONE", flush=True)
