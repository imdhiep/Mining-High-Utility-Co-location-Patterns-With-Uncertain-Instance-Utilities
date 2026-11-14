"""Fairness checks on the finished runs of fair/jobs.py. Prints every violated rule with the jobs to rerun and exits
with status 1 if any rule fails.  Usage: python fair/check.py

Rules
  R1  every job of the manifest has a successful stamp whose fingerprint matches the current code and settings (with the
      versions and input identities recorded in the stamps, so the check can run on another machine)
  R2  all stamps have the same Python/NumPy/SciPy versions and the same input identities for the same input file
  R3  semi-synthetic: within a comparison (same layout and truth), every method has the same valid (seed, K) draws, and
      there are at least 10 valid seeds for every K; all runs use TRUTHK=1, TRUTHC=momc and the same K list and seeds
  R4  timing and z-sweep: exclusive jobs, every variant has 5 timed runs with CPU times at every K, and the process was
      not starved: median CPU time / wall time of every variant at least 0.9. All variants compared at one K are timed
      interleaved inside one job, hence in one session. Jobs may come from several sessions (a Kaggle session stops
      after 12 hours) only if those sessions have the same number of CPUs and the same recorded CPU model
"""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run
from jobs import JOBS, SEMI_SEEDS

ROOT = run.ROOT
bad = []


def fail(rule, msg, jobs=()):
    bad.append((rule, msg, list(jobs))); print(f"FAIL {rule}: {msg}" + (f"  -> rerun {', '.join(jobs)}" if jobs else ""))


def semi_records(name, city):
    R = {}
    p = os.path.join(ROOT, name, "stdout.txt")
    if not os.path.exists(p): return R
    for l in open(p):
        a = l.split(" ", 4)
        if len(a) < 5 or a[0] != city: continue
        try: R[(int(a[2]), int(a[3]))] = json.loads(a[4])
        except Exception: pass
    return R


def first_valid(R, K, n=10):
    return sorted(s for (s, k), v in R.items() if k == K and v)[:n]


def main():
    stamps = {}
    for j in JOBS:
        st = run.state(j, recorded=True)
        if st != "done": fail("R1", f"{j['name']} is {st}", [j["name"]]); continue
        stamps[j["name"]] = json.load(open(run.stamp_path(j["name"])))
    hv = {json.dumps(s["versions"], sort_keys=True) for s in stamps.values()}
    if len(hv) > 1: fail("R2", f"{len(hv)} different version combinations: {sorted(hv)}")
    seen = {}
    for s in stamps.values():
        for i in s["inputs"] or []:
            if i: seen.setdefault(i.split("|")[0], set()).add(i)
    for f, v in seen.items():
        if len(v) > 1: fail("R2", f"input {f} differs between jobs: {sorted(v)}")
    # R3
    groups = {}
    for j in JOBS:
        if j["group"] and j["group"].startswith("semi_"): groups.setdefault(j["group"], []).append(j)
    for g, js in groups.items():
        city = js[0]["args"][0]; recs = {j["name"]: semi_records(j["name"], city) for j in js if j["name"] in stamps}
        for j in js:
            e = j["env"]
            if e.get("TRUTHK") != "1" or e.get("TRUTHC") != "momc" or j["args"][1] != SEMI_SEEDS:
                fail("R3", f"{j['name']} protocol settings differ", [j["name"]])
        if len(recs) < 2 and not g.endswith("_spf"): continue
        valid = {n: {k for k, v in R.items() if v} for n, R in recs.items()}
        if len({frozenset(v) for v in valid.values()}) > 1:
            fail("R3", f"{g}: methods have different valid draws", list(valid))
        for n, R in recs.items():
            for K in (100, 300, 1000):
                if len(first_valid(R, K)) < 10: fail("R3", f"{n}: fewer than 10 valid seeds at K={K}", [n])
    # R4
    T = [j for j in JOBS if j["exclusive"] and j["name"] in stamps]
    ses = {}
    for j in T:
        h = stamps[j["name"]]["host"]; ses.setdefault(h.get("session"), []).append(j["name"])
    if len(ses) > 1:                                     # allowed: every compared variant is timed inside one job
        print(f"note R4: timing jobs come from {len(ses)} sessions:", {s: len(v) for s, v in ses.items()})
        cpus = {stamps[j["name"]]["host"].get("cpus") for j in T}
        models = {stamps[j["name"]]["host"]["cpu_model"] for j in T if stamps[j["name"]]["host"].get("cpu_model")}
        if len(cpus) > 1 or len(models) > 1:
            fail("R4", f"timing sessions differ in hardware: cpus {sorted(cpus)}, cpu models {sorted(models)}", [j["name"] for j in T])
    for j in T:
        for f in glob.glob(os.path.join(ROOT, j["name"], "out_*", "*.json")):
            d = json.load(open(f)); recs = []
            def walk(x):
                if isinstance(x, dict):
                    if "times" in x: recs.append(x)
                    else:
                        for v in x.values(): walk(v)
            walk(d.get("timing", d))
            n = {(len(r["times"]), len(r.get("cpu", []))) for r in recs}
            if n != {(5, 5)}: fail("R4", f"{j['name']} {os.path.basename(f)}: run counts {sorted(n)}", [j["name"]])
            import numpy as np
            low = [float(np.median(np.array(r["cpu"]) / np.maximum(np.array(r["times"]), 1e-9))) for r in recs if r.get("cpu")]
            if low and min(low) < 0.9: fail("R4", f"{j['name']} {os.path.basename(f)}: CPU/wall ratio down to {min(low):.2f}", [j["name"]])
    print("all checks passed" if not bad else f"{len(bad)} failed checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
