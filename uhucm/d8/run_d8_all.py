"""All D8 configurations (degenerate positive setting + Strategy-4 audit with negative features). Seeded."""
import subprocess, sys, json, random, statistics
CAL = {'A': 2, 'B': 4, 'C': 8, 'D': 4, 'E': 1, 'F': 4, 'G': 5, 'H': 3, 'I': 9, 'J': 3, 'K': 5, 'L': 10, 'M': 8}  # authors' default
rnd = random.Random(2026)
LV = {f: rnd.randint(1, 10) for f in "ABCDEFGHIJ"}
which = sys.argv[1]
ds = {"cal": ("California_POI 13f.csv", 1300, CAL), "lv": ("LasVegas_POI 10f.csv", 100, LV)}[which]
name, D, base = ds
confs = [("pos", dict(base), mu) for mu in (0.02, 0.05, 0.1)]
r2 = random.Random(7 if which == "cal" else 8)
feats = sorted(base)
for kneg in (1, 2, 3, 4):
    for mag in (0.5, 1.0, 2.0):
        negs = r2.sample(feats, kneg)
        med = statistics.median(base.values())
        u = {f: (-round(mag * med, 2) if f in negs else v) for f, v in base.items()}
        for mu in (0.05, 0.1):
            confs.append((f"neg{kneg}x{mag}", u, mu))
import os
done = set()
if os.path.exists("d8_results.jsonl"):
    for l in open("d8_results.jsonl"):
        r = json.loads(l)
        if "orig" in r: done.add((r["dataset"], r["minutil"], json.dumps(r["util"], sort_keys=True)))
for tag, u, mu in confs:
    if (name, mu, json.dumps(u, sort_keys=True)) in done: continue
    out = subprocess.run([sys.executable, "compare_d8.py", name, str(D), str(mu), repr(u), "--authors"], capture_output=True, text=True).stdout
    print(tag, out.strip()[:600], flush=True)
