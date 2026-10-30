"""Re-run only the Foursquare part of scale_mem.py (after the Def. 4 change) and merge into out_synth/scale_mem.json."""
import json, os, scale_mem, paths
H = os.path.dirname(os.path.abspath(__file__))
old = [r for r in json.load(open(os.path.join(H, "out_synth", "scale_mem.json"))) if r["kind"] != "fsq"]
for city in ("NYC", "TKY"):
    p = paths.fsq(city)
    for method in ("uhucm", "noprune"):
        reps = [scale_mem.run(["fsq", p, city, 0, method]) for _ in range(3)]
        r = dict(kind="fsq", n=city, k=None, method=method, reps=reps, mem=scale_mem.run(["fsq", p, city, 0, method, "mem"]))
        print(json.dumps(r), flush=True); old.append(r)
json.dump(old, open(os.path.join(H, "out_synth", "scale_mem.json"), "w"), indent=1)
