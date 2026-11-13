"""Hold-rate table at K = 300 for all 13 datasets (Table VII of the paper) from out_synth/stats_rev3.json
-> out_synth/hold_all_K300.tex (kept, matched and rejected hold rates, paired difference with 95% bootstrap interval)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(H, "out_synth", "stats_rev3.json")))
names = {'D1': 'D1 Philadelphia', 'D2': 'D2 New Orleans', 'D3': 'D3 Tampa', 'D4': 'D4 New York', 'D5': 'D5 Tokyo',
         'Y1': 'Y1 Tucson', 'Y2': 'Y2 Indianapolis', 'Y3': 'Y3 Nashville', 'Y4': 'Y4 Reno', 'Y5': 'Y5 Saint Louis',
         'Y6': 'Y6 Edmonton', 'Y7': 'Y7 Santa Barbara', 'Y8': 'Y8 Boise'}
f = lambda x: f"{x:.2f}"; rows = []
for ds in d["datasets"]:
    v = d["per"].get(ds + "/K=300")
    if not v: continue
    p = v["paired"]
    rows.append(f"{names[ds]} & {f(v['kept']['rate'])} & {f(v['matched']['rate'])} & {f(v['rej']['rate'])} & "
                f"{p['diff']:+.2f} [{p['ci'][0]:.2f}, {p['ci'][1]:.2f}] & {p['n_pairs']} \\\\")
open(os.path.join(H, "out_synth", "hold_all_K300.tex"), "w").write("\n".join(rows) + "\n"); print("\n".join(rows))
