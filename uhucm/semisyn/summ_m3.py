"""Matrix summary: mean over seeds and K of FDP / detection; TD ratio to oracle."""
import json, glob, os, collections, numpy as np
D = collections.defaultdict(dict)
for fn in glob.glob('out_semisyn/log_*.txt'):
    tag = os.path.basename(fn)[4:-4].split('_'); city, truth, meth = tag[0], tag[1], '_'.join(tag[2:])
    for l in open(fn):
        a = l.split(' ', 4)
        if len(a) < 5 or a[0] != city: continue
        try: r = json.loads(a[4])
        except Exception: continue
        D[(city, truth, meth)][(a[2], a[3])] = r
rules = ['point', 'lb1.645', 'bfdr0.05', 'bfdr0.1']
print(f"{'city':9s}{'truth':10s}{'method':11s}" + "".join(f"{r:>22s}" for r in rules) + "   TDratio(bf05)")
for key in sorted(D):
    city, truth, meth = key; R = D[key]; O = D.get((city, truth, 'oracle'), {})
    row = []
    for r in rules:
        fdp = np.mean([v[r][1] for v in R.values()]); det = np.mean([v[r][2] for v in R.values()]); row.append(f"{fdp:6.3f}/{det:5.3f} ({len(R)})")
    rat = [R[k]['bfdr0.05'][2] / O[k]['bfdr0.05'][2] for k in R if k in O and O[k]['bfdr0.05'][2] > 0]
    print(f"{city:9s}{truth:10s}{meth:11s}" + "".join(f"{x:>22s}" for x in row) + (f"   {np.mean(rat):.2f}" if rat else ""))
