"""Re-times UHUCM (current implementation) on the six all-positive D6 configurations of d8_results.jsonl, three runs
each, and checks that the answer sets are unchanged. Writes d6_retime.json."""
import json, os, statistics as st
from compare_d8 import uhucm_run, SRC
H = os.path.dirname(os.path.abspath(__file__))
L = [json.loads(l) for l in open(os.path.join(H, "d8_results.jsonl"))]
pos = [r for r in L if r.get("orig") and r["uhucm"] and all(float(v) > 0 for v in r["util"].values())]
out = []
for r in pos:
    path = os.path.join(SRC, "DATA", r["dataset"])
    runs = [uhucm_run(path, r["util"], r["D"], r["minutil"]) for _ in range(3)]
    same = all(set(x["patterns"]) == set(r["uhucm"]["patterns"]) for x in runs)
    rec = dict(dataset=r["dataset"], minutil=r["minutil"], n=runs[0]["n"], same=same, seconds=[x["seconds"] for x in runs],
               median=st.median(x["seconds"] for x in runs), authors=r["orig"].get("seconds"))
    print(json.dumps(rec), flush=True); out.append(rec)
json.dump(out, open(os.path.join(H, "d6_retime.json"), "w"), indent=1)
