"""Repeated-timing statistics (10 runs per variant): mean and SD of the mining time, and per-dataset two-sided Welch
t-tests of UHUCM (exact bound) against NoPrune and against the mean bound, Holm-corrected over the thresholds of each
dataset. Inputs: out_timing/suite_FSQ_<city>_timing*.json (local) and results_kaggle/k5_timing.txt (CRC-checked
transcription of the T| lines of kexp5.py). Output: out_synth/stats_timing.json."""
import json, os, glob, zlib
import numpy as np
from scipy.stats import ttest_ind_from_stats
H = os.path.dirname(os.path.abspath(__file__))
LAB = {"Philadelphia": "D1", "New_Orleans": "D2", "Tampa": "D3", "Philadelphia_d70": "D1 (70 m)", "FSQ_NYC": "D4", "FSQ_TKY": "D5", "FSQ_TKY_d70": "D5 (70 m)"}


def load():
    T = {}
    for f in glob.glob(os.path.join(H, "out_timing", "suite_FSQ_*_timing*.json")):
        r = json.load(open(f)); nm = r["name"].replace("_timing", "")
        for d, res in r["timing"].items():
            key = nm + ("_d70" if float(d) == 70 else "")
            for K, rec in res.items():
                if K == "cliques": continue
                T.setdefault(LAB[key], {})[int(K)] = {v: dict(mean=float(np.mean(x["times"])), sd=float(np.std(x["times"], ddof=1)), n=len(x["times"]), evaluated=x["evaluated"])
                                                       for v, x in rec.items()} | {"cliques": res["cliques"]}
    p = os.path.join(H, "results_kaggle", "k5_timing.txt")
    if os.path.exists(p):
        for line in open(p):
            s, ck = line.strip().rsplit("|ck=", 1)
            assert format(zlib.crc32(s.encode()) & 0xffff, "04x") == ck, s[:40]
            t = s.split("|"); lab = LAB[t[1]]; K = int(t[3]); rec = {"cliques": int(t[4])}
            for item in t[5].split(";"):
                v, x = item.split("="); m, sd, ev = x.split(","); rec[v] = dict(mean=float(m), sd=float(sd), n=10, evaluated=int(ev))
            T.setdefault(lab, {})[K] = rec
    return T


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); prev = 0
    for i, j in enumerate(order):
        prev = max(prev, min(1.0, (m - i) * ps[j])); adj[j] = prev
    return adj


def main():
    T = load(); out = {}
    for lab, byK in sorted(T.items()):
        Ks = sorted(byK); res = {}
        for other in ("noprune", "mean"):
            ps, rows = [], []
            for K in Ks:
                e, o = byK[K]["exact"], byK[K][other]
                st = ttest_ind_from_stats(e["mean"], e["sd"], e["n"], o["mean"], o["sd"], o["n"], equal_var=False)
                ps.append(st.pvalue); rows.append(dict(K=K, ratio=round(e["mean"] / o["mean"], 3), p=float(f"{st.pvalue:.3g}")))
            for r, a in zip(rows, holm(np.array(ps))): r["p_holm"] = float(f"{a:.3g}")
            res[other] = rows
        res["table"] = {K: {v: (round(x["mean"], 3), round(x["sd"], 3), x["evaluated"]) for v, x in byK[K].items() if v != "cliques"} for K in Ks}
        out[lab] = res
    json.dump(out, open(os.path.join(H, "out_synth", "stats_timing.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main()
    for lab, r in o.items():
        print(lab, "vs NoPrune", [(x["K"], x["ratio"], x["p_holm"]) for x in r["noprune"]])
        print(lab, "vs mean   ", [(x["K"], x["ratio"], x["p_holm"]) for x in r["mean"]])
