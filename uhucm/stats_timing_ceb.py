"""Repeated-timing statistics for the binary-search implementation (10 runs per variant): mean and SD of the mining
time, and per configuration (dataset, d, k) two-sided Welch t-tests of UHUCM against NoPrune and against the mean
bound, Holm-corrected over the thresholds of the configuration. Inputs: out_timing2/suite_FSQ_<city>_k<k>[_d70].json
(local) and results_kaggle/k7_timing.txt (CRC-checked transcription of the T| lines of kexp7.py).
Output: out_synth/stats_timing_ceb.json."""
import json, os, glob, zlib
import numpy as np
from scipy.stats import ttest_ind_from_stats
H = os.path.dirname(os.path.abspath(__file__))


def load():
    T = {}
    for f in sorted(glob.glob(os.environ.get("TIMING_GLOB", os.path.join(H, os.environ.get("TIMING_DIR", "out_timing_ceb"), "suite_*.json")))):
        r = json.load(open(f)); nm = r["name"]                     # FSQ_NYC_k40 or FSQ_NYC_k20_d70
        city = "D4" if "_NYC_" in nm else "D5"; k = int(nm.split("_k")[1].split("_")[0])
        for d, res in r["timing"].items():
            key = f"{city}|{int(float(d))}|{k}"
            for K, rec in res.items():
                if K == "cliques": continue
                T.setdefault(key, {"cliques": res["cliques"]})[int(K)] = {v: dict(mean=float(np.mean(x["times"])), sd=float(np.std(x["times"], ddof=1)), times=list(x["times"]), cpu=list(x.get("cpu", [])), cpu_mean=float(np.mean(x["cpu"])) if x.get("cpu") else None,
                                                                              n=len(x["times"]), evaluated=x["evaluated"]) for v, x in rec.items()}
    p = os.path.join(H, "results_kaggle", "_none_")
    if os.path.exists(p):
        lab = {"Philadelphia": "D1", "New_Orleans": "D2", "Tampa": "D3", "Philadelphia_d70": "D1"}
        for line in open(p):
            line = line.strip()
            if not line.startswith("T|"): continue
            s, ck = line.rsplit("|ck=", 1)
            assert format(zlib.crc32(s.encode()) & 0xffff, "04x") == ck, s[:50]
            t = s.split("|"); key = f"{lab[t[1]]}|{t[2]}|20"; K = int(t[3])
            rec = {}
            for item in t[5].split(";"):
                v, x = item.split("="); m, sd, ev = x.split(","); rec[v] = dict(mean=float(m), sd=float(sd), n=10, evaluated=int(ev))
            T.setdefault(key, {"cliques": int(t[4])})[K] = rec
    return T


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); prev = 0
    for i, j in enumerate(order):
        prev = max(prev, min(1.0, (m - i) * ps[j])); adj[j] = prev
    return adj


def main():
    T = load(); out = {}
    for key, byK in sorted(T.items()):
        Ks = sorted(k for k in byK if k != "cliques"); res = {"cliques": byK["cliques"], "rows": {}}
        for other in ("noprune", "mean"):
            ps = []
            for K in Ks:
                e, o = byK[K]["exact"], byK[K][other]
                ps.append(ttest_ind_from_stats(e["mean"], e["sd"], e["n"], o["mean"], o["sd"], o["n"], equal_var=False).pvalue)
            for K, p, a in zip(Ks, ps, holm(np.array(ps))):
                res["rows"].setdefault(K, {})[f"vs_{other}"] = dict(ratio=round(byK[K]["exact"]["mean"] / byK[K][other]["mean"], 3), p=float(f"{p:.3g}"), p_holm=float(f"{a:.3g}"))
        for K in Ks:
            res["rows"][K]["table"] = {v: (round(x["mean"], 3), round(x["sd"], 3), x["evaluated"]) for v, x in byK[K].items()}
        out[key] = res
    json.dump(out, open(os.path.join(H, "out_synth", os.environ.get("TIMING_OUT", "stats_timing_ceb.json")), "w"), indent=1)
    return out


if __name__ == "__main__":
    for key, r in main().items():
        print(key, "C", r["cliques"], [(K, v["vs_noprune"]["ratio"], v["vs_noprune"]["p_holm"], v["vs_mean"]["ratio"], v["vs_mean"]["p_holm"]) for K, v in r["rows"].items()])
