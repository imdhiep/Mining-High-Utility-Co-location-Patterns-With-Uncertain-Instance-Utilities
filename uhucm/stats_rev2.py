"""Statistics for the second revision, computed only from result files:
Yelp (11 cities, z = 1.645): results_kaggle/k5_digest.txt (CRC-checked transcription of kexp5.py output);
Foursquare (D4, D5; calibrated z = 2.45): out_rev2/suite_FSQ_<city>_z2.45.json.
Outputs out_synth/stats_rev2.json with
  - hold rates with Wilson 95% intervals of kept, score-matched kept and rejected patterns;
  - Friedman test and Nemenyi critical difference over datasets at each K (Demsar 2006);
  - paired difference in hold rate, matched kept minus rejected, per dataset and pooled over datasets
    (bootstrap 95% CI from the 2x2 pair counts, 10,000 resamples, seed 0; exact McNemar test on the pooled discordant pairs);
  - filters at K = 300: set sizes, Wilson intervals, mean training scores, and the score-matched comparison
    (only-UHUCM minus only-filter) with the same pooled CI and McNemar test;
  - Holm-corrected permutation p-values of the matched regression difference."""
import json, math, os, zlib
import numpy as np
from scipy.stats import friedmanchisquare, norm, studentized_range, rankdata, binomtest
H = os.path.dirname(os.path.abspath(__file__))
NAMES = {"Philad": "D1", "New_Or": "D2", "Tampa": "D3", "Tucson": "Y1", "Indian": "Y2", "Nashvi": "Y3", "Reno": "Y4",
         "Saint_": "Y5", "Edmont": "Y6", "Santa_": "Y7", "Boise": "Y8"}


def wilson(h, n, a=0.05):
    if not n: return None
    z = norm.ppf(1 - a / 2); p = h / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - w, 3), round(c + w, 3)]


def num(x):
    return None if x == "na" else float(x)


def load():
    D = {}
    for line in open(os.path.join(H, "results_kaggle", "k5_digest.txt")):
        s, ck = line.strip().rsplit(" ", 1)
        assert format(zlib.crc32(s.encode()) & 0xffff, "04x") == ck, s[:40]
        t = s.split(); lab = NAMES[t[1]]; d = D.setdefault(lab, {"z": 1.645, "q": {}})
        if t[0] == "S":
            d["stats"] = dict(n=int(t[2]), cliques=int(t[3]), omega=int(t[4]), ev_median=float(t[5]), insig=float(t[6]), neg=float(t[7]))
            continue
        K = int(t[2]); hk, nk = map(int, t[6].split("/")); hr, nr = map(int, t[7].split("/")); hm, nm = map(int, t[8].split("/"))
        b, om, orj, ne = map(int, t[9].split("/"))
        q = dict(R0=int(t[3]), Rz=int(t[4]), R436=int(t[5]), hits_kept=hk, n_kept=nk, hits_rej=hr, n_rej=nr, hits_matched=hm, n_matched=nm,
                 pair=dict(both=b, only_matched=om, only_rej=orj, neither=ne), perm_p=float(t[10]), FE_miss=float(t[11]), filters={})
        for item in t[12:]:
            k, v = item.split(":"); v = v.split("/")
            q["filters"][{"mi": "minpi", "pi": "pi", "ev": "ev"}[k]] = dict(
                J=num(v[0]), n_only_f=int(v[1]), hits_only_f=int(v[2]), n_only_z=int(v[3]), hits_only_z=int(v[4]),
                train_only_f=num(v[5]), train_only_z=num(v[6]),
                pair=None if v[7] == "na" else dict(both=int(v[7]), only_z=int(v[8]), only_f=int(v[9]), neither=int(v[10])))
        d["q"][K] = q
    for lab, c in (("D4", "NYC"), ("D5", "TKY")):
        r = json.load(open(os.path.join(H, "out_rev2", f"suite_FSQ_{c}_z2.45.json")))
        d = D[lab] = {"z": 2.45, "q": {}, "stats": r["stats"]}
        for x in r["quality"]:
            p = x.get("pair2x2", {})
            q = dict(R0=x["R0"], Rz=x["R2.45"], R436=x["R4.36"], hits_kept=x["hits_kept"], n_kept=x["n_kept"], hits_rej=x["hits_rej"],
                     n_rej=x["n_rej"], hits_matched=x.get("hits_matched"), n_matched=x.get("n_matched"),
                     pair=dict(both=p.get("both"), only_matched=p.get("only_matched"), only_rej=p.get("only_rej"), neither=p.get("neither")),
                     perm_p=x.get("perm_p"), FE_miss=x["FE_miss"], filters={},
                     size_kept=x.get("size_kept"), size_rej=x.get("size_rej"), atoms_kept=x.get("atoms_kept"), atoms_rej=x.get("atoms_rej"),
                     med_ev_kept=x.get("med_ev_kept"), med_ev_rej=x.get("med_ev_rej"))
            if x["K"] == 300:
                for k, v in x["filters"].items():
                    b = v.get("pair2x2")
                    q["filters"][k] = dict(J=v["J"], n_only_f=v["n_only_f"], hits_only_f=v["hits_only_f"], n_only_z=v["n_only_z"],
                                           hits_only_z=v["hits_only_z"], train_only_f=v["train_only_f"], train_only_z=v["train_only_z"],
                                           pair=None if not b else dict(both=b["both"], only_z=b["only_z"], only_f=b["only_f"], neither=b["neither"]))
            d["q"][x["K"]] = q
    return D


def boot_diff(tables, plus, minus, B=10000, seed=0):
    """tables: list of 2x2 dicts (one per stratum); difference = sum(plus - minus) / sum(n), CI by multinomial resampling within strata."""
    rng = np.random.default_rng(seed); keys = ["both", plus, minus, "neither"]
    tot = sum(sum(t[k] for k in keys) for t in tables)
    est = sum(t[plus] - t[minus] for t in tables) / tot
    sims = np.zeros(B)
    for t in tables:
        n = sum(t[k] for k in keys)
        if n == 0: continue
        p = np.array([t[k] for k in keys], float) / n
        draw = rng.multinomial(n, p, size=B); sims += draw[:, 1] - draw[:, 2]
    sims /= tot
    disc_p, disc_m = sum(t[plus] for t in tables), sum(t[minus] for t in tables)
    p_mc = binomtest(disc_p, disc_p + disc_m, 0.5).pvalue if disc_p + disc_m else 1.0
    return dict(diff=round(est, 3), ci=[round(float(np.quantile(sims, 0.025)), 3), round(float(np.quantile(sims, 0.975)), 3)],
                n_pairs=tot, discordant=[disc_p, disc_m], mcnemar_p=float(f"{p_mc:.3g}"))


def cluster_boot(tables, plus, minus, B=10000, seed=0):
    """Pooled difference with a bootstrap over datasets (strata resampled with replacement), robust to dependence among
    the pairs of one dataset; also the number of datasets with a positive, zero and negative difference."""
    rng = np.random.default_rng(seed); keys = ["both", plus, minus, "neither"]
    T = np.array([[t[k] for k in keys] for t in tables], float); n = T.sum(1); d = T[:, 1] - T[:, 2]
    idx = rng.integers(0, len(T), size=(B, len(T)))
    sims = d[idx].sum(1) / n[idx].sum(1)
    per = d / np.maximum(n, 1)
    return dict(ci_cluster=[round(float(np.quantile(sims, 0.025)), 3), round(float(np.quantile(sims, 0.975)), 3)],
                pos=int((per > 0).sum()), zero=int((per == 0).sum()), neg=int((per < 0).sum()))


def nemenyi_cd(k, n, a=0.05):
    return studentized_range.ppf(1 - a, k, 1e6) / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n))


def main():
    D = load(); out = {"datasets": sorted(D), "per": {}, "pooled": {}, "friedman": {}, "filters": {}, "holm": []}
    for lab in sorted(D):
        for K, q in D[lab]["q"].items():
            r = {g: dict(rate=round(q[f"hits_{g}"] / q[f"n_{g}"], 3), n=q[f"n_{g}"], ci=wilson(q[f"hits_{g}"], q[f"n_{g}"]))
                 for g in ("kept", "matched", "rej") if q.get(f"n_{g}")}
            r["paired"] = boot_diff([q["pair"]], "only_matched", "only_rej")
            r["reject_frac"] = round(1 - q["Rz"] / q["R0"], 3); r["FE_miss"] = q["FE_miss"]; r["perm_p"] = q["perm_p"]
            out["per"][f"{lab}/K={K}"] = r
    for K in (100, 300, 1000):
        labs = [l for l in sorted(D) if K in D[l]["q"]]
        tb = [D[l]["q"][K]["pair"] for l in labs]
        out["pooled"][K] = boot_diff(tb, "only_matched", "only_rej") | cluster_boot(tb, "only_matched", "only_rej") | {"datasets": len(labs)}
        A = np.array([[D[l]["q"][K][f"hits_{g}"] / D[l]["q"][K][f"n_{g}"] for g in ("kept", "matched", "rej")] for l in labs])
        st, p = friedmanchisquare(A[:, 0], A[:, 1], A[:, 2])
        rk = np.array([rankdata(-r) for r in A]); mr = rk.mean(0); cd = nemenyi_cd(3, len(labs))
        out["friedman"][K] = dict(n=len(labs), chi2=round(float(st), 3), p=float(f"{p:.3g}"), mean_rank=dict(zip(("kept", "matched", "rej"), map(lambda x: round(float(x), 2), mr))),
                                  CD=round(cd, 3), kept_vs_rej=bool(mr[2] - mr[0] > cd), matched_vs_rej=bool(mr[2] - mr[1] > cd), kept_vs_matched=bool(mr[1] - mr[0] > cd),
                                  matched_gt_rej=int(sum(A[:, 1] > A[:, 2])), matched_eq_rej=int(sum(A[:, 1] == A[:, 2])), kept_gt_rej=int(sum(A[:, 0] > A[:, 2])))
    # filters at K = 300
    for f in ("minpi", "pi", "ev"):
        rows, tabs = {}, []
        for lab in sorted(D):
            v = D[lab]["q"][300]["filters"][f]
            rows[lab] = dict(J=v["J"], n_only_f=v["n_only_f"], hold_only_f=round(v["hits_only_f"] / v["n_only_f"], 3) if v["n_only_f"] else None,
                             ci_only_f=wilson(v["hits_only_f"], v["n_only_f"]), n_only_z=v["n_only_z"],
                             hold_only_z=round(v["hits_only_z"] / v["n_only_z"], 3) if v["n_only_z"] else None, ci_only_z=wilson(v["hits_only_z"], v["n_only_z"]),
                             train_only_f=v["train_only_f"], train_only_z=v["train_only_z"])
            if v["pair"]: tabs.append(v["pair"]); rows[lab]["paired"] = boot_diff([v["pair"]], "only_z", "only_f")
        out["filters"][f] = dict(rows=rows, pooled=boot_diff(tabs, "only_z", "only_f") | cluster_boot(tabs, "only_z", "only_f") | {"datasets": len(tabs)},
                                 unmatched_wins=sum(1 for r in rows.values() if r["hold_only_f"] is not None and r["hold_only_z"] is not None and r["hold_only_z"] > r["hold_only_f"]),
                                 unmatched_n=sum(1 for r in rows.values() if r["hold_only_f"] is not None and r["hold_only_z"] is not None),
                                 train_higher_z=sum(1 for r in rows.values() if r["train_only_f"] is not None and r["train_only_z"] is not None and r["train_only_z"] > r["train_only_f"]))
    # Holm over all matched-regression permutation tests
    P = sorted(((v["perm_p"], k) for k, v in out["per"].items() if v["perm_p"] is not None))
    m = len(P); prev = 0
    for i, (p, k) in enumerate(P):
        adj = min(1.0, max(prev, (m - i) * p)); prev = adj
        out["holm"].append(dict(test=k, p=p, p_holm=round(adj, 4)))
    json.dump(out, open(os.path.join(H, "out_synth", "stats_rev2.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps({k: o[k] for k in ("pooled", "friedman")}, indent=1))
    for f, v in o["filters"].items(): print(f, v["pooled"], v["unmatched_wins"], "/", v["unmatched_n"], "train_higher_z", v["train_higher_z"])
    print([h for h in o["holm"] if h["p_holm"] < 0.05])
