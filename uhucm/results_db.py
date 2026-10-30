"""Single source of truth for all reported numbers: parses the transcribed Kaggle summary (Yelp) and the local
JSON outputs (Foursquare, synthetic, D8). Tables and figures are generated only from here."""
import json, os, re, glob
H = os.path.dirname(os.path.abspath(__file__))
NAMES = {"Phi": "D1", "New": "D2", "Tam": "D3", "FSQ_NYC": "D4", "FSQ_TKY": "D5", "Las": "D6-LV", "Pho": "D6-PHX", "Tor": "D6-TOR"}
QK = ["K", "mu", "R0", "R1.645", "R2.45", "R4.36", "nested", "med_ev_rej", "med_ev_kept", "size_rej", "size_kept", "atoms_rej", "atoms_kept", "J_raw", "R_FE", "FE_miss"]
TK = ["K", "hold_kept", "hold_rej", "hold_matched", "regr_kept", "regr_rej", "regr_matched", "train_rej", "train_matched", "regr_diff", "perm_p", "test_cov_rej", "Lglobal_miss_z0.0", "Lsum_miss_z0.0", "Lglobal_miss_z1.645", "Lsum_miss_z1.645"]
SK = ["n", "features", "cliques", "omega", "Z", "insig_1645", "insig_245", "neg", "ev_median", "test_cov"]


def num(x):
    if x in ("True", "False"): return x == "True"
    if x == "None": return None
    try: return int(x)
    except ValueError:
        try: return float(x)
        except ValueError: return x


def parse_kaggle():
    db = {}
    for line in open(os.path.join(H, "results_kaggle", "summary_transcribed.txt")):
        if not line.strip() or line.startswith("#"): continue
        t = line.split(); tag, nm = t[0], t[1]
        d = db.setdefault(nm, {"quality": {}, "stats": None})
        if tag == "S": d["stats"] = dict(zip(SK, map(num, t[2:])))
        elif tag == "Q":
            q = dict(zip(QK, map(num, t[2:]))); d["quality"].setdefault(q["K"], {}).update(q)
        elif tag == "T":
            q = dict(zip(TK, map(num, t[2:2 + len(TK)]))); q["FE_bins"] = json.loads(t[2 + len(TK)])
            d["quality"].setdefault(q["K"], {}).update(q)
        elif tag == "I":
            v = list(map(num, t[2:])); d["interaction"] = dict(pairs=v[0], pos=v[1], var=v[2], R0=v[3], Rz=v[4], J=v[5], J_vs_beta0=v[6])
        elif tag == "E8":
            d["sensitivity"] = {k: dict(R=int(v.split("/")[0]), J=float(v.split("/")[1])) for k, v in (x.split("=") for x in t[2:])}
        elif tag == "E9":
            rows = {}
            for x in t[4:]:
                k, v = x.split(":"); R, e, tm = v.split("/"); rows[float(k)] = dict(R=int(R), evaluated=int(e), time=float(tm))
            d["dependence"] = dict(rho_hat=float(t[2].split("=")[1]), nested=t[3].endswith("True"), rows=rows)
        elif tag == "C": d["case"] = json.loads(line.split(" ", 2)[2])
        elif tag == "M":
            K = int(t[2]); d.setdefault("eff_mu", {})[K] = {k: dict(zip(("time", "evaluated", "R"), map(num, v.split("/")))) for k, v in (x.split("=") for x in t[3:])}
        elif tag == "A":
            d["ablation"] = {k: dict(zip(("time", "evaluated", "bounds"), map(num, v.split("/")))) for k, v in (x.split("=") for x in t[2:])}
        elif tag == "Z":
            d["eff_z"] = {float(k): dict(zip(("exact", "chord", "mean"), map(int, v.split("/")))) for k, v in (x.split("=") for x in t[2:])}
        elif tag == "D":
            e = t[6].split("/"); n_ = t[7].split("/")
            d.setdefault("eff_d", {})[int(t[2])] = dict(cliques=int(t[3]), omega=int(t[4]), build=float(t[5]),
                                                       exact=dict(time=float(e[0]), evaluated=int(e[1])), noprune=dict(time=float(n_[0]), evaluated=int(n_[1])))
    return db


def parse_local(path, nm):
    r = json.load(open(path)); d = {"stats": r["stats"], "quality": {q["K"]: q for q in r["quality"]}}
    i = r.get("interaction", {}); b = i.get("beta1.0", {})
    d["interaction"] = dict(pairs=i.get("pairs"), pos=i.get("pos"), var=i.get("var"), **b)
    d["sensitivity"] = r.get("sensitivity")
    dep = r.get("dependence")
    if dep: d["dependence"] = dict(rho_hat=dep.get("rho_hat"), nested=dep.get("nested"), rows={float(k): v for k, v in dep["rows"].items()})
    for q in r["quality"]:
        if "case" in q: d["case"] = q["case"]
    e = r.get("efficiency")
    if e:
        d["eff_mu"] = {int(K): v for K, v in e["vs_mu"].items()}; d["ablation"] = e["ablation"]
        d["eff_z"] = {float(z): {k: v[k]["evaluated"] for k in ("exact", "chord", "mean")} for z, v in e["vs_z"].items()}
        d["eff_d"] = {int(float(k)): v for k, v in e["vs_d"].items()}
    return d


def load():
    db = parse_kaggle()
    for nm in ("FSQ_NYC", "FSQ_TKY"):
        p = os.path.join(H, "out_suite", f"suite_{nm}.json")
        if os.path.exists(p): db[nm] = parse_local(p, nm)
    syn = {}
    for k in ("e1", "e2", "scale_n", "scale_k", "stress"):
        p = os.path.join(H, "out_synth", f"{k}.json")
        if os.path.exists(p): syn[k] = json.load(open(p))
    d8 = [json.loads(l) for l in open(os.path.join(H, "d8", "d8_results.jsonl"))] if os.path.exists(os.path.join(H, "d8", "d8_results.jsonl")) else []
    for k in ("scale_mem", "stats_tests"):
        p = os.path.join(H, "out_synth", f"{k}.json")
        if os.path.exists(p): syn[k] = json.load(open(p))
    if "scale_mem" not in syn and os.path.exists(os.path.join(H, "out_synth", "log_scale_mem.txt")):   # run still in progress: completed rows
        syn["scale_mem"] = [json.loads(l) for l in open(os.path.join(H, "out_synth", "log_scale_mem.txt")) if l.startswith("{")]
    syn["ext"] = {}
    for key, p in (("FSQ_NYC", "out_baseline/ext_FSQ_NYC.json"), ("FSQ_TKY", "out_baseline/ext_FSQ_TKY.json"),
                   ("Phi", "results_kaggle/ext_Philadelphia.json"), ("New", "results_kaggle/ext_New_Orleans.json"), ("Tam", "results_kaggle/ext_Tampa.json")):
        if os.path.exists(os.path.join(H, p)): syn["ext"][key] = json.load(open(os.path.join(H, p)))
        elif key.startswith("FSQ") and os.path.exists(os.path.join(H, "out_baseline", f"log_ext_{key[4:]}.txt")):   # run in progress: finished parts
            fl = {}
            for l in open(os.path.join(H, "out_baseline", f"log_ext_{key[4:]}.txt")):
                if l.startswith(f"X|{key}|FL|K="):
                    a = l.strip().split("|", 4); fl[a[3][2:]] = json.loads(a[4])
            if fl: syn["ext"][key] = dict(fl=fl, cq={}, mem={})
    syn["dsens"] = {}
    for key, pat in (("FSQ_NYC", "out_suite_d/suite_FSQ_NYC_d{}.json"), ("FSQ_TKY", "out_suite_d/suite_FSQ_TKY_d{}.json"),
                     ("Phi", "results_kaggle/suite_Philadelphia_d{}.json"), ("New", "results_kaggle/suite_New_Orleans_d{}.json"), ("Tam", "results_kaggle/suite_Tampa_d{}.json")):
        for dd in (30, 70):
            p = os.path.join(H, pat.format(dd))
            if os.path.exists(p): syn["dsens"][(key, dd)] = parse_local(p, key)
    return db, syn, d8


if __name__ == "__main__":
    db, syn, d8 = load()
    for k, v in db.items():
        print(k, NAMES.get(k), sorted(v.keys()), sorted(v["quality"].keys()))
    print("synthetic:", sorted(syn)); print("d8 records:", len(d8))
