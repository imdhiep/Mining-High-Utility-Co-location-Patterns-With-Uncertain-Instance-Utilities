"""D8 (public PSO-MUCP data): degenerate-setting comparison and Strategy-4 audit.
For each configuration: the authors' PSO-MUCP (original), the authors' code with Strategy 4 disabled (reference,
since Strategies 1-3 are valid), and UHUCM with feature-level utilities (sigma = 0, z = 0) under the code's own
normalisation (sum of all instance utilities). Records wall time and result sets.
Usage: python compare_d8.py DATASET D MINUTIL "UTILDICT" [--authors]"""
import sys, os, csv, json, time, subprocess, ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, fastmine

import paths
SRC = paths.PSOMUCP_DIR
HERE = os.path.dirname(os.path.abspath(__file__))


def uhucm_run(path, util, D, mu):
    R = [r for r in csv.DictReader(open(path)) if r["Feature"] in util]
    names = sorted(util); fi = {f: j for j, f in enumerate(names)}
    feat = [fi[r["Feature"]] for r in R]; xy = [(float(r["LocX"]), float(r["LocY"])) for r in R]
    mu_i = [float(util[r["Feature"]]) for r in R]
    t = time.time(); Pb = fastmine.Problem(feat, xy, mu_i, [0.0] * len(R), D, 0.0, feat_names=names); tb = time.time() - t
    U_all = float(sum(mu_i))
    if U_all <= 0: return None
    t = time.time(); res, st, _ = fastmine.mine(Pb, mu * U_all / Pb.Z); tm = time.time() - t
    return dict(n=len(res), patterns=sorted("".join(p) for p in res), build_s=round(tb, 2), mine_s=round(tm, 2),
                seconds=round(tb + tm, 2), cliques=len(Pb.cliques), evaluated=st["evaluated"])


def authors(path, util, D, mu, nos4=False, timeout=1800):
    cmd = [sys.executable, "-I", os.path.join(HERE, "run_authors.py"), os.path.join(SRC, "HUCPM", "PSO-MUCP.py"), path, repr(util), str(mu), str(D)]
    if nos4: cmd.append("--no-s4")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip().splitlines()[-1]
        return json.loads(out)
    except subprocess.TimeoutExpired:
        return dict(timeout=timeout)


if __name__ == "__main__":
    ds, D, mu, util = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), ast.literal_eval(sys.argv[4])
    path = os.path.join(SRC, "DATA", ds)
    rec = dict(dataset=ds, D=D, minutil=mu, util=util, uhucm=uhucm_run(path, util, D, mu))
    if "--authors" in sys.argv:
        rec["orig"] = authors(path, util, D, mu); rec["nos4"] = authors(path, util, D, mu, True)
        u = set(rec["uhucm"]["patterns"]) if rec["uhucm"] else set()
        if "patterns" in rec["orig"]:
            rec["orig_missed"] = sorted(u - set(rec["orig"]["patterns"])); rec["orig_extra"] = sorted(set(rec["orig"]["patterns"]) - u)
        if "patterns" in rec["nos4"]:
            rec["nos4_equal_uhucm"] = set(rec["nos4"]["patterns"]) == u
    short = {k: (v if k not in ("uhucm", "orig", "nos4") else {a: b for a, b in v.items() if a != "patterns"} if isinstance(v, dict) else v) for k, v in rec.items()}
    print(json.dumps(short), flush=True)
    with open(os.path.join(HERE, "d8_results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
