"""Parse the transcribed Kaggle digest (results_kaggle/k4_digest.txt) into results_kaggle/rev_<city>.json, with consistency checks:
R1.645 = n_kept, R0 - R1.645 = n_rej = n_matched, and each filter's Jaccard index equals (|Rz|-n_only_z)/(|Rz|+n_only_f)."""
import json, os, sys, collections
H = os.path.dirname(os.path.abspath(__file__))
CITY = {"Phi": "Philadelphia", "New": "New_Orleans", "Tam": "Tampa"}
num = lambda x: None if x == "None" else (int(x) if x.lstrip("-").isdigit() else float(x))
# clique counts at d = 30, 50, 70 m from the earlier Kaggle run (D lines of summary_transcribed.txt)
CL = {}
for line in open(os.path.join(H, "results_kaggle", "summary_transcribed.txt")):
    t = line.split()
    if t and t[0] == "D": CL[(t[1], int(t[2]))] = int(t[3])
out = collections.defaultdict(lambda: collections.defaultdict(list)); bad = 0
for line in open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(H, "results_kaggle", "k4_digest.txt")):
    t = line.split()
    if not t or t[0].startswith("#"): continue
    c, v, K = t[0], t[1], int(t[2]); R = list(map(int, t[3:7]))
    q = {"K": K, "R0": R[0], "R1.645": R[1], "R2.45": R[2], "R4.36": R[3]}
    for g, s in zip(("kept", "rej", "matched"), t[7:10]):
        h, n = map(int, s.split("/")); q[f"hits_{g}"], q[f"n_{g}"] = h, n; q[f"hold_{g}"] = round(h / n, 4) if n else None
    q["perm_p"], q["FE_miss"] = num(t[10]), num(t[11])
    if v.startswith("d"): q["cliques"] = CL.get((c, int(v[1:])))
    F = {}
    for item in t[12:]:
        k, val = item.split(":"); J, hof, hoz, nf, nz = map(num, val.split("/"))
        F[{"minp": "minpi"}.get(k, k)] = dict(J=J, hold_only_f=hof, hold_only_z=hoz, n_only_f=nf, n_only_z=nz)
        Jc = (R[1] - nz) / (R[1] + nf)
        if abs(Jc - J) > 0.002: print("J mismatch", c, v, K, k, J, round(Jc, 4)); bad += 1
    q["filters"] = F
    if q["n_kept"] != R[1] or q["n_rej"] != R[0] - R[1] or q["n_matched"] != q["n_rej"]: print("count mismatch", c, v, K); bad += 1
    out[c][v].append(q)
for c, d in out.items():
    json.dump(d, open(os.path.join(H, "results_kaggle", f"rev_{CITY[c]}.json"), "w"), indent=1)
    print(c, {k: [q["K"] for q in x] for k, x in d.items()})
print("checks failed:", bad)
