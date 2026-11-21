"""C++ count posterior (bin/post_counts) against uhucm/ceb.py on the same counts.
Deterministic parts (posterior means, variances, E[u+], Z) must agree to 1e-8; the bootstrap matrices use different
random streams, so their spread is compared: per-feature mean of the bootstrap variance of the posterior means, and
V_T of random groups of instances (ratio C++ / Python).
Usage: python verify_ceb.py COUNTS_CSV COUNT_COLUMN E T [B]"""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "uhucm"))
from ceb import ceb_posterior, cboot  # noqa: E402

csvp, col, E, t = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]); B = int(sys.argv[5]) if len(sys.argv) > 5 else 20
D = pd.read_csv(csvp); x = D[col].values.astype(float); F = D["Feature"].values
mu, s2, ep, st = ceb_posterior(x, E, F, t=t)
with tempfile.TemporaryDirectory() as tmp:
    post, epf = os.path.join(tmp, "p.bin"), os.path.join(tmp, "ep.txt")
    r = subprocess.run([os.path.join(HERE, "..", "bin", "post_counts"), csvp, col, repr(E), post, "--t", str(t), "--B", str(B), "--seed", "1",
                        "--ep", epf], capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    with open(post, "rb") as f:
        n, Bc = np.fromfile(f, np.int64, 2); Z = np.fromfile(f, float, 1)[0]
        cm = np.fromfile(f, float, n); cs = np.fromfile(f, float, n); CU = np.fromfile(f, float, Bc * n).reshape(Bc, n)
    cep = np.loadtxt(epf)
out = dict(n=int(n), max_abs_mu=float(np.abs(cm - mu).max()), max_rel_s2=float((np.abs(cs - s2) / s2).max()),
           max_abs_ep=float(np.abs(cep - ep).max()), Z_py=float(ep.sum()), Z_cpp=float(Z), cpp_seconds=info["seconds"])
PU = cboot(x, E, st, B=B, seed=1)
vp, vc = PU.var(0, ddof=1), CU.var(0, ddof=1)
feat_ratio = {f: float(vc[F == f].mean() / max(vp[F == f].mean(), 1e-300)) for f in np.unique(F)}
rng = np.random.default_rng(0); vt = []
for _ in range(400):
    f = rng.choice(np.unique(F)); idx = np.flatnonzero(F == f); g = rng.choice(idx, min(len(idx), int(rng.integers(5, 300))), replace=False)
    vt.append((CU[:, g].sum(1).var(ddof=1) + cs[g].sum()) / (PU[:, g].sum(1).var(ddof=1) + s2[g].sum()))
out.update(boot_var_ratio_by_feature=[round(min(feat_ratio.values()), 3), round(float(np.median(list(feat_ratio.values()))), 3), round(max(feat_ratio.values()), 3)],
           VT_ratio_random_groups=[round(float(np.quantile(vt, q)), 3) for q in (0.05, 0.5, 0.95)])
print(json.dumps(out))
