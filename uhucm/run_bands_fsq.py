"""Distance-banded dependence on Foursquare (D4/D5) at the calibrated z = 2.45: correlation of posterior residuals
(posterior means minus the overall mean) between instances in distance bands up to 1.6 km (negative estimates set to 0,
zero beyond 1.6 km), mining with this covariance (bounds under independence remain valid), and comparison with
independence and with the equicorrelated bound at the neighbour correlation. Writes out_rev2/bands_<city>.json."""
import sys, json, numpy as np
import fsq_loader, paths
from suite import Suite, jac
from fastmine import mine
from scipy.spatial import cKDTree
EDGES = [0, 50, 100, 200, 400, 800, 1600]
city = sys.argv[1]; z = 2.45
rows, users, NU, info = fsq_loader.load(paths.fsq(city), city)
xy = np.array([(r["x"], r["y"]) for r in rows]); r = np.array([r["mu"] for r in rows]); r = r - r.mean()
T = cKDTree(xy); rho = []
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    P = T.query_pairs(hi, output_type="ndarray"); dd = np.linalg.norm(xy[P[:, 0]] - xy[P[:, 1]], axis=1); P = P[dd > lo]
    a, b = r[P[:, 0]], r[P[:, 1]]; c = float(np.corrcoef(np.r_[a, b], np.r_[b, a])[0, 1]); rho.append(round(c, 4))
rb = [max(0.0, x) for x in rho]
S = Suite(f"FSQ_{city}", rows, 50.0, "out_rev2", zmain=z, log=lambda s: None)
P0 = S.mk(); R0all, sc = S.topk(P0, 1000)
Pi = S.mk(z=z); Pb = S.mk(z=z); Pb.set_bands(EDGES, rb); Pe = S.mk(z=z, rho=rho[0])
out = dict(edges=EDGES, rho=rho, rho_used=rb, K={})
has_t = ~np.isnan(S.mut); Zt = float(np.clip(S.mut[has_t], 0, None).sum())
from fastmine import pattern_atoms
for K in (100, 300, 1000):
    m = sc[K - 1]; R0 = {p for p, v in R0all.items() if v >= m}
    Ri, sti, _ = mine(Pi, m); Rb, stb, _ = mine(Pb, m); Re, ste, _ = mine(Pe, m)
    Ri, Rb, Re = set(Ri), set(Rb), set(Re)
    te = lambda p: float(np.nansum(S.mut[pattern_atoms(P0, p)]) / Zt)
    hold = lambda Ps: round(float(np.mean([te(p) >= m for p in Ps])), 4) if Ps else None
    out["K"][K] = dict(R0=len(R0), R_indep=len(Ri), R_banded=len(Rb), R_equi=len(Re), banded_subset=Rb <= Ri, J_banded_indep=jac(Rb, Ri),
                       nodes_indep=sti["evaluated"], nodes_banded=stb["evaluated"],
                       hold_banded=hold(Rb), hold_removed_by_band=hold(Ri - Rb), hold_rej_indep=hold(R0 - Ri))
    print(city, K, out["K"][K], flush=True)
json.dump(out, open(f"out_rev2/bands_{city}.json", "w"), indent=1)
print(city, "rho", rho, flush=True)
