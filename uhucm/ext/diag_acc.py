"""Diagnostics for the accuracy experiment (root causes of low detection and of the empty PSO-MUCP answers).

For each seed of the default setting on one layout it records, over the patterns whose point score reaches gamma/2:
  calibration   z_err = (true total - M) / sqrt(V) with V = V_T and with V = sum of sigma^2:
                mean, sd and coverage Pr(true total >= M - z* sqrt(V)); a well calibrated V gives sd ~ 1, coverage ~ 0.95
  normalizers   Z (expected positive part, used by the methods) against Zt (true positive total, used by the truth)
  centers       estimated moment center against the true center of each feature
  margin        detection of UHUCPM and of the point answer (CQ-HUCPM) by true score bins [1, 1.25), [1.25, 1.5), [1.5, inf) x gamma
  curves        FDP and detection of the lower-bound rule for z in a grid, and of the point rule with threshold m*gamma for m in a grid
  bfdr          posterior expected FDP rule: with p = Pr(true score >= gamma) = Phi((M - gamma Z) / sqrt(V_T)), report the
                largest set of patterns with the highest p whose mean of (1 - p) is at most q, for q in a grid
  error split   (true total - M) = instance part (true log-rate minus posterior mean, summed) + center part (estimated minus
                true center, summed); each part scaled by its own variance (sum of sigma^2, bootstrap part of V_T), and the
                center error of each feature divided by its bootstrap sd (Poisson redraws of the counts)
  oracle        the same lower-bound and bfdr curves with posteriors under the true law of the log-rates and the true
                center (upper limit of what any posterior rule can detect at a given FDP)
  min-count     CQ-HUCPM after dropping the venues with fewer than m = 2, 5, 10 training check-ins, threshold c * gamma
  pso           feature means of the posterior means (the utilities PSO-MUCP receives) and of the true utilities, and FDP and
                detection of the K patterns with the highest feature-level score (PSO-MUCP ranked by its own score)
Usage: python diag_acc.py LAYOUT SEEDS [K]     -> out_diag/diag_<LAYOUT>.json"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "acc.py"), encoding="utf-8").read()
src = src[:src.index("res = dict(layout=layout")]
layout, NS = sys.argv[1], int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 100
sys.argv = [os.path.join(HERE, "acc.py"), layout, "K", str(K), str(NS)]
g_ = {"__file__": os.path.join(HERE, "acc.py"), "__name__": "acc_lib"}
exec(compile(src, "acc.py", "exec"), g_)
np = g_["np"]; mine = g_["mine"]; pattern_atoms = g_["pattern_atoms"]; Problem = g_["Problem"]
OUT = "out_diag"; os.makedirs(OUT, exist_ok=True)
F0, XY0, W, loc, scl, order = g_["load_layout"](20)
base = Problem(F0, XY0, np.zeros(len(F0)), np.ones(len(F0)), 50.0, 0.0, feat_names=g_["ALPHA"][:len(order)])
QGRID = [0.01, 0.025, 0.05, 0.1, 0.2]
ZGRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.645, 2.0, 2.5]; MGRID = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
MINCNT = [int(x) for x in os.environ.get("MINCNT", "2,5,10").split(",")]; FGRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
res = dict(layout=layout, K=K, seeds={})
for seed in range(NS):
    rng = np.random.default_rng(1000 + seed); F, XY, n = F0, XY0, len(F0)
    lt = loc[F] + scl[F] * rng.normal(size=n)
    ctrue = np.array([g_["momc_true"](np.exp(lt[F == j])) for j in range(len(order))])
    ut = lt - ctrue[F]
    cnt = rng.poisson(np.exp(lt) * W).astype(float)
    mu, s2, Z, G = g_["posterior"](cnt, F, W); UB = g_["boot"](cnt, F, W, G, np.random.default_rng(seed))
    chat = np.array([g_["momc"](cnt[F == j], W) for j in range(len(order))])
    Zt = float(np.clip(ut, 0, None).sum())
    Pt = g_["with_values"](base, ut, np.zeros(n)); Rt, sct = g_["topk"](Pt, K)
    if len(sct) < K: continue
    g = sct[K - 1]; T = {p for p, v in Rt.items() if v >= g}
    P0 = g_["with_values"](base, mu, s2, Z); R0 = mine(P0, g / 2)[0]
    pats = list(R0)
    A = [pattern_atoms(P0, p) for p in pats]
    M = np.array([mu[a].sum() for a in A]); Vs = np.array([s2[a].sum() for a in A])
    VT = np.array([s2[a].sum() + UB[:, a].sum(1).var(ddof=1) for a in A]); TT = np.array([ut[a].sum() for a in A])
    tsc = np.array([Rt.get(p, TT[i] / Zt) for i, p in enumerate(pats)])
    cal = {}
    for nm, V in (("VT", VT), ("sum_s2", Vs)):
        zr = (TT - M) / np.sqrt(V)
        cal[nm] = dict(mean=round(float(zr.mean()), 3), sd=round(float(zr.std()), 3), coverage=round(float((TT >= M - 1.645 * np.sqrt(V)).mean()), 4))
    # split of the error TT - M into the instance part (log-rate minus posterior mean) and the center part
    pm = mu + chat[F]; cerr = chat - ctrue
    EI = np.array([(lt[a] - pm[a]).sum() for a in A]); EC = np.array([cerr[F[a]].sum() for a in A]); VB = VT - Vs
    zi = EI / np.sqrt(Vs); zc = EC / np.sqrt(np.maximum(VB, 1e-12))
    cal["instance"] = dict(mean=round(float(zi.mean()), 3), sd=round(float(zi.std()), 3))
    cal["center"] = dict(mean=round(float(zc.mean()), 3), sd=round(float(zc.std()), 3), share_var=round(float(EC.var() / max((TT - M).var(), 1e-12)), 3))
    bsd = []
    brng = np.random.default_rng(10_000 + seed)
    for j in range(len(order)):
        x = cnt[F == j]; bsd.append(float(np.std([g_["momc"](brng.poisson(x).astype(float), W) for _ in range(50)])))
    cal["center_z_by_feature"] = [round(float(e / max(b, 1e-9)), 2) for e, b in zip(cerr, bsd)]
    cal["n_by_feature"] = [int((F == j).sum()) for j in range(len(order))]
    inT = np.array([p in T for p in pats])

    def fd(sel):
        k = int(sel.sum()); tp = int((sel & inT).sum()); return [round(1 - tp / k, 4) if k else 0.0, round(tp / len(T), 4), k]
    u_sel = M - 1.645 * np.sqrt(VT) >= g * Z; p_sel = M >= g * Z
    bins = {}
    for lo, hi in ((1, 1.25), (1.25, 1.5), (1.5, 1e9)):
        tb = {p for p in T if lo * g <= Rt[p] < hi * g}
        bins[f"{lo}-{hi}"] = dict(nT=len(tb), uhucpm=round(sum(1 for p, s in zip(pats, u_sel) if s and p in tb) / max(1, len(tb)), 4),
                                  cq=round(sum(1 for p, s in zip(pats, p_sel) if s and p in tb) / max(1, len(tb)), 4))
    curves = dict(lb=[[z] + fd(M - z * np.sqrt(VT) >= g * Z) for z in ZGRID], lb_sum_s2=[[z] + fd(M - z * np.sqrt(Vs) >= g * Z) for z in ZGRID],
                  point=[[m] + fd(M >= m * g * Z) for m in MGRID],
                  lb_trueZ=[[z] + fd(M - z * np.sqrt(VT) >= g * Zt) for z in ZGRID], point_trueZ=[[m] + fd(M >= m * g * Zt) for m in MGRID])
    pp = np.array([0.5 * math.erfc(-(m - g * Z) / math.sqrt(2 * v)) for m, v in zip(M, VT)]); o = np.argsort(-pp)
    efdp = np.cumsum(1 - pp[o]) / np.arange(1, len(o) + 1); bf = []
    for q in QGRID:
        ok = np.nonzero(efdp <= q)[0]; sel = np.zeros(len(pats), bool); sel[o[:ok[-1] + 1] if len(ok) else []] = True
        bf.append([q] + fd(sel) + [round(float(efdp[ok[-1]]), 4) if len(ok) else 0.0])
    curves["bfdr"] = bf

    # oracle: the true law N(loc_f, scl_f^2) of the log-rates and the true center are known; the posterior of each venue
    # is computed on a fine grid, the pattern variance is the sum of the instance variances (no prior error to add), and
    # the normalizer is the expected positive part under these posteriors.  Same pattern rules as above.
    def fd_on(sel, pp_):
        k = int(sel.sum()); inT_ = np.array([p in T for p in pp_]); tp = int((sel & inT_).sum())
        return [round(1 - tp / k, 4) if k else 0.0, round(tp / len(T), 4), k]

    def bfdr_curve(M_, V_, Z_, pp_):
        pr = np.array([0.5 * math.erfc(-(m - g * Z_) / math.sqrt(2 * v)) for m, v in zip(M_, V_)]); oo = np.argsort(-pr)
        ef = np.cumsum(1 - pr[oo]) / np.arange(1, len(oo) + 1); out = []
        for q in QGRID:
            ok = np.nonzero(ef <= q)[0]; sel = np.zeros(len(pp_), bool); sel[oo[:ok[-1] + 1] if len(ok) else []] = True
            out.append([q] + fd_on(sel, pp_) + [round(float(ef[ok[-1]]), 4) if len(ok) else 0.0])
        return out

    mo = np.zeros(n); so = np.zeros(n); epo = 0.0
    for j in range(len(order)):
        ix = np.nonzero(F == j)[0]; th = np.linspace(loc[j] - 8 * scl[j], loc[j] + 8 * scl[j], 4001)
        lp = -0.5 * ((th - loc[j]) / scl[j]) ** 2
        xs, inv = np.unique(cnt[ix], return_inverse=True)
        ll = xs[:, None] * th[None, :] - np.exp(th)[None, :] * W + lp[None, :]
        w_ = np.exp(ll - ll.max(1, keepdims=True)); w_ /= w_.sum(1, keepdims=True)
        u_ = th - ctrue[j]; m_ = w_ @ u_; v_ = w_ @ (u_ ** 2) - m_ ** 2; e_ = w_ @ np.clip(u_, 0, None)
        mo[ix] = m_[inv]; so[ix] = np.maximum(v_[inv], 1e-12); epo += float(e_[inv].sum())
    Po = g_["with_values"](base, mo, so, epo); pats_o = list(mine(Po, g / 2)[0])
    Ao = [pattern_atoms(Po, p) for p in pats_o]
    Mo = np.array([mo[a].sum() for a in Ao]); Vo = np.array([so[a].sum() for a in Ao])
    curves["oracle_lb"] = [[z] + fd_on(Mo - z * np.sqrt(Vo) >= g * epo, pats_o) for z in ZGRID]
    curves["oracle_bfdr"] = bfdr_curve(Mo, Vo, epo, pats_o)

    # minimum-evidence filter: drop the venues with fewer than m training check-ins, then report the patterns whose
    # point score (posterior means of the kept venues over the full-data normalizer Z) reaches c * gamma
    for mmin in MINCNT:
        keep = np.nonzero(cnt >= mmin)[0]
        if len(keep) == 0 or len(set(F[keep])) < 2: curves[f"min{mmin}"] = []; continue
        bm = Problem(F[keep], XY[keep], np.zeros(len(keep)), np.ones(len(keep)), 50.0, 0.0, feat_names=g_["ALPHA"][:len(order)])
        Pm = g_["with_values"](bm, mu[keep], s2[keep], Z); pats_m = list(mine(Pm, FGRID[0] * g)[0])
        Mm = np.array([mu[keep][pattern_atoms(Pm, p)].sum() for p in pats_m])
        curves[f"min{mmin}"] = [[c_] + fd_on(Mm >= c_ * g * Z, pats_m) for c_ in FGRID]
    fmu = np.array([mu[F == j].mean() for j in range(len(order))])
    if (fmu > 0).any():
        Pf = g_["with_values"](base, fmu[F], np.zeros(n)); Rf, scf = g_["topk"](Pf, K)
        Sf = set(sorted(Rf, key=lambda p: -Rf[p])[:K])
    else: Sf = set()
    tpf = len(Sf & T)
    pso_top = dict(n=len(Sf), fdp=round(1 - tpf / len(Sf), 4) if Sf else 0.0, det=round(tpf / len(T), 4))
    fv = [round(float(mu[F == j].mean()), 4) for j in range(len(order))]; ftv = [round(float(ut[F == j].mean()), 4) for j in range(len(order))]
    res["seeds"][seed] = dict(nT=len(T), npool=len(pats), Z=Z, Zt=Zt, Z_over_Zt=round(Z / Zt, 4),
                              center_err=[round(float(a - b), 3) for a, b in zip(chat, ctrue)], calibration=cal, margin=bins, curves=curves, pso_topK=pso_top,
                              pso_feature_mean_mu=fv, pso_feature_mean_true=ftv, mean_s2=round(float(s2.mean()), 4), median_count=float(np.median(cnt)))
    print(seed, json.dumps(dict(Z_over_Zt=round(Z / Zt, 3), cal=cal, margin=bins, bfdr=bf, pso_topK=pso_top)), flush=True)
    json.dump(res, open(os.path.join(OUT, f"diag_{layout}.json"), "w"), indent=1)
    cmp_ = {k: {"curves": v.get("curves"), "calibration": {"VT": v["calibration"]["VT"]} if "calibration" in v else None}
            for k, v in res["seeds"].items() if isinstance(v, dict) and "curves" in v}
    json.dump(dict(layout=layout, seeds=cmp_), open(os.path.join(OUT, f"diag_{layout}_c.json"), "w"), separators=(",", ":"))  # compact copy: curves and V_T calibration only
print("DONE")
