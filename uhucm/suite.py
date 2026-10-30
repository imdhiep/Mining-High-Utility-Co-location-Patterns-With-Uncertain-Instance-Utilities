"""Experiment suite shared by all real datasets (Yelp on Kaggle, Foursquare locally).

Every number printed comes from the run; nothing is filled in by hand. Lines starting with 'R|' are the compact
summary used to transcribe results; the full record is written to <out>/suite_<name>.json.

Experiments (numbering of Section VI):
  stats  dataset / evidence statistics
  E0     answer-set change between z=0 and z>0 at thresholds giving K patterns for z=0
  E3     FE (mu - z sigma fed to an additive engine) miss rate, by |A(P)|
  E7     temporal hold-out: hold rate, regression, score-matched comparison, sign-flip permutation test
  E4     audit of Rule L (two variants) against UHUCM
  E8     sensitivity of the answer set to posterior hyperparameters (callback supplied per dataset)
  E9     dependence: estimated rho and mining with rho in {0, rho_hat, 0.3, 0.6}
  EFF    efficiency vs threshold, vs z, vs distance; ablation of bounds and pruning strategies
  CASE   patterns accepted by z=0 and rejected by z=1.645
"""
import json, time, math, statistics as st, os
import numpy as np
from fastmine import Problem, mine, pattern_atoms, Timeout

ZS = (1.645, 2.45, 4.36)
ZMAIN = 1.645


def jac(a, b):
    a, b = set(a), set(b)
    return round(len(a & b) / max(1, len(a | b)), 4)


def caliper_pairs(A, B, score, cal):
    """Greedy nearest-score matching without replacement: each element of A (in increasing score) is paired with the
    unused element of B of nearest score if the absolute difference is at most cal; unmatched elements are dropped."""
    Bl = sorted(B, key=score); bv = np.array([score(x) for x in Bl]); used = np.zeros(len(Bl), bool); out = []
    for a in sorted(A, key=score):
        if not len(Bl): break
        dist = np.abs(bv - score(a)); dist[used] = np.inf; j = int(np.argmin(dist))
        if dist[j] <= cal: used[j] = True; out.append((a, Bl[j]))
    return out


def table2x2(pairs, hit):
    """pairs (a, b): counts of (hit_a, hit_b) as both / only_b / only_a / neither."""
    return dict(both=int(sum(bool(hit(a) and hit(b)) for a, b in pairs)), only_b=int(sum(bool((not hit(a)) and hit(b)) for a, b in pairs)),
                only_a=int(sum(bool(hit(a) and not hit(b)) for a, b in pairs)), neither=int(sum(bool((not hit(a)) and (not hit(b))) for a, b in pairs)), n=len(pairs))


def timed(fn, reps=1):
    ts, out = [], None
    for _ in range(reps):
        t = time.time(); out = fn(); ts.append(time.time() - t)
    return out, float(np.median(ts))


class Suite:
    def __init__(self, name, rows, d, out, recompute=None, variants=(), users=None, NU=None, interactions=None,
                 resid=None, log=print, deadline=900, reps=3, zmain=ZMAIN):
        self.name, self.rows, self.d, self.out = name, rows, d, out
        self.recompute, self.variants = recompute, variants
        self.users, self.NU, self.interactions, self.resid = users, NU, interactions, resid
        self.log, self.deadline, self.reps = log, deadline, reps
        self.zmain = zmain
        self.names = sorted({r["feat"] for r in rows}); fi = {f: j for j, f in enumerate(self.names)}
        self.F = [fi[r["feat"]] for r in rows]; self.XY = [(r["x"], r["y"]) for r in rows]
        self.mu = np.array([r["mu"] for r in rows], float); self.s2 = np.array([r["s2"] for r in rows], float)
        self.raw = np.array([r.get("raw", r["mu"]) for r in rows], float)
        self.mut = np.array([r.get("mu_test", float("nan")) for r in rows], float)
        self.ev = np.array([r["n_r"] for r in rows], float)
        self.rec = {"name": name, "d": d}
        os.makedirs(out, exist_ok=True)

    def P(self, *a):
        self.log(" ".join(str(x) for x in a))

    def R(self, tag, **kw):
        self.P("R|" + self.name + "|" + tag + "|" + " ".join(f"{k}={v}" for k, v in kw.items()))

    def mk(self, mu=None, s2=None, z=0.0, rho=0.0, d=None, pairs=None, beta=0.0):
        mu = self.mu if mu is None else mu; s2 = self.s2 if s2 is None else s2
        return Problem(self.F, self.XY, mu, s2, d or self.d, z, rho, pairs, beta, self.names)

    # ------------------------------------------------------------------ thresholds
    @staticmethod
    def topk(Pb, Kmax, start=0.05):
        m = start
        while True:
            R = mine(Pb, m)[0]
            if len(R) >= Kmax or m < 1e-9: return R, sorted(R.values(), reverse=True)
            m /= 2

    # ------------------------------------------------------------------ stats + E0/E3/E7/E4/CASE
    def quality(self, Ks=(100, 300, 1000), zmain=None):
        """zmain: z that splits kept/rejected (1.645 normal choice; 2.45 sub-Gaussian choice for the activity posterior)."""
        zmain = self.zmain if zmain is None else zmain
        t = time.time(); P0 = self.mk(); tb = time.time() - t
        act = self.mu != 0
        stats = dict(n=len(self.rows), features=len(self.names), cliques=len(P0.cliques),
                     omega=max(map(len, P0.cliques)), Z=round(P0.Z, 2), build_s=round(tb, 1),
                     insig_1645=round(float((np.sqrt(self.s2[act]) > np.abs(self.mu[act]) / 1.645).mean()), 3),
                     insig_245=round(float((np.sqrt(self.s2[act]) > np.abs(self.mu[act]) / 2.45).mean()), 3),
                     neg=round(float((self.mu[act] < 0).mean()), 3), ev_median=float(np.median(self.ev)),
                     test_cov=round(float((~np.isnan(self.mut)).mean()), 3))
        self.R("stats", **stats); self.rec["stats"] = stats
        R0all, sc = self.topk(P0, max(Ks))
        self.P0, self.R0all, self.sc = P0, R0all, sc
        ZQ = tuple(sorted(set(ZS) | {zmain}))
        Pz = {z: self.mk(z=z) for z in ZQ}
        Praw = self.mk(mu=self.raw, s2=np.zeros_like(self.raw))
        fe = self.mu - zmain * np.sqrt(self.s2); Pfe = self.mk(mu=fe, s2=np.zeros_like(fe))
        has_t = ~np.isnan(self.mut); Zt = float(np.clip(self.mut[has_t], 0, None).sum())
        rows = []
        for K in Ks:
            if K > len(sc): continue
            muK = sc[K - 1]; tau = muK * P0.Z
            R0 = {p for p, v in R0all.items() if v >= muK}
            Rz = {z: set(mine(Pz[z], muK)[0]) for z in ZQ}
            kept, rej = R0 & Rz[zmain], R0 - Rz[zmain]
            A = {p: pattern_atoms(P0, p) for p in R0}
            med = lambda Ps: float(st.median(float(np.median(self.ev[A[p][A[p] < P0.n]])) for p in Ps)) if Ps else None
            row = dict(K=K, mu=round(muK, 6), R0=len(R0), zmain=zmain, **{f"R{z}": len(Rz[z]) for z in ZQ},
                       **{f"J{z}": jac(R0, Rz[z]) for z in ZQ}, nested=all(Rz[z] <= R0 for z in ZQ),
                       med_ev_rej=med(rej), med_ev_kept=med(kept),
                       size_rej=round(st.mean(map(len, rej)), 2) if rej else None,
                       size_kept=round(st.mean(map(len, kept)), 2) if kept else None,
                       atoms_rej=float(st.median(len(A[p]) for p in rej)) if rej else None,
                       atoms_kept=float(st.median(len(A[p]) for p in kept)) if kept else None)
            Rr = set(mine(Praw, tau / Praw.Z)[0]); row["J_raw"] = jac(R0, Rr); row["R_raw"] = len(Rr)
            # E3
            Rf = set(mine(Pfe, tau / Pfe.Z)[0]); Ru = Rz[zmain]
            Au = {p: (A[p] if p in A else pattern_atoms(P0, p)) for p in Ru}
            bins = [(1, 10), (11, 50), (51, 200), (201, 10 ** 9)]
            row.update(R_FE=len(Rf), FE_miss=round(len(Ru - Rf) / max(1, len(Ru)), 4),
                       FE_bins=[[len([p for p in Ru - Rf if a <= len(Au[p]) <= b]), len([p for p in Ru if a <= len(Au[p]) <= b])] for a, b in bins])
            # E7
            if Zt > 0 and has_t.any():
                tr_score = lambda p: R0all[p]
                te_score = lambda p: float(np.nansum(self.mut[A[p]][has_t[A[p]]]) / Zt)
                hold = lambda Ps: round(float(np.mean([te_score(p) >= muK for p in Ps])), 4) if Ps else None
                regr = lambda Ps: round(float(np.mean([tr_score(p) - te_score(p) for p in Ps])), 6) if Ps else None
                row.update(hold_kept=hold(kept), hold_rej=hold(rej), regr_kept=regr(kept), regr_rej=regr(rej))
                if rej and kept:
                    kl = sorted(kept, key=tr_score); kv = np.array([tr_score(p) for p in kl]); used = set(); pairs = []
                    for p in sorted(rej, key=tr_score):
                        order = np.argsort(np.abs(kv - tr_score(p)))
                        j = next((int(i) for i in order if int(i) not in used), int(order[0])); used.add(j)
                        pairs.append((p, kl[j]))
                    mk_ = [q for _, q in pairs]
                    diff = np.array([(tr_score(p) - te_score(p)) - (tr_score(q) - te_score(q)) for p, q in pairs])
                    rng = np.random.default_rng(0); obs = diff.mean()
                    flips = rng.choice([-1.0, 1.0], size=(10000, len(diff)))
                    pval = float((np.abs((flips * diff).mean(1)) >= abs(obs) - 1e-15).mean())
                    hr_, hm_ = [te_score(p) >= muK for p, _ in pairs], [te_score(q) >= muK for _, q in pairs]
                    row["pair2x2"] = dict(both=int(sum(x and y for x, y in zip(hr_, hm_))), only_matched=int(sum((not x) and y for x, y in zip(hr_, hm_))),
                                          only_rej=int(sum(x and not y for x, y in zip(hr_, hm_))), neither=int(sum((not x) and (not y) for x, y in zip(hr_, hm_))),
                                          distinct_kept=len(set(mk_)), max_gap=round(float(max(abs(tr_score(p) - tr_score(q)) for p, q in pairs)), 6),
                                          mean_gap=round(float(np.mean([tr_score(q) - tr_score(p) for p, q in pairs])), 6))
                    cp = caliper_pairs(rej, kept, tr_score, 0.05 * muK)
                    row["pair_cal"] = table2x2(cp, lambda p: te_score(p) >= muK)   # a = rejected, b = matched kept
                    row.update(hold_matched=hold(mk_), regr_matched=regr(mk_),
                               train_rej=round(float(np.mean([tr_score(p) for p in rej])), 6),
                               train_matched=round(float(np.mean([tr_score(q) for q in mk_])), 6),
                               regr_diff=round(float(obs), 6), perm_p=pval, test_cov_rej=round(float(np.mean([has_t[A[p]].mean() for p in rej])), 3))
            # FILTER baselines: point utility + (min participating count | participation index | median evidence),
            # each tuned on R0 to the output size closest to |R_z|; Jaccard with R_z and hold rates
            if Zt > 0 and has_t.any():
                Fa = np.array(self.F); nf = np.bincount(Fa, minlength=len(self.names)); n0 = P0.n
                def pstats(p):
                    a = A[p]; ins = a[a < n0]; c = np.bincount(Fa[ins], minlength=len(self.names))
                    fid = [self.names.index(f) for f in p]
                    return dict(minpi=int(min(c[j] for j in fid)), pi=float(min(c[j] / nf[j] for j in fid)), ev=float(np.median(self.ev[ins])))
                PS = {p: pstats(p) for p in R0}
                hits = {p: te_score(p) >= muK for p in R0}
                Rzs = Rz[zmain]; target = len(Rzs); filt = {}
                for key in ("minpi", "pi", "ev"):
                    vals = sorted({PS[p][key] for p in R0})
                    best = None
                    for th in vals:
                        S_ = {p for p in R0 if PS[p][key] >= th}
                        if best is None or abs(len(S_) - target) < abs(len(best[1]) - target): best = (th, S_)
                    th, S_ = best
                    only_f, only_z = S_ - Rzs, Rzs - S_
                    hr = lambda Ps: round(float(np.mean([hits[p] for p in Ps])), 4) if Ps else None
                    filt[key] = dict(th=th, n=len(S_), J=jac(S_, Rzs), hold=hr(S_), hold_z=hr(Rzs), n_only_f=len(only_f), n_only_z=len(only_z),
                                     hold_only_f=hr(only_f), hold_only_z=hr(only_z),
                                     hits_only_f=int(sum(hits[p] for p in only_f)), hits_only_z=int(sum(hits[p] for p in only_z)),
                                     train_only_f=round(float(np.mean([R0all[p] for p in only_f])), 6) if only_f else None,
                                     train_only_z=round(float(np.mean([R0all[p] for p in only_z])), 6) if only_z else None)
                    if only_f and only_z:   # score-matched: each only-filter pattern paired with the only-UHUCM pattern of nearest training score
                        zl = sorted(only_z, key=lambda q: R0all[q]); zv = np.array([R0all[q] for q in zl]); used_z = set(); fp = []
                        for p in sorted(only_f, key=lambda q: R0all[q]):
                            order = np.argsort(np.abs(zv - R0all[p]))
                            jj = next((int(i) for i in order if int(i) not in used_z), int(order[0])); used_z.add(jj); fp.append((p, zl[jj]))
                        hf_, hz_ = [hits[p] for p, _ in fp], [hits[q] for _, q in fp]
                        filt[key]["pair_cal"] = table2x2(caliper_pairs(only_f, only_z, lambda q: R0all[q], 0.05 * muK), lambda q: hits[q])   # a = only-filter, b = only-UHUCM
                        filt[key]["pair2x2"] = dict(both=int(sum(x and y for x, y in zip(hf_, hz_))), only_z=int(sum((not x) and y for x, y in zip(hf_, hz_))),
                                                    only_f=int(sum(x and not y for x, y in zip(hf_, hz_))), neither=int(sum((not x) and (not y) for x, y in zip(hf_, hz_))),
                                                    mean_gap=round(float(np.mean([R0all[q] - R0all[p] for p, q in fp])), 6))
                row["filters"] = filt
                row["n_kept"], row["n_rej"] = len(kept), len(rej)
                row["hits_kept"] = int(sum(hits[p] for p in kept)); row["hits_rej"] = int(sum(hits[p] for p in rej))
                if rej and kept: row["hits_matched"] = int(sum(hits[q] for q in mk_)); row["n_matched"] = len(mk_)
            # E4 (Rule L), z = 0 and z = 1.645
            for z, Pb in ((0.0, P0), (zmain, Pz[zmain])):
                Rfull = R0 if z == 0 else Ru
                for var in ("global", "sum"):
                    RL = set(mine(Pb, muK, rule_l=var)[0])
                    row[f"L{var}_miss_z{z}"] = len(Rfull - RL)
            # CASE (K = 300): five rejected patterns with the largest train score
            if K == 300 and rej:
                top = sorted(rej, key=lambda p: -R0all[p])[:5]
                row["case"] = [dict(p="+".join(p), s0=round(R0all[p], 5), lb=round(Pz[zmain].score(A[p]), 5), atoms=int(len(A[p])),
                                    med_ev=float(np.median(self.ev[A[p][A[p] < P0.n]]))) for p in top]
            rows.append(row)
            short = {k: v for k, v in row.items() if k not in ("case", "FE_bins", "filters")}
            if "filters" in row: self.R("FILT", K=K, f=json.dumps(row["filters"]))
            self.R("E0", **short); self.R("E3bins", K=K, bins=row["FE_bins"])
            if "case" in row: self.R("CASE", K=K, top=json.dumps(row["case"]))
        self.rec["quality"] = rows
        return rows

    # ------------------------------------------------------------------ interactions (beta)
    def interaction(self, K=100, betas=(1.0,)):
        if self.interactions is None: return
        pairs = self.interactions(self.rows, self.users, self.NU, self.P0.nb)
        iv = np.array(list(pairs.values())) if pairs else np.zeros((0, 2))
        out = dict(pairs=len(pairs), pos=round(float((iv[:, 0] > 0).mean()), 3) if len(iv) else 0, var=round(float(iv[:, 1].mean()), 5) if len(iv) else 0)
        A0 = {p for p, v in self.R0all.items() if v >= self.sc[K - 1]}
        for b in betas:
            Pb0 = self.mk(pairs=pairs, beta=b); Pb1 = self.mk(z=self.zmain, pairs=pairs, beta=b)
            Rb0, scb = self.topk(Pb0, K)
            mK = scb[K - 1]; B0 = {p for p, v in Rb0.items() if v >= mK}; B1 = set(mine(Pb1, mK)[0])
            out[f"beta{b}"] = dict(R0=len(B0), Rz=len(B1), J=jac(B0, B1), J_vs_beta0=jac(B0, A0))
        self.R("INTER", **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in out.items()})
        self.rec["interaction"] = out

    # ------------------------------------------------------------------ E8
    def sensitivity(self, K=300):
        if self.recompute is None: return
        muK = self.sc[K - 1]
        base = set(mine(self.mk(z=self.zmain), muK)[0]); out = {}
        for name, kw in self.variants:
            mu, s2 = self.recompute(**kw)
            Pb = self.mk(mu=np.array(mu), s2=np.array(s2), z=self.zmain)
            P0v = self.mk(mu=np.array(mu), s2=np.array(s2))
            # same rank-based threshold: the K-th z=0 score of the variant
            _, scv = self.topk(P0v, K)
            Rv = set(mine(Pb, scv[K - 1])[0])
            out[name] = dict(R=len(Rv), J=jac(base, Rv))
        self.R("E8", **{k: f"{v['R']}/{v['J']}" for k, v in out.items()})
        self.rec["sensitivity"] = out

    # ------------------------------------------------------------------ E9
    def dependence(self, K=300):
        out = {}
        if self.resid is not None:
            r = np.asarray(self.resid, float); a, b = [], []
            for i, S in enumerate(self.P0.nb):
                for j in S:
                    if i < j: a.append(r[i]); b.append(r[j])
            a, b = np.array(a + b), np.array(b + a)
            out["rho_hat"] = round(float(np.corrcoef(a, b)[0, 1]), 4) if len(a) > 2 else None
            out["pairs"] = len(a) // 2
        muK = self.sc[K - 1]; prev = None; rows = {}
        rhos = [0.0, 0.3, 0.6] + ([max(0.0, out["rho_hat"])] if out.get("rho_hat") is not None else [])
        for rho in sorted(set(round(x, 4) for x in rhos)):
            Pb = self.mk(z=self.zmain, rho=rho)
            (R, s, _), t = timed(lambda: mine(Pb, muK))
            rows[str(rho)] = dict(R=len(R), evaluated=s["evaluated"], time=round(t, 2), set=set(R))
        ks = sorted(rows, key=float)
        nested = all(rows[ks[i + 1]]["set"] <= rows[ks[i]]["set"] for i in range(len(ks) - 1))
        for k in ks: rows[k].pop("set")
        out.update(rows=rows, nested=nested)
        self.R("E9", **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in out.items()})
        self.rec["dependence"] = out

    # ------------------------------------------------------------------ efficiency
    def run_variant(self, Pb, mu, use=(1, 1, 1, 1), kind="exact", reps=None):
        try:
            (R, s, _), t = timed(lambda: mine(Pb, mu, use, kind, deadline=self.deadline), reps or self.reps)
            return dict(time=round(t, 3), evaluated=s["evaluated"], bounds=s["bound"], R=len(R))
        except Timeout:
            return dict(time=f">{self.deadline}", evaluated=None, bounds=None, R=None)

    def efficiency(self, Ks=(30, 100, 300, 1000, 3000), zs=(0.0, 1.0, 1.645, 2.45, 4.36), ds=(30, 40, 50, 60, 70)):
        Pz = self.mk(z=ZMAIN)
        R0all, sc = self.topk(self.P0, max(Ks))
        vs_mu = {}
        for K in Ks:
            if K > len(sc): continue
            muK = sc[K - 1]; row = {}
            for name, use, kind in (("exact", (1, 1, 1, 1), "exact"), ("chord", (1, 1, 1, 1), "chord"),
                                    ("mean", (1, 1, 1, 1), "mean"), ("noprune", (0, 0, 0, 0), "exact")):
                row[name] = self.run_variant(Pz, muK, use, kind)
            vs_mu[K] = row
            self.R("EFFmu", K=K, mu=round(muK, 6), **{k: f"{v['time']}s/{v['evaluated']}n/{v['R']}" for k, v in row.items()})
        # ablation at K = 300
        muK = sc[min(300, len(sc)) - 1]; abl = {}
        for name, use, kind in (("full", (1, 1, 1, 1), "exact"), ("noFUB", (0, 1, 1, 1), "exact"), ("noPCB", (1, 0, 1, 1), "exact"),
                                ("noSB", (1, 1, 0, 1), "exact"), ("noNP", (1, 1, 1, 0), "exact"), ("chord", (1, 1, 1, 1), "chord"),
                                ("mean", (1, 1, 1, 1), "mean"), ("noprune", (0, 0, 0, 0), "exact")):
            abl[name] = self.run_variant(Pz, muK, use, kind)
        self.R("ABL", **{k: f"{v['time']}s/{v['evaluated']}n/{v['bounds']}b/{v['R']}" for k, v in abl.items()})
        # vs z at a fixed threshold (K = 300 of z = 0)
        vs_z = {}
        for z in zs:
            Pb = self.mk(z=z)
            vs_z[z] = {k: self.run_variant(Pb, muK, (1, 1, 1, 1), k, reps=1) for k in ("exact", "chord", "mean")}
        self.R("EFFz", **{str(z): "/".join(f"{v[k]['evaluated']}" for k in ("exact", "chord", "mean")) for z, v in vs_z.items()})
        # vs distance threshold (K = 300 of z = 0 at each d)
        vs_d = {}
        for d in ds:
            try:
                t = time.time(); P0d = self.mk(d=d); tb = time.time() - t
                if len(P0d.cliques) == 0: continue
                _, scd = self.topk(P0d, 300); m = scd[min(300, len(scd)) - 1]
                Pzd = self.mk(z=ZMAIN, d=d)
                vs_d[d] = dict(cliques=len(P0d.cliques), omega=max(map(len, P0d.cliques)), build=round(tb, 2),
                               exact=self.run_variant(Pzd, m, (1, 1, 1, 1), "exact", reps=1),
                               noprune=self.run_variant(Pzd, m, (0, 0, 0, 0), "exact", reps=1))
                v = vs_d[d]
                self.R("EFFd", d=d, cliques=v["cliques"], omega=v["omega"], build=v["build"],
                       exact=f"{v['exact']['time']}s/{v['exact']['evaluated']}n", noprune=f"{v['noprune']['time']}s/{v['noprune']['evaluated']}n")
            except MemoryError:
                self.R("EFFd", d=d, status="memory"); break
        self.rec["efficiency"] = dict(vs_mu=vs_mu, ablation=abl, vs_z=vs_z, vs_d=vs_d)

    def timing(self, Ks=(30, 100, 300, 1000, 3000), reps=10, ds=None, z=None, abl_K=300):
        """Repeated wall-clock timing of the mining step (clique build excluded): exact bound, mean bound and NoPrune at
        each rank threshold and distance, plus the bound/strategy ablation at K = abl_K; the variants are run in
        interleaved order, reps times each, and every time is recorded."""
        z = self.zmain if z is None else z; out = {}
        V = [("exact", (1, 1, 1, 1), "exact"), ("mean", (1, 1, 1, 1), "mean"), ("noprune", (0, 0, 0, 0), "exact")]
        A = [("noFUB", (0, 1, 1, 1), "exact"), ("noPCB", (1, 0, 1, 1), "exact"), ("noSB", (1, 1, 0, 1), "exact"), ("noNP", (1, 1, 1, 0), "exact"), ("chord", (1, 1, 1, 1), "chord")]
        for d in (ds or (self.d,)):
            P0d = self.mk(d=d); _, scd = self.topk(P0d, max(Ks)); Pzd = self.mk(z=z, d=d); res = {"cliques": len(P0d.cliques)}
            for K in Ks:
                if K > len(scd): continue
                m = scd[K - 1]; VV = V + (A if (K == abl_K and d == self.d) else []); rec = {n: dict(times=[]) for n, _, _ in VV}
                for _ in range(reps):
                    for n, use, kind in VV:
                        t = time.time(); R, stt, _ = mine(Pzd, m, use, kind); rec[n]["times"].append(round(time.time() - t, 4))
                        rec[n]["evaluated"], rec[n]["R"] = stt["evaluated"], len(R)
                res[K] = rec
                self.R("TIME", d=d, K=K, **{n: f"{np.mean(r['times']):.4f}+-{np.std(r['times'], ddof=1):.4f}/{r['evaluated']}" for n, r in rec.items()})
            out[d] = res
        self.rec["timing"] = out
        return out

    def save(self):
        json.dump(self.rec, open(os.path.join(self.out, f"suite_{self.name}.json"), "w"), indent=1, default=str)
