"""fastmine: vectorised UHUCM (same semantics as uhucm.mine, verified against it and against brute force).

Data layout
  instances 0..n-1 with feature index feat[i] in 0..k-1, posterior (mu_i, s2_i)
  pair atoms n..n+m-1 for neighbouring pairs with an interaction value: mean beta*iota, var beta^2*ivar
  maximal cliques as a dense matrix Mem (C x k): Mem[c,f] = instance of feature f in clique c, or -1
  pair atoms per clique as Pid (C x k x k) (only when there are interaction atoms)
Bounds
  exact  : SR_z via prefix scan in the global sigma^2/mu order (Thm. prefix), cross-multiplication comparator
  chord  : greatest-affine chord bound (Cor. chord), capped by the mean bound
  mean   : sum of positive means (ignores variance; the UL-(mu,sigma^2) style baseline)
  rho>0  : robust bound min(SR_{z sqrt(1-rho)}, sum (mu - z sqrt(rho) sigma)^+) (Thm. robust)
Search (Alg. UHUCM): iterative FUB, order by FUB (non-positive tier last), DFS over clique lists,
  PS2 = subtree bound at the node (PCB), PS3 = suffix bound with break (SB), PS4 = non-positive tier.
"""
import math, functools, collections
import numpy as np


class Problem:
    def __init__(self, feat, xy, mu, s2, d, z, rho=0.0, pairs=None, beta=0.0, feat_names=None):
        """feat: int array (n,), xy: (n,2) float, mu/s2: (n,) float, d: distance threshold.
        pairs: optional dict {(i,j): (iota, ivar)} with i<j (kept only for neighbours)."""
        self.n = len(feat); self.feat = np.asarray(feat); self.k = int(self.feat.max()) + 1
        self.names = feat_names or [str(f) for f in range(self.k)]
        self.z, self.rho, self.beta, self.d = z, rho, beta, d
        xy = np.asarray(xy, float); self.xy = xy; self.bands = None
        self.nb = self._neighbours(xy, d)
        self.cliques = self._maximal_cliques()
        C = len(self.cliques)
        self.Mem = -np.ones((C, self.k), dtype=np.int64)
        for c, Q in enumerate(self.cliques):
            for o in Q:
                self.Mem[c, self.feat[o]] = o
        amu = [float(x) for x in mu]; as2 = [float(x) for x in s2]
        self.pair_index = {}
        self.Pid = None
        if pairs and beta > 0:
            for (i, j), (io, iv) in pairs.items():
                a, b = min(i, j), max(i, j)
                if b in self.nb[a] and (a, b) not in self.pair_index:
                    self.pair_index[(a, b)] = len(amu)
                    amu.append(beta * io); as2.append(beta * beta * iv)
            if self.pair_index:
                self.Pid = -np.ones((C, self.k, self.k), dtype=np.int64)
                for c, Q in enumerate(self.cliques):
                    for a in Q:
                        for b in Q:
                            if a < b and (a, b) in self.pair_index:
                                fa, fb = self.feat[a], self.feat[b]
                                self.Pid[c, fa, fb] = self.Pid[c, fb, fa] = self.pair_index[(a, b)]
        self.amu = np.array(amu); self.as2 = np.array(as2)
        self.Z = float(np.clip(self.amu, 0, None).sum())
        # global order of positive-mean atoms by s2/mu (exact comparator, no division, no epsilon)
        posi = [a for a in range(len(amu)) if amu[a] > 0]
        cmp = lambda a, b: (as2[a] * amu[b] > as2[b] * amu[a]) - (as2[a] * amu[b] < as2[b] * amu[a])
        posi.sort(key=functools.cmp_to_key(cmp))
        self.rank = -np.ones(len(amu), dtype=np.int64)
        self.rank[posi] = np.arange(len(posi))
        self.smu = self.amu[posi]; self.ss2 = self.as2[posi]

    # ---------------- geometry ----------------
    def _neighbours(self, xy, d):
        grid = collections.defaultdict(list)
        cell = np.floor(xy / d).astype(np.int64)
        for i, (gx, gy) in enumerate(cell):
            grid[(gx, gy)].append(i)
        nb = [set() for _ in range(self.n)]
        d2 = d * d + 1e-9
        for (gx, gy), L in grid.items():
            cand = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cand += grid.get((gx + dx, gy + dy), [])
            cand = np.array(cand); P = xy[cand]
            for i in L:
                dd = ((P - xy[i]) ** 2).sum(1)
                for j in cand[(dd <= d2)]:
                    if j != i and self.feat[j] != self.feat[i]:
                        nb[i].add(int(j))
        return nb

    def _maximal_cliques(self):
        """Bron-Kerbosch with pivoting in degeneracy order (iterative), cliques of size >= 2."""
        nb = self.nb; out = []
        deg = {v: len(nb[v]) for v in range(self.n) if nb[v]}
        order = []; D = dict(deg); buckets = collections.defaultdict(set)
        for v, dv in D.items(): buckets[dv].add(v)
        removed = set(); cur = 0
        for _ in range(len(D)):
            cur = 0
            while not buckets[cur]: cur += 1
            v = buckets[cur].pop(); removed.add(v); order.append(v)
            for w in nb[v]:
                if w not in removed and w in D:
                    buckets[D[w]].discard(w); D[w] -= 1; buckets[D[w]].add(w)
        pos = {v: i for i, v in enumerate(order)}
        for v in order:
            P = {w for w in nb[v] if pos[w] > pos[v]}
            X = {w for w in nb[v] if pos[w] < pos[v]}
            stack = [({v}, P, X)]
            while stack:
                R, P, X = stack.pop()
                if not P and not X:
                    if len(R) >= 2: out.append(tuple(sorted(R)))
                    continue
                piv = max(P | X, key=lambda u: len(nb[u] & P))
                for u in list(P - nb[piv]):
                    stack.append((R | {u}, P & nb[u], X & nb[u]))
                    P = P - {u}; X = X | {u}
        return out

    # ---------------- atoms and bounds ----------------
    def atoms(self, L, feats):
        """atom ids of the instances of `feats` in cliques L, plus pair atoms among them."""
        if len(L) == 0: return np.empty(0, dtype=np.int64)
        fs = np.asarray(sorted(feats))
        M = self.Mem[np.ix_(L, fs)].ravel()
        mark = np.zeros(len(self.amu), dtype=bool); mark[M[M >= 0]] = True
        if self.Pid is not None and len(fs) > 1:
            Pm = self.Pid[np.ix_(L, fs, fs)].ravel(); mark[Pm[Pm >= 0]] = True
        return np.flatnonzero(mark)

    def set_bands(self, edges, rhos):
        """Distance-banded correlation of instance atoms: corr(a, b) = rhos[k] for distances in (edges[k], edges[k+1]]
        (the first band includes 0) and 0 beyond edges[-1]. All rhos must be non-negative, so the variance can only
        grow and the bounds computed under independence (rho = 0) remain valid upper bounds."""
        assert self.rho == 0 and all(r >= 0 for r in rhos) and len(edges) == len(rhos) + 1
        self.bands = (np.asarray(edges[1:], float), np.append(np.asarray(rhos, float), 0.0))

    def _banded_extra(self, inst):
        if len(inst) < 2: return 0.0
        s = np.sqrt(self.as2[inst]); X = self.xy[inst]; e, rr = self.bands; tot = 0.0
        for a in range(0, len(inst), 1024):
            D = np.sqrt(((X[a:a + 1024, None, :] - X[None, :, :]) ** 2).sum(-1))
            R = rr[np.searchsorted(e, D, side="left")]
            tot += float(s[a:a + 1024] @ (R @ s))
        return tot - rr[0] * float((s * s).sum())          # remove the diagonal (distance 0 falls in the first band)

    def score(self, A):
        m = self.amu[A].sum(); v = self.as2[A]
        if self.bands is not None:
            V = v.sum() + self._banded_extra(A[A < self.n])
            return (m - self.z * math.sqrt(max(V, 0.0))) / self.Z
        if self.rho > 0:
            V = (1 - self.rho) * v.sum() + self.rho * np.sqrt(v).sum() ** 2
        else:
            V = v.sum()
        return (m - self.z * math.sqrt(max(V, 0.0))) / self.Z

    def _sr(self, A, z):
        r = self.rank[A]; r = np.sort(r[r >= 0])
        if len(r) == 0: return 0.0
        M = np.cumsum(self.smu[r])
        if z <= 0: return float(M[-1])
        V = np.cumsum(self.ss2[r])
        return max(0.0, float((M - z * np.sqrt(V)).max()))

    def bound(self, A, kind='exact'):
        if len(A) == 0: return 0.0
        mu = self.amu[A]; P = mu > 0
        meanb = float(mu[P].sum())
        if kind == 'mean' or self.z <= 0: return meanb / self.Z
        if kind == 'chord':
            zz = self.z * math.sqrt(max(0.0, 1 - self.rho)) if self.rho < 1 else 0.0
            s2 = self.as2[A][P]; Vb = s2.sum()
            ch = meanb if Vb <= 0 else float(np.clip(mu[P] - zz * s2 / math.sqrt(Vb), 0, None).sum())
            return min(ch, meanb) / self.Z
        if self.rho > 0:
            b1 = self._sr(A, self.z * math.sqrt(1 - self.rho))
            b2 = float(np.clip(mu - self.z * math.sqrt(self.rho) * np.sqrt(self.as2[A]), 0, None).sum())
            return min(b1, b2) / self.Z
        return self._sr(A, self.z) / self.Z


    # ---------------- fast pruning test (works on ranks of positive-mean atoms only) ----------------
    def _ranks(self, L, feats):
        fs = np.asarray(sorted(feats))
        M = self.Mem[np.ix_(L, fs)].ravel(); M = M[M >= 0]
        r = self.rank[M]
        if self.Pid is not None and len(fs) > 1:
            Pm = self.Pid[np.ix_(L, fs, fs)].ravel(); Pm = Pm[Pm >= 0]
            r = np.concatenate([r, self.rank[Pm]])
        return r[r >= 0]

    def prunable(self, L, feats, thr, kind="two", st=None):
        """True iff the subtree bound is < thr (thr already multiplied by Z). Same value as bound() for
        exact/chord/mean; 'two' = chord first, exact only if the chord bound does not prune."""
        r = self._ranks(L, feats)
        if len(r) == 0: return 0.0 < thr
        if kind == "mean" or self.z <= 0:
            return float(self.smu[np.unique(r)].sum()) < thr
        if self.rho > 0:
            A = np.unique(r)                      # robust bound needs per-atom sigma: use the general routine
            mu, s2 = self.smu[A], self.ss2[A]
            b1 = self._sr_sorted(mu, s2, self.z * math.sqrt(1 - self.rho))
            b2 = float(np.clip(mu - self.z * math.sqrt(self.rho) * np.sqrt(s2), 0, None).sum())
            return min(b1, b2) < thr
        u = np.unique(r); mu, s2 = self.smu[u], self.ss2[u]
        if kind in ("chord", "two"):
            Vb = s2.sum()
            ch = float(mu.sum()) if Vb <= 0 else float(np.clip(mu - self.z * s2 / math.sqrt(Vb), 0, None).sum())
            if st is not None: st["chord"] += 1
            if ch < thr:
                if st is not None: st["chord_pruned"] += 1
                return True
            if kind == "chord": return False
        if st is not None: st["exact"] += 1
        return self._sr_sorted(mu, s2, self.z) < thr

    @staticmethod
    def _sr_sorted(mu, s2, z):
        if len(mu) == 0: return 0.0
        M = np.cumsum(mu)
        if z <= 0: return float(M[-1])
        return max(0.0, float((M - z * np.sqrt(np.cumsum(s2))).max()))

    def suffix_bounds(self, L, P, T, kind="two", st=None, thr=None):
        """Bounds of all suffix candidate sets at a node in one pass.
        Entry j of the result is the bound (times Z) of the candidate set used by SB for T[j:]
        (rows of L having a feature in T[j:], features P + T[j:]); entry 0 is the PCB bound."""
        P = list(P); T = list(T); cols = P + T; nP = len(P); nT = len(T)
        Msub = self.Mem[np.ix_(L, cols)]                       # rows x cols
        posT = np.full(len(cols), -1); posT[nP:] = np.arange(nT)
        present = Msub >= 0
        tp = np.where(present[:, nP:], np.arange(nT)[None, :], -1)
        maxpos = tp.max(1)                                      # largest T position present in the row
        keyP = np.repeat(maxpos[:, None], nP, 1) if nP else np.zeros((len(L), 0), int)
        keyT = np.where(present[:, nP:], np.arange(nT)[None, :], -1)
        key = np.concatenate([keyP, keyT], 1)
        inst = Msub[present]; ikey = key[present]
        parts_r = [self.rank[inst]]; parts_k = [ikey]
        if self.Pid is not None and len(cols) > 1:
            Pm = self.Pid[np.ix_(L, cols, cols)]
            ok = Pm >= 0
            if ok.any():
                rr, a, b = np.nonzero(ok)
                pa, pb = posT[a], posT[b]
                k2 = np.where((pa < 0) & (pb < 0), maxpos[rr], np.where(pa < 0, pb, np.where(pb < 0, pa, np.minimum(pa, pb))))
                parts_r.append(self.rank[Pm[ok]]); parts_k.append(k2)
        r = np.concatenate(parts_r); k = np.concatenate(parts_k)
        keep = (r >= 0) & (k >= 0)
        r, k = r[keep], k[keep]
        out = np.zeros(nT)
        if len(r) == 0: return out
        # marker array indexed by global rank: no sorting needed, ranks come out in sigma^2/mu order
        bestall = np.full(len(self.smu), -1); np.maximum.at(bestall, r, k)
        u = np.flatnonzero(bestall >= 0); best = bestall[u]
        mu, s2 = self.smu[u], self.ss2[u]
        for j in range(nT):
            m = best >= j
            if not m.any(): out[j] = 0.0; continue
            mj, sj = mu[m], s2[m]
            if kind == "mean" or self.z <= 0:
                out[j] = float(mj.sum()); continue
            if self.rho > 0:
                b1 = self._sr_sorted(mj, sj, self.z * math.sqrt(1 - self.rho))
                b2 = float(np.clip(mj - self.z * math.sqrt(self.rho) * np.sqrt(sj), 0, None).sum())
                out[j] = min(b1, b2); continue
            if kind in ("chord", "two"):
                Vb = sj.sum()
                ch = float(mj.sum()) if Vb <= 0 else float(np.clip(mj - self.z * sj / math.sqrt(Vb), 0, None).sum())
                out[j] = ch
                if st is not None: st["chord"] += 1
                if kind == "chord": continue
                if ch < thr:
                    if st is not None: st["chord_pruned"] += 1
                    continue
            if st is not None: st["exact"] += 1
            out[j] = self._sr_sorted(mj, sj, self.z)
        return out


    def suffix_search(self, L, P, T, kind="two", st=None, thr=None, need_sb=True):
        """Same pruning decisions as suffix_bounds, with O(log |T|) bound evaluations per node: the candidate sets of
        the suffix bound are nested (I[j+1] is a subset of I[j]), so the exact bound is non-increasing in j
        (Proposition 7) and the first j with a bound below thr is found by binary search. The chord (or mean) bound is
        only used as a cheaper sufficient test inside each probe. Returns (pcb_prunes, brk)."""
        P = list(P); T = list(T); cols = P + T; nP = len(P); nT = len(T)
        Msub = self.Mem[np.ix_(L, cols)]
        posT = np.full(len(cols), -1); posT[nP:] = np.arange(nT)
        present = Msub >= 0
        tp = np.where(present[:, nP:], np.arange(nT)[None, :], -1)
        maxpos = tp.max(1)
        keyP = np.repeat(maxpos[:, None], nP, 1) if nP else np.zeros((len(L), 0), int)
        keyT = np.where(present[:, nP:], np.arange(nT)[None, :], -1)
        key = np.concatenate([keyP, keyT], 1)
        inst = Msub[present]; ikey = key[present]
        parts_r = [self.rank[inst]]; parts_k = [ikey]
        if self.Pid is not None and len(cols) > 1:
            Pm = self.Pid[np.ix_(L, cols, cols)]
            ok = Pm >= 0
            if ok.any():
                rr, a, b = np.nonzero(ok)
                pa, pb = posT[a], posT[b]
                k2 = np.where((pa < 0) & (pb < 0), maxpos[rr], np.where(pa < 0, pb, np.where(pb < 0, pa, np.minimum(pa, pb))))
                parts_r.append(self.rank[Pm[ok]]); parts_k.append(k2)
        r = np.concatenate(parts_r); k = np.concatenate(parts_k)
        keep = (r >= 0) & (k >= 0); r, k = r[keep], k[keep]
        if len(r) == 0: return True, 0
        bestall = np.full(len(self.smu), -1); np.maximum.at(bestall, r, k)
        u = np.flatnonzero(bestall >= 0); best = bestall[u]
        mu, s2 = self.smu[u], self.ss2[u]
        mean_only = kind == "mean" or self.z <= 0

        def below(j):
            m = best >= j
            if not m.any(): return True
            mj, sj = mu[m], s2[m]
            msum = float(mj.sum())
            if msum < thr: return True                      # the mean bound already prunes
            if mean_only: return False
            if self.rho > 0:
                b1 = self._sr_sorted(mj, sj, self.z * math.sqrt(1 - self.rho))
                b2 = float(np.clip(mj - self.z * math.sqrt(self.rho) * np.sqrt(sj), 0, None).sum())
                return min(b1, b2) < thr
            if kind in ("chord", "two"):
                Vb = sj.sum()
                ch = msum if Vb <= 0 else float(np.clip(mj - self.z * sj / math.sqrt(Vb), 0, None).sum())
                if st is not None: st["chord"] += 1
                if ch < thr:
                    if st is not None: st["chord_pruned"] += 1
                    return True
                if kind == "chord": return False
            if st is not None: st["exact"] += 1
            return self._sr_sorted(mj, sj, self.z) < thr

        if st is not None: st["probe"] += 1
        if below(0): return True, 0
        if not need_sb: return False, nT
        if kind == "chord":                                  # the chord bound is not monotone in j: linear scan
            for j in range(1, nT):
                if st is not None: st["probe"] += 1
                if below(j): return False, j
            return False, nT
        lo, hi = 1, nT                                       # invariant: below(lo - 1) is False; answer in [lo, hi]
        while lo < hi:
            mid = (lo + hi) // 2
            if st is not None: st["probe"] += 1
            if below(mid): hi = mid
            else: lo = mid + 1
        return False, lo


class Timeout(Exception):
    pass


def mine(Pb, mu, use=(1, 1, 1, 1), kind='two', max_size=None, rule_l=None, deadline=None, search=True):
    """Returns (results {tuple(sorted feature names): score}, counters, info).
    rule_l: None | 'global' | 'sum'  -- the local pruning Rule L of Section III (for the audit E4 only; unsafe).
      'global': prune P+{g} if L(P) < mu and every atom of feature g (all its instances and pairs) has mean <= 0.
      'sum'   : prune P+{g} if L(P) < mu and the summed mean of g's atoms in the cliques hosting P+{g} is <= 0
                (feature-sign heuristic in the spirit of Strategy 4 of PSO-MUCP).
    deadline: wall-clock seconds; raises Timeout when exceeded."""
    import time as _t
    t_end = None if deadline is None else _t.time() + deadline
    st = collections.Counter(); tol = 1e-12
    allf = set(range(Pb.k)); C = len(Pb.cliques)
    hasf = Pb.Mem >= 0
    alive = set(f for f in range(Pb.k) if hasf[:, f].any())
    fub = {}
    while True:
        fub = {}
        for f in alive:
            L = np.nonzero(hasf[:, f])[0]
            fub[f] = Pb.bound(Pb.atoms(L, alive), 'exact' if kind == 'two' else kind); st['bound'] += 1
        bad = [f for f in alive if fub[f] < mu - tol] if use[0] else []
        if not bad: break
        alive -= set(bad)
    # non-positive tier: every alive instance mean <= 0 and every pair atom touching it <= 0
    npf = set()
    for f in alive:
        L = np.nonzero(hasf[:, f])[0]
        inst = np.unique(Pb.Mem[L, f])
        ok = (Pb.amu[inst] <= 0).all()
        if ok and Pb.Pid is not None:
            pa = Pb.Pid[L][:, f, :].ravel(); pa = pa[pa >= 0]
            ok = (Pb.amu[pa] <= 0).all() if len(pa) else True
        if ok: npf.add(f)
    order = sorted([f for f in alive if f not in npf], key=lambda f: (-fub[f], f)) + \
            sorted([f for f in alive if f in npf], key=lambda f: (-fub[f], f))
    rank = {f: r for r, f in enumerate(order)}
    res = {}
    gneg = {}
    if rule_l == 'global':
        for f in alive:
            L = np.nonzero(hasf[:, f])[0]
            inst = np.unique(Pb.Mem[L, f]); ok = bool((Pb.amu[inst] <= 0).all())
            if ok and Pb.Pid is not None:
                pa = Pb.Pid[L][:, f, :].ravel(); pa = pa[pa >= 0]
                ok = bool((Pb.amu[pa] <= 0).all()) if len(pa) else True
            gneg[f] = ok

    def visit(P, L):
        st['visited'] += 1
        if t_end is not None and st['visited'] % 64 == 0 and _t.time() > t_end:
            raise Timeout()
        v = None
        if len(P) >= 2:
            v = Pb.score(Pb.atoms(L, P)); st['evaluated'] += 1
            if v >= mu - tol: res[tuple(sorted(Pb.names[f] for f in P))] = v
        if max_size and len(P) >= max_size: return
        T = order[rank[P[-1]] + 1:]
        if not T: return
        if use[3] and all(f in npf for f in list(P) + T):
            st['ps4'] += 1; return
        Lp = L[hasf[np.ix_(L, T)].any(1)]
        if len(Lp) == 0: return
        thr = (mu - tol) * Pb.Z
        brk = len(T)
        if (use[1] or use[2]) and search:
            pcb, brk_ = Pb.suffix_search(Lp, P, T, kind, st, thr, need_sb=bool(use[2]))
            st['bound'] += 1
            if pcb:
                if use[1]:
                    st['ps2'] += 1; return
                if use[2]: brk = 0                       # PCB off: the suffix bound of T[0:] still applies
            elif use[2] and brk_ < len(T): brk = brk_; st['ps3'] += 1
        elif use[1] or use[2]:
            sb = Pb.suffix_bounds(Lp, P, T, kind, st, thr)   # 'two': chord first, exact only where needed
            st['bound'] += 1 + (len(T) if use[2] else 0)
            if use[1] and sb[0] < thr:
                st['ps2'] += 1; return
            if use[2]:
                below = np.nonzero(sb < thr)[0]
                if len(below): brk = int(below[0]); st['ps3'] += 1
        for gi, g in enumerate(T):
            if gi >= brk: break
            Lq = Lp[hasf[Lp, g]]
            if not len(Lq): continue
            if rule_l and v is not None and v < mu - tol:
                if rule_l == 'global' and gneg.get(g, False):
                    st['ruleL'] += 1; continue
                if rule_l == 'sum':
                    gi_ = Pb.Mem[Lq, g]; m = float(Pb.amu[np.unique(gi_)].sum())
                    if Pb.Pid is not None:
                        pa = Pb.Pid[np.ix_(Lq, [g], list(P))].ravel(); pa = np.unique(pa[pa >= 0]); m += float(Pb.amu[pa].sum())
                    if m <= 0:
                        st['ruleL'] += 1; continue
            visit(P + (g,), Lq)

    for f in order:
        L = np.nonzero(hasf[:, f])[0]
        if len(L): visit((f,), L)
    return res, st, dict(order=[Pb.names[f] for f in order], npf=[Pb.names[f] for f in npf],
                         removed=[Pb.names[f] for f in set(range(Pb.k)) - alive])


def pattern_atoms(Pb, names):
    idx = [Pb.names.index(x) for x in names]
    hasf = Pb.Mem >= 0
    L = np.nonzero(hasf[:, idx].all(1))[0]
    return Pb.atoms(L, idx)
