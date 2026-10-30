"""UHUCM: high-utility co-location mining under uncertain instance utilities (reference implementation for the paper outline).

Atoms = instances (mean mu, var s2) and neighbour pairs (mean beta*iota, var beta^2*ivar).
Credible score  Lam_z(P) = ( sum_{a in A(P)} mu_a - z * sqrt(V(P)) ) / Z,
  V(P) = sum s2_a (independent)  or  V_rho(P) = (1-rho)*sum s2 + rho*(sum s)^2 (equicorrelated).
Bounds: SR_z(I) = max_{S subset I} [sum_S mu - z sqrt(sum_S s2)]  (prefix-optimal, Thm 4),
        computed on the candidate atom sets of FUB / PCB / SB (Thm 6); robust variant for rho (Thm 7).
"""
import itertools, math, collections

def pos(x): return x if x > 0 else 0.0

# ---------------- subset relaxation (Thm 4) ----------------
def SR(atoms, z, key=None):
    """atoms: list of (mu, s2). Returns max over subsets S (incl. empty) of sum mu - z*sqrt(sum s2)."""
    P = [(m, v) for m, v in atoms if m > 0]
    if z <= 0: return sum(m for m, _ in P)
    P.sort(key=lambda t: t[1] / t[0])
    best = 0.0; M = V = 0.0
    for m, v in P:
        M += m; V += v
        best = max(best, M - z * math.sqrt(V))
    return best

def SR_brute(atoms, z):
    best = 0.0
    for r in range(1, len(atoms) + 1):
        for S in itertools.combinations(atoms, r):
            best = max(best, sum(m for m, _ in S) - z * math.sqrt(sum(v for _, v in S)))
    return best

def chord_bound(atoms, z):
    """Cor 2: chord of sqrt on [0, Vbar+], Vbar+ = total variance of the POSITIVE-mean atoms only
    (dropping a non-positive-mean atom never lowers F, so only I+ matters)."""
    P = [(m, v) for m, v in atoms if m > 0]
    Vbar = sum(v for _, v in P)
    if Vbar <= 0: return sum(m for m, _ in P)
    c1 = 1 / math.sqrt(Vbar)
    return sum(pos(m - z * c1 * v) for m, v in P)

def SR_rho(atoms, z, rho):
    """valid upper bound of max_S [sum mu - z sqrt(V_rho(S))] (Thm 7)."""
    if rho <= 0: return SR(atoms, z)
    b1 = SR(atoms, z * math.sqrt(1 - rho))
    b2 = sum(pos(m - z * math.sqrt(rho) * math.sqrt(v)) for m, v in atoms)
    return min(b1, b2)

# ---------------- data ----------------
class CData:
    def __init__(self, inst, mu, s2, iota, ivar, d, beta, z, rho=0.0):
        self.inst = inst; self.feat = {i: f for i, f, _, _ in inst}
        self.xy = {i: (x, y) for i, _, x, y in inst}
        self.mu, self.s2, self.beta, self.z, self.rho = mu, s2, beta, z, rho
        ids = [i for i, *_ in inst]
        self.nb = {i: set() for i in ids}
        grid = collections.defaultdict(list)           # grid index: O(n) neighbour search instead of O(n^2)
        for i in ids: grid[(int(self.xy[i][0] // d), int(self.xy[i][1] // d))].append(i)
        for (gx, gy), cell in grid.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for b in grid.get((gx + dx, gy + dy), ()):
                        for a in cell:
                            if a < b and self.feat[a] != self.feat[b] and math.dist(self.xy[a], self.xy[b]) <= d + 1e-12:
                                self.nb[a].add(b); self.nb[b].add(a)
        self.iota = {k: v for k, v in iota.items() if tuple(k)[1] in self.nb[tuple(k)[0]]}
        self.ivar = {k: ivar.get(k, 0.0) for k in self.iota}
        self.F = sorted({f for f in self.feat.values()})
        self.Z = sum(pos(mu[i]) for i in ids) + beta * sum(pos(v) for v in self.iota.values())

    def atom(self, a):
        """a is an instance id or a frozenset pair -> (mean, var)"""
        if isinstance(a, frozenset):
            return (self.beta * self.iota.get(a, 0.0), self.beta ** 2 * self.ivar.get(a, 0.0))
        return (self.mu[a], self.s2[a])

    def score(self, atoms):
        ms = [self.atom(a) for a in atoms]
        M = sum(m for m, _ in ms)
        if self.rho > 0:
            V = (1 - self.rho) * sum(v for _, v in ms) + self.rho * sum(math.sqrt(v) for _, v in ms) ** 2
        else:
            V = sum(v for _, v in ms)
        return (M - self.z * math.sqrt(V)) / self.Z

    def bound(self, atoms):
        ms = [self.atom(a) for a in atoms]
        return SR_rho(ms, self.z, self.rho) / self.Z

    def maximal_cliques(self, alive=None):
        V = set(self.nb) if alive is None else set(alive)
        nb = {v: self.nb[v] & V for v in V}; out = []
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2: out.append(frozenset(R))
                return
            piv = max(P | X, key=lambda w: len(nb[w] & P))
            for v in list(P - nb[piv]):
                bk(R | {v}, P & nb[v], X & nb[v]); P = P - {v}; X = X | {v}
        bk(set(), set(V), set())
        return out

def pattern_atoms(D, P, cliques):
    atoms = set(); found = False
    for M in cliques:
        FM = {D.feat[o]: o for o in M}
        if set(P) <= set(FM):
            found = True
            R = [FM[f] for f in P]
            atoms |= set(R)
            for a, b in itertools.combinations(R, 2):
                if frozenset((a, b)) in D.iota: atoms.add(frozenset((a, b)))
    return atoms if found else None

def brute_force(D, mu_thr):
    by = collections.defaultdict(list)
    for i, f, *_ in D.inst: by[f].append(i)
    allv = {}
    for k in range(2, len(D.F) + 1):
        for P in itertools.combinations(D.F, k):
            atoms = set(); found = False
            for combo in itertools.product(*(by[f] for f in P)):
                if all(b in D.nb[a] for a, b in itertools.combinations(combo, 2)):
                    found = True; atoms |= set(combo)
                    for a, b in itertools.combinations(combo, 2):
                        if frozenset((a, b)) in D.iota: atoms.add(frozenset((a, b)))
            if found: allv[frozenset(P)] = D.score(atoms)
    return {P: v for P, v in allv.items() if v >= mu_thr - 1e-12}, allv

# ---------------- mining ----------------
def fub_candidates(D, alive, f):
    I = set()
    for o in alive:
        if D.feat[o] != f: continue
        N = [x for x in D.nb[o] if x in alive]
        I.add(o); I |= set(N)
        closed = N + [o]
        for a, b in itertools.combinations(closed, 2):
            p = frozenset((a, b))
            if b in D.nb[a] and p in D.iota: I.add(p)
    return I

def nonpositive(D, alive):
    feats = {D.feat[o] for o in alive}; out = set()
    for f in feats:
        inst = [o for o in alive if D.feat[o] == f]
        if all(D.mu[o] <= 0 for o in inst) and all(D.iota.get(frozenset((o, x)), 0) <= 0 for o in inst for x in D.nb[o] if x in alive):
            out.add(f)
    return out

def mine(D, mu_thr, use=(1, 1, 1, 1), bound_kind='exact', trace=None):
    """bound_kind: 'exact' (SR), 'chord', or 'mean' (bounds ignore variance: sum of positive means)."""
    def B(atoms):
        ms = [D.atom(a) for a in atoms]
        if bound_kind == 'exact': return SR_rho(ms, D.z, D.rho) / D.Z
        if bound_kind == 'chord': return min(chord_bound(ms, D.z * math.sqrt(1 - D.rho)) if D.rho < 1 else 1e18,
                                             sum(pos(m) for m, _ in ms)) / D.Z
        return sum(pos(m) for m, _ in ms) / D.Z
    st = collections.Counter()
    alive = set(D.nb); removed = []
    fubv = {}
    while use[0]:
        fubv = {f: B(fub_candidates(D, alive, f)) for f in {D.feat[o] for o in alive}}
        if trace is not None: trace.append(('FUB', {f: round(v, 4) for f, v in sorted(fubv.items())}))
        bad = [f for f, v in fubv.items() if v < mu_thr - 1e-12]
        if not bad: break
        removed += bad; alive = {o for o in alive if D.feat[o] not in bad}
    if not fubv: fubv = {f: B(fub_candidates(D, alive, f)) for f in {D.feat[o] for o in alive}}
    npf = nonpositive(D, alive)
    feats = sorted({D.feat[o] for o in alive})
    order = sorted([f for f in feats if f not in npf], key=lambda f: (-fubv[f], f)) + \
            sorted([f for f in feats if f in npf], key=lambda f: (-fubv[f], f))
    rank = {f: r for r, f in enumerate(order)}
    CQ = [{D.feat[o]: o for o in M} for M in D.maximal_cliques(alive)]
    if trace is not None: trace.append(('order', order, sorted(npf), 'removed', removed))
    res = {}

    def cand(P, Lp, keep):
        atoms = set()
        for c in Lp:
            M = CQ[c]; R = [M[f] for f in M if f in keep]
            atoms |= set(R)
            for a, b in itertools.combinations(R, 2):
                p = frozenset((a, b))
                if p in D.iota: atoms.add(p)
        return atoms

    def visit(P, L):
        st['visited'] += 1
        if len(P) >= 2:
            atoms = set()
            for c in L:
                R = [CQ[c][f] for f in P]; atoms |= set(R)
                for a, b in itertools.combinations(R, 2):
                    p = frozenset((a, b))
                    if p in D.iota: atoms.add(p)
            v = D.score(atoms); st['evaluated'] += 1
            if trace is not None: trace.append(('node', ''.join(P), round(v, 4)))
            if v >= mu_thr - 1e-12: res[frozenset(P)] = v
        last = rank[P[-1]]; T = order[last + 1:]
        if not T: return
        if use[3] and all(f in npf for f in list(P) + T):
            st['ps4'] += 1
            if trace is not None: trace.append(('PS4', ''.join(P)))
            return
        Lp = [c for c in L if any(rank[f] > last for f in CQ[c])]
        pcb = B(cand(P, Lp, set(P) | set(T)))
        if trace is not None: trace.append(('PCB', ''.join(P), round(pcb, 4)))
        if use[1] and pcb < mu_thr - 1e-12:
            st['ps2'] += 1; return
        for gi, g in enumerate(T):
            if use[2]:
                keep = set(T[gi:])
                Lg_plus = [c for c in Lp if any(f in keep for f in CQ[c])]   # cliques that can host P+{g'}, g' >= g
                sb = B(cand(P, Lg_plus, set(P) | keep))
                if sb < mu_thr - 1e-12:
                    st['ps3'] += 1
                    if trace is not None: trace.append(('PS3-break', ''.join(P), g, round(sb, 4)))
                    break
            Lq = [c for c in Lp if g in CQ[c]]
            if Lq: visit(P + (g,), Lq)

    for f in order:
        L = [c for c in range(len(CQ)) if f in CQ[c]]
        if L: visit((f,), L)
    return res, st, dict(order=order, npf=npf, removed=removed)
