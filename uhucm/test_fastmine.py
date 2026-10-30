"""fastmine vs reference uhucm.brute_force on random instances: mixed signs, variances, interactions, rho, bounds."""
import random, itertools, numpy as np, sys
import uhucm, fastmine
from evigen import evigen
bad = 0; tot = 0; nonempty = 0
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
    rnd = random.Random(seed)
    n, k = rnd.randint(8, 30), rnd.randint(3, 6)
    G = evigen(n, k, 12.0, seed)
    inst = G['inst']; ids = [i for i, *_ in inst]
    names = sorted({f for _, f, _, _ in inst}); fi = {f: j for j, f in enumerate(names)}
    z = rnd.choice([0.0, 1.0, 1.645, 3.0]); rho = rnd.choice([0.0, 0.0, 0.3, 0.6]); beta = rnd.choice([0.0, 1.0])
    d = rnd.uniform(3, 6)
    D = uhucm.CData(inst, G['post_m'], G['post_v'], {}, {}, d, 0.0, z, rho)
    iota, ivar, pairs = {}, {}, {}
    if beta > 0:
        for a in ids:
            for b in D.nb[a]:
                if a < b and rnd.random() < 0.6:
                    v, w = rnd.uniform(-1, 1), rnd.uniform(0, 0.3)
                    iota[frozenset((a, b))] = v; ivar[frozenset((a, b))] = w
                    pairs[(ids.index(a), ids.index(b))] = (v, w)
    D = uhucm.CData(inst, G['post_m'], G['post_v'], iota, ivar, d, beta, z, rho)
    if D.Z <= 0: continue
    mu = rnd.choice([0.0, 0.02, 0.05, 0.1, 0.2])
    bf, allv = uhucm.brute_force(D, mu)
    Pb = fastmine.Problem([fi[f] for _, f, _, _ in inst], [(x, y) for _, _, x, y in inst],
                          [G['post_m'][i] for i in ids], [G['post_v'][i] for i in ids], d, z, rho, pairs, beta, names)
    assert abs(Pb.Z - D.Z) < 1e-9, (Pb.Z, D.Z)
    want = {tuple(sorted(P)) for P in bf}
    for use in [(1, 1, 1, 1), (0, 0, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1)]:
        for kind in ('two', 'exact', 'chord', 'mean'):
            r, st, _ = fastmine.mine(Pb, mu, use, kind)
            got = {tuple(sorted(P)) for P in r}
            tot += 1
            if got != want:
                bad += 1; print('MISMATCH seed', seed, use, kind, z, rho, beta, want ^ got); break
            for P, v in r.items():
                assert abs(v - allv[frozenset(P)]) < 1e-9
    nonempty += bool(want)
print(f'checked {tot} runs, mismatches {bad}, instances with non-empty answer {nonempty}')
