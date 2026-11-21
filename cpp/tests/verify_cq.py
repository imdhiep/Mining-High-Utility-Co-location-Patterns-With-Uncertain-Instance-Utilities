"""The authors' CQ-HUCPM (C++, built by cq/build_cq.sh) in total-utility mode against brute force.
Brute force: every clique with one instance per feature found by extending cliques one instance at a time, participating
instances of each pattern, score = sum of their utilities / NORM. A pattern is reported when its score reaches the
threshold. CQ stores coordinates and utilities in float32, so patterns whose score is within TOL of the threshold are
not compared, and coordinates are kept small (as in the experiments, where they are translated to the origin).
Usage: python verify_cq.py CQ_BINARY [N_DATASETS]
  The fixed build must agree on every dataset; the original build is expected to miss patterns (the 2-clique bug)."""
import math, os, random, re, subprocess, sys, tempfile

TOL = 1e-4


def brute(inst, d):
    n = len(inst)
    adj = [[inst[i][0] != inst[j][0] and (inst[i][1] - inst[j][1]) ** 2 + (inst[i][2] - inst[j][2]) ** 2 <= d * d
            for j in range(n)] for i in range(n)]
    pats = {}
    frontier = [(i,) for i in range(n)]
    while frontier:
        nxt = []
        for c in frontier:
            for j in range(c[-1] + 1, n):
                if all(adj[i][j] for i in c):
                    e = c + (j,)
                    feats = tuple(sorted(inst[i][0] for i in e))
                    if len(set(feats)) < len(feats):
                        continue
                    pats.setdefault(feats, set()).update(e)
                    nxt.append(e)
        frontier = nxt
    return pats


def run_cq(binary, path, d, thr, norm):
    out = subprocess.run([binary, path, repr(d), repr(thr), "--patterns", "--score", "total", "--norm", repr(norm)],
                         capture_output=True, text=True, check=True).stdout
    found = {}
    for line in out.splitlines():
        m = re.fullmatch(r"([A-Za-z]+) : (\S+)", line.strip())   # pattern lines only (timing lines also contain " : ")
        if m:
            found["".join(sorted(m.group(1)))] = float(m.group(2))
    return found


def main(binary, count):
    rng = random.Random(4)
    bad = compared = nonempty = 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "d.csv")
        for it in range(count):
            k = rng.randint(2, 6); n = rng.randint(8, 45); d = rng.choice([10.0, 15.0, 22.0])
            inst = [(chr(65 + rng.randrange(k)), round(rng.uniform(0, 60), 3), round(rng.uniform(0, 60), 3),
                     round(rng.uniform(-1, 3) if it % 2 else rng.uniform(0.1, 3), 4)) for _ in range(n)]
            with open(path, "w") as f:
                f.write("Feature,Instance,LocX,LocY,Checkin\n")
                cnt = {}
                for ft, x, y, u in inst:
                    cnt[ft] = cnt.get(ft, 0) + 1
                    f.write(f"{ft},{cnt[ft]},{x},{y},{u}\n")
            norm = sum(max(u, 0) for *_, u in inst)
            pats = brute([(ft, x, y) for ft, x, y, _ in inst], d)
            scores = {"".join(p): sum(inst[i][3] for i in s) / norm for p, s in pats.items() if len(p) >= 2}
            for thr in (0.0, 0.05, 0.15):
                got = run_cq(binary, path, d, thr, norm)
                expected = {p for p, s in scores.items() if s >= thr}
                nonempty += bool(expected)
                for p in set(expected) | set(got):
                    if abs(scores.get(p, math.inf) - thr) < TOL:
                        continue
                    compared += 1
                    if (p in expected) != (p in got) or (p in got and abs(got[p] - scores[p]) > TOL):
                        bad += 1
                        if bad <= 5:
                            print("mismatch", it, thr, p, scores.get(p), got.get(p))
    print(f"{count} datasets x 3 thresholds, {compared} pattern decisions compared, {bad} mismatches, "
          f"{nonempty} non-empty answers")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200) else 0)
