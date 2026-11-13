"""Runner for the experiment manifest (fair/jobs.py).

  python fair/run.py status [pattern]
  python fair/run.py run [-j N] [--budget MIN] [--exclusive-only | --no-exclusive] [pattern]
        runs stale jobs, at most N at a time (default 2). Exclusive jobs start only when nothing else runs, after the
        machine has been idle for 2 minutes, and nothing else starts while they run. No new job starts after the
        budget (minutes, default 100); running jobs finish.
  python fair/run.py invalidate pattern        forces a rerun of the matching jobs

Data locations come from the environment: FSQ_DIR, YELP_DIR, PSOMUCP_DIR (see paths.py for the defaults).
A job is stale when it has no successful stamp or when its fingerprint differs from the stamp: command and environment
with the data-location placeholders unresolved, source of the script, of every local module it imports (found by
scanning the import statements) and of the extra files listed in `deps`, identity of the input files (name, size and
a hash of the first megabyte), Python/NumPy/SciPy versions and fingerprints of the jobs in `after`. A stale job is
rerun from an empty directory fair_runs/<name>/. Each stamp keeps what is needed to recompute its fingerprint on
another machine with the same code (versions and input identities), which fair/check.py does."""
import fnmatch, hashlib, json, os, platform, shutil, subprocess, sys, time, ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import JOBS, H

ROOT = os.environ.get("FAIR_RUNS", os.path.join(H, "fair_runs"))
BY = {j["name"]: j for j in JOBS}
sys.path.insert(0, H)
import paths
PLACE = {"FSQ_DIR": paths.FSQ_DIR, "PSOMUCP_DIR": paths.PSOMUCP_DIR, "RUNS": ROOT, "H": H,
         "YELP_DIR": os.environ.get("YELP_DIR", os.path.join(paths.ROOT, "data", "yelp"))}
SESSION = f"{platform.node()}-{int(time.time())}"


def resolve(x):
    for k, v in PLACE.items(): x = x.replace("{" + k + "}", v)
    return x


def local_deps(script):
    """The script and, recursively, every module it imports that is a file of the experiment code under H."""
    seen, stack = set(), [os.path.abspath(script)]
    while stack:
        f = stack.pop()
        if f in seen: continue
        seen.add(f)
        for node in ast.walk(ast.parse(open(f).read())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    ([node.module] + [f"{node.module}.{a.name}" for a in node.names] if isinstance(node, ast.ImportFrom) and node.module else [])
            for nm in names:
                for base in (os.path.dirname(f), H):
                    cand = os.path.join(base, *nm.split(".")) + ".py"
                    if os.path.isfile(cand) and "fair_runs" not in cand: stack.append(os.path.abspath(cand))
    return sorted(seen)


def versions():
    import numpy, scipy
    return dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__)


def input_id(path):
    p = resolve(path)
    if not os.path.exists(p): return None
    with open(p, "rb") as f: head = hashlib.sha256(f.read(1 << 20)).hexdigest()[:16]
    return f"{os.path.basename(p)}|{os.path.getsize(p)}|{head}"


def stamp_path(name): return os.path.join(ROOT, name, "stamp.json")


def stamp(name):
    p = stamp_path(name)
    return json.load(open(p)) if os.path.exists(p) else None


def fingerprint(j, recorded=False, memo=None):
    """(fingerprint, code files, input identities). recorded=True uses the versions and input identities stored in the
    stamps (to check runs made on another machine against the current code)."""
    memo = {} if memo is None else memo
    if j["name"] in memo: return memo[j["name"]]
    s = stamp(j["name"]) if recorded else None
    vers = s["versions"] if s else versions()
    ids = s["inputs"] if s else [input_id(f) for f in j["inputs"]]
    h = hashlib.sha256()
    h.update(json.dumps(dict(script=os.path.relpath(j["script"], H), args=j["args"], env=j["env"], v=vers), sort_keys=True).encode())
    files = sorted(set(local_deps(j["script"])) | {f for d in j.get("deps", []) for f in local_deps(d)})
    for f in files:
        h.update(os.path.relpath(f, H).encode()); h.update(open(f, "rb").read())
    h.update(json.dumps(ids).encode())
    for a in j["after"]:
        h.update(fingerprint(BY[a], recorded, memo)[0].encode())
    memo[j["name"]] = (h.hexdigest()[:16], [os.path.relpath(f, H) for f in files], ids)
    return memo[j["name"]]


def state(j, recorded=False):
    if os.path.exists(os.path.join(ROOT, j["name"], "RUNNING")): return "running"
    s = stamp(j["name"])
    if not recorded and any(input_id(f) is None for f in j["inputs"]): return "blocked"
    if s is None: return "missing"
    if s.get("exit") != 0: return "failed"
    if s.get("fingerprint") != fingerprint(j, recorded)[0]: return "stale"
    return "done"


def select(pattern):
    return [j for j in JOBS if not pattern or fnmatch.fnmatch(j["name"], pattern)]


def loadavg(): return float(open("/proc/loadavg").read().split()[0])


def start(j):
    d = os.path.join(ROOT, j["name"]); fp = fingerprint(j)
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)
    for s in j["subdirs"]: os.makedirs(os.path.join(d, s), exist_ok=True)
    open(os.path.join(d, "RUNNING"), "w").write(str(time.time()))
    env = {**os.environ, **{k: resolve(v) for k, v in j["env"].items()}}
    log = open(os.path.join(d, "stdout.txt"), "w")
    p = subprocess.Popen([sys.executable, "-u", j["script"]] + [resolve(a) for a in j["args"]], cwd=d, env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    return dict(job=j, proc=p, t0=time.time(), load=[loadavg()], log=log, dir=d, fp=fp)


def finish(r):
    j, d = r["job"], r["dir"]; r["log"].close()
    fp, deps, ids = r["fp"]                               # fingerprint of the code and data as they were at the start
    outs = {}
    for root, _, files in os.walk(d):
        for f in files:
            if f in ("stamp.json", "RUNNING"): continue
            p = os.path.join(root, f); outs[os.path.relpath(p, d)] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    json.dump(dict(name=j["name"], fingerprint=fp, exit=r["proc"].returncode, start=r["t0"], seconds=round(time.time() - r["t0"], 1),
                   exclusive=j["exclusive"], load_samples=r["load"], deps=deps, inputs=ids, cmd=[os.path.relpath(j["script"], H)] + j["args"],
                   env=j["env"], versions=versions(), host=dict(node=platform.node(), cpus=os.cpu_count(), session=SESSION),
                   outputs=outs), open(stamp_path(j["name"]), "w"), indent=1)
    os.remove(os.path.join(d, "RUNNING"))
    print(f"{time.strftime('%H:%M')} {'ok ' if r['proc'].returncode == 0 else 'ERR'} {j['name']} ({time.time() - r['t0']:.0f}s)", flush=True)


def run(pattern, nj, budget, mode):
    os.makedirs(ROOT, exist_ok=True)
    for j in JOBS:                                       # clear leftovers of interrupted runs
        if os.path.exists(os.path.join(ROOT, j["name"], "RUNNING")): os.remove(os.path.join(ROOT, j["name"], "RUNNING"))
    todo = [j for j in select(pattern) if state(j) in ("missing", "stale", "failed")
            and (mode != "exclusive" or j["exclusive"]) and (mode != "shared" or not j["exclusive"])]
    print(f"{len(todo)} jobs to run, session {SESSION}, versions {versions()}", flush=True)
    t_end = time.time() + 60 * budget; running = []
    while todo or running:
        for r in [r for r in running if r["proc"].poll() is not None]:
            finish(r); running.remove(r)
        for r in running:
            if r["job"]["exclusive"] and int(time.time() - r["t0"]) % 5 == 0: r["load"].append(loadavg())
        ex = any(r["job"]["exclusive"] for r in running)
        if todo and time.time() < t_end and not ex:
            ready = [j for j in todo if all(state(BY[a]) == "done" for a in j["after"])]
            nxt = next((j for j in ready if not j["exclusive"]), None) if len(running) < nj else None
            if nxt is None and not running:
                nxt = next((j for j in ready if j["exclusive"]), None)
                if nxt is not None: time.sleep(120)       # let the machine settle before an exclusive job
            if nxt is not None:
                todo.remove(nxt); running.append(start(nxt)); continue
            if not running and not ready: break           # the rest waits for jobs that failed or are blocked
        if not running and (not todo or time.time() >= t_end): break
        time.sleep(1)
    left = [j["name"] for j in select(pattern) if state(j) != "done"]
    print("not done:", len(left), " ".join(left), flush=True)


def status(pattern):
    for j in select(pattern):
        s = state(j); st = stamp(j["name"]); sec = st.get("seconds") if st else ""
        print(f"{s:8s} {'X' if j['exclusive'] else ' '} {j['name']:34s} {sec}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]; cmd = a.pop(0) if a else "status"
    nj = int(a[a.index("-j") + 1]) if "-j" in a else 2
    budget = float(a[a.index("--budget") + 1]) if "--budget" in a else 100
    mode = "exclusive" if "--exclusive-only" in a else "shared" if "--no-exclusive" in a else "all"
    pat = next((x for i, x in enumerate(a) if not x.startswith("-") and (i == 0 or a[i - 1] not in ("-j", "--budget"))), "")
    if cmd == "status": status(pat)
    elif cmd == "run": run(pat, nj, budget, mode)
    elif cmd == "invalidate":
        for j in select(pat):
            if os.path.exists(stamp_path(j["name"])): os.remove(stamp_path(j["name"]))
