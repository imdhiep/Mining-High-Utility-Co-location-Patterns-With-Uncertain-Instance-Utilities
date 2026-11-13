"""Kaggle driver: locates the inputs under /kaggle/input, fetches the authors' PSO-MUCP code, then runs one phase of the
manifest with fair/run.py and packs the results into /kaggle/working.

  python fair/kaggle.py shared    all jobs that do not measure time (one job per CPU core)
  python fair/kaggle.py timing    the timing and z-sweep jobs, one at a time with nothing else running
  python fair/kaggle.py check     fairness checks on the results in FAIR_RUNS
  An optional second argument restricts the phase to the jobs matching a pattern, e.g. "yelp*".

Results: /kaggle/working/fair_runs/ and /kaggle/working/fair_runs_<phase>.tar.gz. To continue in a later session, add
the earlier output as input and pass its fair_runs folder with FAIR_PREV; it is copied first, so finished jobs whose
fingerprint still matches are not run again."""
import glob, os, shutil, subprocess, sys, tarfile, time

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = "/kaggle/working"


FSQ_SHA = {"dataset_TSMC2014_NYC.txt": "2f39ce698e5bff6683c74bec30c436cc022c160b0f3d73c5268b55491b2445f9",
           "dataset_TSMC2014_TKY.txt": "85542c605ce708d8584637756590383a84e9bbdb71342d0d8e78858ba2a0e5c8"}
FSQ_GDRIVE = "1VE67jfSN-GV6fkCfuyBEQJzk_qy-SfOk"     # dataset_tsmc2014.zip, linked from the authors' page (D. Yang)


def find(name, roots=("/kaggle/input", W)):
    hits = [h for r in roots for h in glob.glob(f"{r}/**/{name}", recursive=True) if "fair_runs" not in h]
    return os.path.dirname(sorted(hits, key=len)[0]) if hits else None


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def fsq_dir():
    d = find("dataset_TSMC2014_NYC.txt")
    if d is None:                                        # not attached: fetch the original archive
        sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        sh([sys.executable, "-m", "gdown", FSQ_GDRIVE, "-O", f"{W}/dataset_tsmc2014.zip"])
        shutil.unpack_archive(f"{W}/dataset_tsmc2014.zip", f"{W}/fsq")
        d = find("dataset_TSMC2014_NYC.txt")
    for f, h in FSQ_SHA.items():                          # the same bytes as the files behind the paper
        if sha(os.path.join(d, f)) != h: sys.exit(f"{f}: checksum differs from the reference file")
    return d


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True); return subprocess.run(cmd, check=False, **kw).returncode


phase = sys.argv[1]; pattern = sys.argv[2] if len(sys.argv) > 2 else ""
env = dict(os.environ, FAIR_RUNS=f"{W}/fair_runs")
env["FSQ_DIR"] = fsq_dir()
env["YELP_DIR"] = find("yelp_academic_dataset_business.json") or ""
if phase == "shared" and not env["YELP_DIR"] and (not pattern or pattern.startswith("yelp")):
    sys.exit("Yelp JSON files not attached")
pso = f"{W}/PSO-MUCP"
if not os.path.isdir(pso):
    sh(["git", "clone", "--depth", "1", "https://github.com/yuanshuaikang/PSO-MUCP", pso])
env["PSOMUCP_DIR"] = pso
try:
    import pyproj  # noqa: F401
except ImportError:
    sh([sys.executable, "-m", "pip", "install", "-q", "pyproj"])
prev = os.environ.get("FAIR_PREV") or next(iter(sorted(glob.glob("/kaggle/input/**/fair_runs", recursive=True), key=len)), None)
if prev and not os.path.isdir(env["FAIR_RUNS"]):                # earlier output attached as input: keep its finished jobs
    print("copying earlier results from", prev, flush=True)
    shutil.copytree(prev, env["FAIR_RUNS"])
print({k: env[k] for k in ("FSQ_DIR", "YELP_DIR", "PSOMUCP_DIR", "FAIR_RUNS")}, "cpus", os.cpu_count(), flush=True)
run = [sys.executable, os.path.join(H, "fair", "run.py")]
if phase == "shared":
    sh(run + ["run", "-j", str(os.cpu_count()), "--budget", "660", "--no-exclusive"] + ([pattern] if pattern else []), env=env)
elif phase == "timing":
    sh(run + ["run", "-j", "1", "--budget", "660", "--exclusive-only"] + ([pattern] if pattern else []), env=env)
sh(run + ["status"] + ([pattern] if pattern else []), env=env)
for st in glob.glob(f"{env['FAIR_RUNS']}/*/stamp.json"):          # show the end of the output of every failed job
    import json
    if json.load(open(st)).get("exit") != 0:
        out = os.path.join(os.path.dirname(st), "stdout.txt")
        print(f"==== {os.path.basename(os.path.dirname(st))} (failed) ====\n" + open(out, errors="replace").read()[-3000:], flush=True)
sh([sys.executable, os.path.join(H, "fair", "check.py")], env=env)
if phase in ("shared", "timing"):
    with tarfile.open(f"{W}/fair_runs_{phase}.tar.gz", "w:gz") as t:
        t.add(env["FAIR_RUNS"], arcname="fair_runs")
print("finished", time.strftime("%H:%M"), flush=True)
