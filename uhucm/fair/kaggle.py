"""Kaggle driver: locates the inputs under /kaggle/input, fetches the authors' PSO-MUCP code, then runs one phase of the
manifest with fair/run.py and packs the results into /kaggle/working.

  python fair/kaggle.py shared    all jobs that do not measure time (one job per CPU core)
  python fair/kaggle.py timing    the timing and z-sweep jobs, one at a time with nothing else running
  python fair/kaggle.py check     fairness checks on the results in FAIR_RUNS

Results: /kaggle/working/fair_runs/ and /kaggle/working/fair_runs_<phase>.tar.gz. To continue in a later session, add
the earlier output as input and pass its fair_runs folder with FAIR_PREV; it is copied first, so finished jobs whose
fingerprint still matches are not run again."""
import glob, os, shutil, subprocess, sys, tarfile, time

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = "/kaggle/working"


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits: sys.exit(f"input not found: {name}")
    return os.path.dirname(sorted(hits, key=len)[0])


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True); return subprocess.run(cmd, check=False, **kw).returncode


phase = sys.argv[1]
env = dict(os.environ, FAIR_RUNS=f"{W}/fair_runs")
env["FSQ_DIR"] = find("dataset_TSMC2014_NYC.txt")
env["YELP_DIR"] = find("yelp_academic_dataset_business.json")
pso = f"{W}/PSO-MUCP"
if not os.path.isdir(pso):
    sh(["git", "clone", "--depth", "1", "https://github.com/yuanshuaikang/PSO-MUCP", pso])
env["PSOMUCP_DIR"] = pso
try:
    import pyproj  # noqa: F401
except ImportError:
    sh([sys.executable, "-m", "pip", "install", "-q", "pyproj"])
if os.environ.get("FAIR_PREV") and not os.path.isdir(env["FAIR_RUNS"]):
    shutil.copytree(os.environ["FAIR_PREV"], env["FAIR_RUNS"])
print({k: env[k] for k in ("FSQ_DIR", "YELP_DIR", "PSOMUCP_DIR", "FAIR_RUNS")}, "cpus", os.cpu_count(), flush=True)
run = [sys.executable, os.path.join(H, "fair", "run.py")]
if phase == "shared":
    sh(run + ["run", "-j", str(os.cpu_count()), "--budget", "660", "--no-exclusive"], env=env)
elif phase == "timing":
    sh(run + ["run", "-j", "1", "--budget", "660", "--exclusive-only"], env=env)
sh(run + ["status"], env=env)
sh([sys.executable, os.path.join(H, "fair", "check.py")], env=env)
if phase in ("shared", "timing"):
    with tarfile.open(f"{W}/fair_runs_{phase}.tar.gz", "w:gz") as t:
        t.add(env["FAIR_RUNS"], arcname="fair_runs")
print("finished", time.strftime("%H:%M"), flush=True)
