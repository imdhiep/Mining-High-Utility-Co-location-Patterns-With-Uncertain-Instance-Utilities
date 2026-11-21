"""Raw result files (experiment rule 1.6): one JSON line per run with the full configuration and every output; files
are never overwritten (one new file per invocation, named by the start time)."""
import glob, json, os, platform, socket, subprocess, time


def machine():
    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            model = next((l.split(":", 1)[1].strip() for l in f if l.startswith("model name")), "")
    except OSError:
        pass
    return dict(cpu=model or platform.processor(), cores=os.cpu_count())


def code_version(repo_dir):
    try:
        return subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


class RawWriter:
    def __init__(self, results_dir, experiment, repo_dir):
        self.dir = os.path.join(results_dir, "raw", experiment)
        os.makedirs(self.dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        self.path = os.path.join(self.dir, f"{stamp}.jsonl")
        if os.path.exists(self.path):
            raise FileExistsError(self.path)
        self.context = dict(experiment=experiment, code=code_version(repo_dir), host=socket.gethostname(), machine=machine(),
                            started=stamp)

    def completed(self):
        """(layout, value, seed) of every run already written for this experiment"""
        keys = set()
        for path in glob.glob(os.path.join(self.dir, "*.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    keys.add((r["layout"], json.dumps(r["value"]), r["seed"]))
        return keys

    def write(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(self.context, **record), separators=(",", ":")) + "\n")
