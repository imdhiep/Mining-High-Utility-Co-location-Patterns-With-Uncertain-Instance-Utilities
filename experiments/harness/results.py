"""Raw result files (experiment rule 1.6): one JSON line per run with the full configuration and every output; files
are never overwritten (one new file per invocation, named by the start time)."""
import json, os, socket, subprocess, time


def code_version(repo_dir):
    try:
        return subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


class RawWriter:
    def __init__(self, results_dir, experiment, repo_dir):
        os.makedirs(os.path.join(results_dir, "raw", experiment), exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(results_dir, "raw", experiment, f"{stamp}.jsonl")
        if os.path.exists(self.path):
            raise FileExistsError(self.path)
        self.context = dict(experiment=experiment, code=code_version(repo_dir), host=socket.gethostname(), started=stamp)

    def write(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(self.context, **record), separators=(",", ":")) + "\n")
