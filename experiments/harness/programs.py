"""Calls to the C++ programs. Each method returns the parsed JSON line of the program (or, for CQ-HUCPM, the patterns
and scores it prints)."""
import json, os, re, subprocess


class ProgramError(RuntimeError):
    pass


class Programs:
    def __init__(self, bin_dir, cq_binary):
        self.bin_dir = bin_dir
        self.cq_binary = cq_binary

    def _json(self, name, args):
        cmd = [os.path.join(self.bin_dir, name)] + [str(a) for a in args]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode not in (0, 3):
            raise ProgramError(f"{name} failed ({p.returncode}): {p.stderr[-2000:]}")
        return json.loads(p.stdout.strip().splitlines()[-1])

    def posterior(self, data_csv, count_column, exposure, out, truncation=0, samples=20, seed=0, prior="npmle",
                  center="moment"):
        return self._json("post_counts", [data_csv, count_column, repr(float(exposure)), out, "--t", truncation, "--B", samples,
                                          "--seed", seed, "--prior", prior, "--center", center])

    def mine(self, data_csv, posterior, gamma, distance, z, top=None, set_level=None, top_probability=None):
        """uhucpm_mine; with top_probability the scores are the probabilities p of reaching gamma"""
        args = [data_csv, posterior, repr(float(gamma)), repr(float(distance)), repr(float(z)), "--scores"]
        if top is not None:
            args += ["--top", int(top)]
        if top_probability is not None:
            args += ["--top-probability", int(top_probability)]
        if set_level is not None:
            args += ["--set-level", repr(float(set_level))]
        r = self._json("uhucpm_mine", args)
        r["score_of"] = dict(zip(r["patterns"], r.get("scores", [])))
        return r

    def pattern_scores(self, data_csv, posterior, distance, z, patterns, work_dir):
        path = os.path.join(work_dir, "patterns.txt")
        with open(path, "w") as f:
            f.write("".join(p + "\n" for p in patterns))
        return self._json("pattern_scores", [data_csv, posterior, repr(float(distance)), repr(float(z)), path])

    def pso(self, data_csv, utility, min_utility, distance, normalizer):
        spec = ",".join(f"{k}:{float(v)!r}" for k, v in sorted(utility.items()))
        return self._json("pso_mucp", [data_csv, spec, repr(float(min_utility)), repr(float(distance)), "--norm", repr(float(normalizer))])

    def cq(self, data_csv, distance, threshold, normalizer):
        """the authors' CQ-HUCPM (fixed build) with the total-utility score; returns {pattern: score} and seconds"""
        cmd = [self.cq_binary, data_csv, repr(float(distance)), repr(float(threshold)), "--patterns", "--score", "total",
               "--norm", repr(float(normalizer))]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise ProgramError(f"CQ-HUCPM failed ({p.returncode}): {p.stderr[-2000:]}")
        scores, seconds, memory = {}, None, None
        for line in p.stdout.splitlines():
            m = re.fullmatch(r"([A-Za-z]+) : (\S+)", line.strip())
            if m:
                scores["".join(sorted(m.group(1)))] = float(m.group(2))
            elif line.startswith("Time taken by the program is"):
                seconds = float(line.split(":")[1].split()[0])
            elif line.startswith("Peack memory usage"):
                memory = int(line.split(":")[1].split()[0])
        return dict(scores=scores, seconds=seconds, max_rss_kb=memory)
