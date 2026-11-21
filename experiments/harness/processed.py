"""Reading raw result files and writing processed tables (experiment rule 1.6): the aggregation scripts read
results/raw/EXPERIMENT/*.jsonl and write CSV files to results/processed; nothing here runs an algorithm."""
import csv, glob, json, math, os


def raw_records(results_dir, experiment):
    """every JSON line of every raw file of one experiment, in file order"""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "raw", experiment, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def mean_sd(values):
    """mean and sample standard deviation of the values that are not None; (None, None) when there are none"""
    v = [x for x in values if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, sd


def write_csv(path, rows, columns):
    """one CSV file; None is written as an empty cell (shown as "-" in the tables)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(["" if r.get(c) is None else (f"{r[c]:.10g}" if isinstance(r[c], float) else r[c]) for c in columns])
    return path


def ranked(score_of, n=None):
    """patterns by decreasing score, ties by name; the first n"""
    order = sorted(score_of, key=lambda q: (-score_of[q], q))
    return order if n is None else order[:n]
