"""One UHUCPM run on a PSO-MUCP dataset with feature-level utilities (sigma = 0, z = 0) and the code's own normalizer,
in its own process, so that it can be timed exactly like the authors' code (run_authors.py).
Usage: python run_uhucm_d8.py DATA_CSV UTILDICT MINUTIL D   -> one JSON line with n, patterns and the inner times"""
import sys, os, ast, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from compare_d8 import uhucm_run

data, util, mu, d = sys.argv[1], ast.literal_eval(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
r = uhucm_run(data, util, d, mu)
print(json.dumps(dict(mode="uhucpm", n=r["n"], patterns=r["patterns"], seconds=r["seconds"])))
