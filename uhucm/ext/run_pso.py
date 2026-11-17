"""Run the authors' PSO-MUCP.py with its mining logic unchanged and substituted I/O (data path, Utility dict,
Min_utility, D), as run_authors.py does, and with the normalizer of run_cq_fair.py: the code divides by the sum of all
instance utilities, which is negative on D4 and D5, so it receives the total positive utility of the instances.
Usage: python run_pso_fair.py PSO_SRC DATA_CSV UTILDICT MINUTIL D -> one JSON line."""
import sys, re, io, csv, contextlib, time, json, ast
src, data, util, mu, d = sys.argv[1:6]
code = open(src, encoding="utf-8").read()
code = re.sub(r'f = open\(r".*?", "r",\s*encoding="UTF-8"\)', f'f = open(r"{data}", "r", encoding="UTF-8")', code, flags=re.S)
code = re.sub(r"^Utility = \{.*?\}$", f"Utility = {util}", code, flags=re.M)
code = re.sub(r"^Min_utility = .*$", f"Min_utility = {mu}", code, flags=re.M)
code = re.sub(r"^D = .*$", f"D = {d}", code, flags=re.M)
code = code.replace("import psutil", "").replace("process = psutil.Process(os.getpid())\n    return process.memory_info().rss / 1024 / 1024", "return 0.0")
for a, b in (("print(Hash_table)", "pass"),
             ("    S_1, ALL_utility = Compute_utility(Utility, Instance)\n",
              "    S_1, ALL_utility = Compute_utility(Utility, Instance)\n    ALL_utility = __NORM\n")):
    assert code.count(a) == 1, a
    code = code.replace(a, b)
u = ast.literal_eval(util)
norm = sum(max(float(u[r["Feature"]]), 0.0) for r in csv.DictReader(open(data)))
buf = io.StringIO(); t = time.time()
with contextlib.redirect_stdout(buf):
    exec(compile(code, src, "exec"), {"__name__": "__main__", "__NORM": norm})
el = time.time() - t
m = re.search(r"^(\[.*\]) 高效用模式$", buf.getvalue(), flags=re.M)
pats = eval(m.group(1)) if m else None
print(json.dumps(dict(mode="psomucp", n=len(pats) if pats is not None else None,
                      patterns=sorted("".join(sorted(p)) for p in pats) if pats else [], seconds=round(el, 2))))
