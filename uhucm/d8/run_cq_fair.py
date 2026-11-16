"""Run the CQ-HUCPM.py of the PSO-MUCP repository with its mining logic unchanged and substituted I/O: data path,
per-instance utilities (JSON keyed by feature letter + instance number), Min_util and D. The memory tracer that the
script starts (tracemalloc) and the psutil probe are switched off, so that its time is not inflated by measurement
code that UHUCPM does not run. Usage: python run_cq_fair.py CQ_SRC DATA_CSV UTIL_JSON MINUTIL D -> one JSON line."""
import sys, re, io, contextlib, time, json
src, data, ujson, mu, d = sys.argv[1:6]
code = open(src, encoding="utf-8").read()
code = re.sub(r'f = open\(r".*?", "r",\s*encoding="UTF-8"\)', f'f = open(r"{data}", "r", encoding="UTF-8")', code, flags=re.S)
code = code.replace("Instance_util[s] = Instance_utils[temp[0]]", "Instance_util[s] = __U[s]")
code = re.sub(r"^Min_util = .*$", f"Min_util = {mu}", code, flags=re.M)
code = re.sub(r"^D = .*$", f"D = {d}", code, flags=re.M)
code = code.replace("import psutil", "")
code = code.replace("process = psutil.Process(os.getpid())\n    return process.memory_info().rss / 1024 / 1024", "return 0.0")
assert code.count("tracemalloc.start()") == 1 and "__U[s]" in code
code = code.replace("tracemalloc.start()", "pass")
g = {"__name__": "__main__", "__U": json.load(open(ujson))}
buf = io.StringIO(); t = time.time()
with contextlib.redirect_stdout(buf):
    exec(compile(code, src, "exec"), g)
el = time.time() - t
m = re.search(r"^(\[.*\]) 高效用模式$", buf.getvalue(), flags=re.M)
pats = eval(m.group(1)) if m else None
print(json.dumps(dict(mode="cq", n=len(pats) if pats is not None else None,
                      patterns=sorted("".join(sorted(p)) for p in pats) if pats else [], seconds=round(el, 2))))
