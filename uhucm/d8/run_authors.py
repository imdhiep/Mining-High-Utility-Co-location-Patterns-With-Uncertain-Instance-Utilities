"""Run the authors' PSO-MUCP.py (logic unchanged) with substituted I/O: data path, Utility dict, Min_utility, D.
--no-s4 disables only Pruning Strategy 4 (one condition in gener_k). Prints #HUCP, patterns and wall time."""
import sys, re, io, contextlib, time, json
src, data, util, mu, d = sys.argv[1:6]; no_s4 = '--no-s4' in sys.argv
code = open(src, encoding='utf-8').read()
code = re.sub(r'f = open\(r".*?", "r",\s*encoding="UTF-8"\)', f'f = open(r"{data}", "r", encoding="UTF-8")', code, flags=re.S)
code = re.sub(r"^Utility = \{.*?\}$", f"Utility = {util}", code, flags=re.M)
code = re.sub(r"^Min_utility = .*$", f"Min_utility = {mu}", code, flags=re.M)
code = re.sub(r"^D = .*$", f"D = {d}", code, flags=re.M)
code = code.replace("import psutil", "").replace("process = psutil.Process(os.getpid())\n    return process.memory_info().rss / 1024 / 1024", "return 0.0")
s4 = "if compute_list[k_1_can_key][1] / all_utility <= min_utility and sort_list[\n                i] in negative:"
assert s4 in code, "Strategy-4 line not found"
if no_s4: code = code.replace(s4, "if False:")
code = code.replace("print(Hash_table)", "pass")
buf = io.StringIO(); t = time.time()
with contextlib.redirect_stdout(buf):
    exec(compile(code, src, 'exec'), {'__name__': '__main__'})
el = time.time() - t
m = re.search(r"^(\[.*\]) 高效用模式$", buf.getvalue(), flags=re.M)
pats = eval(m.group(1)) if m else None
print(json.dumps(dict(mode="nos4" if no_s4 else "orig", n=len(pats) if pats is not None else None,
                      patterns=sorted("".join(sorted(p)) for p in pats) if pats else [], seconds=round(el, 2))))
