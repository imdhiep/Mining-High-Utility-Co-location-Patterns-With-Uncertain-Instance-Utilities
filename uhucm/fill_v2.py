"""Fill numeric placeholders of tkde_v2/main_tmpl.tex from result files -> tkde_v2/main.tex.
Calibration table: out_ceb/semisyn_table2.{json,tex} (make_semisyn_table2.py). Extra text: out_ceb/fill_text.json."""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__)); T = os.path.join(H, "..", "tkde_v2")
s = open(os.path.join(T, "main_tmpl.tex")).read()
J = json.load(open(os.path.join(H, "out_ceb", "semisyn_table2.json")))
pt = [v["point"] for v in J.values() if v["point"] == v["point"]]
sub = {"@@SEMISYNROWS@@": open(os.path.join(H, "out_ceb", "semisyn_table2.tex")).read().strip(),
       "@@POINT@@": f"{100 * min(pt):.0f}\\%--{100 * max(pt):.0f}\\%"}
ex = os.path.join(H, "out_ceb", "fill_text.json")
if os.path.exists(ex): sub.update(json.load(open(ex)))
for k, v in sub.items(): s = s.replace(k, v)
left = sorted(set(re.findall(r"@@[A-Z0-9]+@@", s)))
open(os.path.join(T, "main.tex"), "w").write(s)
print("left:", left)
