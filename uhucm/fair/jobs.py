"""Manifest of every experiment behind the paper, run by fair/run.py under one protocol.

Each job runs in its own empty directory fair_runs/<name>/ (scripts write their usual relative output folders there),
with one thread, a fixed hash seed and the environment given here. A job is rerun whenever its fingerprint changes:
command, environment, the source of the script and of every local module it imports, the identity of its input files
(name, size and a hash of their first megabyte), the Python/NumPy/SciPy versions and the fingerprints of the jobs it
depends on. Data locations are placeholders ({FSQ_DIR}, {YELP_DIR}, {PSOMUCP_DIR}, {RUNS}, {H}) resolved from the
environment at run time, so the fingerprint does not depend on where the data are mounted.

Protocol shared by all jobs
  * Semi-synthetic counts: seeds 0-19 are drawn for every setting, and the tables use, for each (layout, truth, K),
    the first 10 seeds whose true scores contain at least K patterns. Validity depends on the truth only, so every
    method and every ablation is scored on the same draws, the same truth set (moment center, threshold = K-th
    largest TRUE score) and the same z = 1.645.
  * Real data: z = 1.645, K in {100, 300, 1000}, d = 50 m (30 and 70 m for the sensitivity), total variance V_T with
    B = 20 and the expected normalizer for ratings (Yelp) and counts (Foursquare).
  * Timing: exclusive jobs (nothing else runs), all in one session, 1 untimed warm-up run, then 5 timed runs of every
    variant at every K in rotated order, garbage collector off inside the timed region, wall and CPU time recorded.
"""
import os

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # .../uhucm
BASE_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
            "FSQ_DIR": "{FSQ_DIR}", "PSOMUCP_DIR": "{PSOMUCP_DIR}"}
FSQ = {c: f"{{FSQ_DIR}}/dataset_TSMC2014_{c}.txt" for c in ("NYC", "TKY")}
SEMI_SEEDS = "20"                     # seeds 0-19 drawn; tables keep the first 10 valid ones per (setting, K)

JOBS = []


def job(name, script, args=(), env=None, inputs=(), exclusive=False, after=(), group=None, subdirs=(), deps=()):
    JOBS.append(dict(name=name, script=os.path.join(H, script), args=list(args), env={**BASE_ENV, **(env or {})},
                     inputs=list(inputs), exclusive=exclusive, after=list(after), group=group, subdirs=list(subdirs),
                     deps=[os.path.join(H, d) for d in deps]))


# ---- semi-synthetic calibration (Table V, ablations, spatial field) ----------------------------------------------
SEMI_ENV = {"FR": "1.0", "KS": "100,300,1000", "OUTD": "out", "TRUTHK": "1", "TRUTHC": "momc", "SEED0": "0"}
METHODS = {"np": ["np", "zpost", "cboot", "diag", "momc"],          # UHUCPM: compound EB, V_T, Z = sum E[u+], moment center
           "oracle": ["oracle", "zpost", "diag", "momc"],           # true prior and center
           "gamma": ["gamma", "diag"]}                              # parametric Gamma-Poisson posterior
ABL = {"abl_plugZ": ["np", "cboot", "diag", "momc"],                # plug-in normalizer sum mu+
       "abl_median": ["np", "zpost", "cboot", "diag"],              # median center
       "abl_noVT": ["np", "zpost", "diag", "momc"],                 # posterior variance only
       "abl_venueboot": ["np", "zpost", "boot", "diag", "momc"]}    # resample venues instead of counts
for c in ("NYC", "TKY"):
    for t in ("lognormal", "gamma", "mixture"):
        for m, fl in METHODS.items():
            job(f"semi_{c}_{t}_{m}", "semisyn/semisyn_fsq.py", [c, SEMI_SEEDS, t] + fl, SEMI_ENV, [FSQ[c]],
                group=f"semi_{c}_{t}", subdirs=["pilot/out"])
    for a, fl in ABL.items():
        job(f"semi_{c}_lognormal_{a}", "semisyn/semisyn_fsq.py", [c, SEMI_SEEDS, "lognormal"] + fl, SEMI_ENV, [FSQ[c]],
            group=f"semi_{c}_lognormal", subdirs=["pilot/out"])
    job(f"semi_{c}_lognormal_spf", "semisyn/semisyn_fsq.py", [c, SEMI_SEEDS, "lognormal"] + METHODS["np"],
        {**SEMI_ENV, "SPF": "100,0.3"}, [FSQ[c]], group=f"semi_{c}_spf", subdirs=["pilot/out"])

# ---- EviGen (rating setting) and exactness --------------------------------------------------------------------------
for e in ("e1", "e2", "e2b", "e2c", "e2d"):
    job(f"synth_{e}", "synth.py", [e])
job("synth_vt", "synth_vt.py")
job("running_example", "running_example.py")

# ---- Foursquare real data (Tables IV, VI, VII for D4/D5, filters, Strategy 4 on real data, case study) -------------
for c in ("NYC", "TKY"):
    job(f"fsq_{c}_quality", "run_suite_ceb.py", [FSQ[c], c, "out_suite_ceb", "quality"], inputs=[FSQ[c]], group="real")
    for d in ("30", "70"):
        job(f"fsq_{c}_quality_d{d}", "run_suite_ceb.py", [FSQ[c], c, "out_suite_ceb", "quality", d], inputs=[FSQ[c]], group="real_d")
    job(f"fsq_{c}_sens", "run_inter_ceb.py", [c], inputs=[FSQ[c]], subdirs=["out_suite_ceb"])
    job(f"fsq_{c}_bands", "run_bands_ceb.py", [c], inputs=[FSQ[c]], subdirs=["out_suite_ceb"])

# ---- Yelp (Tables IV, VI, VII for D1-D3 and Y1-Y8, sensitivity, d = 30/70 m) ----------------------------------------
YCITIES = ["Philadelphia", "New Orleans", "Tampa", "Tucson", "Indianapolis", "Nashville", "Reno", "Saint Louis",
           "Edmonton", "Santa Barbara", "Boise"]
YJ = [f"{{YELP_DIR}}/yelp_academic_dataset_{x}.json" for x in ("business", "checkin", "review")]
job("yelp_prep", "prep_yelp_reviews.py", ["--archive", "{YELP_DIR}", "--out", "slim", "--no-tips", "--cities", ",".join(YCITIES)],
    inputs=YJ)
for i, c in enumerate(YCITIES):
    a = ["{YELP_DIR}", "{RUNS}/yelp_prep/slim", c]
    job(f"yelp_{c.replace(' ', '_')}_quality", "run_yelp_fair.py", a + ["50"] + (["sens"] if i < 3 else []),
        inputs=YJ[:2], after=["yelp_prep"], group="real")
    if i < 3:
        for d in ("30", "70"):
            job(f"yelp_{c.replace(' ', '_')}_quality_d{d}", "run_yelp_fair.py", a + [d], inputs=YJ[:2], after=["yelp_prep"], group="real_d")

# ---- PSO-MUCP public data: authors' code vs UHUCPM, Strategy-4 audit and counterexample ------------------------------
D8 = ["d8/compare_d8.py", "d8/run_authors.py"]
PSO = ["{PSOMUCP_DIR}/HUCPM/PSO-MUCP.py"]
for w in ("cal", "lv"):
    job(f"d8_{w}", "d8/run_d8_all.py", [w], inputs=PSO, deps=D8)
job("d8_counterexample", "d8/compare_d8.py", ["{H}/d8/strategy4_counterexample.csv", "10", "0.5", "{'A': 10, 'B': -10, 'C': -1}", "--authors"],
    inputs=PSO + ["{H}/d8/strategy4_counterexample.csv"], deps=D8)

# ---- timing (Fig. 3, Table VIII); exclusive, one session ------------------------------------------------------------
for c, ks in (("NYC", (20, 40, 60, 80)), ("TKY", (20, 40, 60))):
    for k in ks:
        job(f"time_{c}_k{k}", "run_timing_fair.py", [c, str(k)], inputs=[FSQ[c]], exclusive=True, group="timing")
for K in ("30", "100", "300", "1000"):                                 # D5 at k = 80, one K per job
    job(f"time_TKY_k80_K{K}", "run_timing_fair.py", ["TKY", "80", K], inputs=[FSQ["TKY"]], exclusive=True, group="timing")
for c in ("NYC", "TKY"):
    for k in ("20", "60"):
        job(f"zsweep_{c}_k{k}", "run_zsweep.py", [c, k], inputs=[FSQ[c]], exclusive=True, group="timing")
