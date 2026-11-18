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
  * Timing: exclusive jobs (nothing else runs), 1 untimed warm-up run, then 5 timed runs of every
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

# PLB and the prevalence/evidence filters on the same draws as the np runs (the records of semi_<c>_<t>_np are repeated)
for c in ("NYC", "TKY"):
    for t in ("lognormal", "gamma", "mixture"):
        job(f"x_semi_{c}_{t}", "semisyn/semisyn_extra.py", [c, SEMI_SEEDS, t] + METHODS["np"], SEMI_ENV, [FSQ[c]],
            group=f"x_semi_{c}_{t}", subdirs=["pilot/out"])

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

for c in ("NYC", "TKY"):                                              # effect of the second term of V_T alone
    job(f"x_vtonly_{c}", "run_vt_only.py", [c], inputs=[FSQ[c]])

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
# UHUCPM against the authors' code (whole processes, positive setting) and growth with n on EviGen
for w, f in (("cal", "California_POI 13f.csv"), ("lv", "LasVegas_POI 10f.csv")):
    job(f"time_d8_{w}", "d8/time_d8_fair.py", [w], inputs=PSO + [f"{{PSOMUCP_DIR}}/DATA/{f}"], exclusive=True, group="timing",
        deps=["d8/run_uhucm_d8.py", "d8/run_authors.py"])
job("time_scale_n", "run_scale_fair.py", exclusive=True, group="timing")
# UHUCPM at z = 0 against the authors' CQ-HUCPM (instance-level utilities) and PSO-MUCP (feature-level utilities) on D4/D5
CQS = ["{PSOMUCP_DIR}/HUCPM/CQ-HUCPM.py"]
EXTD = ["d8/run_uhucm_inst.py", "d8/run_cq_fair.py", "d8/run_pso_fair.py"]
job("time_ext_NYC", "d8/time_ext_fsq.py", ["NYC"], inputs=[FSQ["NYC"]] + PSO + CQS, exclusive=True, group="timing", deps=EXTD)
# D5: one K per Kaggle session. CQ-HUCPM takes hours per run on D5, so time_ext_d5.py gives it one timed run without
# warm-up (stopped after 40000 s to stay inside the 12-hour session); the fast programs get warm-up + 5 timed runs.
# Two sessions per K (r2 repeats r1 on another machine of the same type) give two CQ-HUCPM measurements.
for K in ("100", "300", "1000"):
    for r in ("", "_r2"):
        job(f"time_ext_TKY_K{K}{r}", "d8/time_ext_d5.py", [K], inputs=[FSQ["TKY"]] + PSO + CQS, exclusive=True,
            group="timing", deps=EXTD)
# Yelp D1-D3: same comparison with the rating posterior means; one Kaggle session per city (yelp_prep first)
for c in YCITIES[:3]:
    job(f"time_ext_YELP_{c.replace(' ', '_')}", "d8/time_ext_yelp.py", ["{YELP_DIR}", "{RUNS}/yelp_prep/slim", c],
        inputs=YJ[:2] + PSO + CQS, after=["yelp_prep"], exclusive=True, group="timing", deps=EXTD)

# ---- redesigned Section VI: accuracy (B) and running time (C) against CQ-HUCPM and PSO-MUCP ---------------------------
# Each subsection changes one parameter on two or three datasets, and the datasets differ between subsections.
# Defaults K = 100, d = 50 m, k = 20, all instances, c = 1, lognormal truth with the fitted spread (sd = 1).
EXT = ["ext/run_uhucpm.py", "ext/run_cq.py", "ext/run_pso.py"]
YENV = {"YELP_DIR": "{YELP_DIR}", "REVIEW_DIR": "{RUNS}/yelp_prep/slim"}


def _src(L):
    return ([FSQ[L]], {}, ()) if L in ("NYC", "TKY") else (YJ[:2], YENV, ("yelp_prep",))


ACC = [("K", "100,300,1000", ("Philadelphia", "Tampa", "NYC")),
       ("d", "30,50,70", ("New_Orleans", "NYC")),
       ("k", {"Tampa": "8,12,16,20", "NYC": "10,20,30,40"}, ("Tampa", "NYC")),
       ("frac", "0.25,0.5,0.75,1", ("Philadelphia", "NYC")),
       ("c", "0.5,1,2,4", ("New_Orleans", "Tampa")),
       ("sd", "0.5,1,1.5,2", ("Philadelphia", "NYC"))]
for vary, values, layouts in ACC:
    for L in layouts:
        inp, env, aft = _src(L)
        v = values[L] if isinstance(values, dict) else values
        if L == "Philadelphia": env = {**env, "NCHK": "0"}   # CQ-HUCPM does not finish on D1: no run of the authors' code
        job(f"acc_{L}_{vary}", "ext/acc.py", [L, vary, v, SEMI_SEEDS], env, inp + PSO + CQS, after=aft, group="acc",
            subdirs=["out_acc"], deps=EXT)

BENCH = {}                                                  # (layout, k, d, frac) -> thresholds K
for L in ("Tampa", "New_Orleans", "NYC"):                   # C1: threshold (D1 left out: CQ-HUCPM does not finish)
    BENCH.setdefault((L, 20, 50, 1.0), set()).update((100, 300, 1000))
for L, ks in (("Tampa", (8, 12, 16, 20)), ("NYC", (20, 40, 60, 80))):    # C2: number of features
    for k in ks: BENCH.setdefault((L, k, 50, 1.0), set()).add(100)
for L in ("Tampa", "NYC"):                                  # C3: distance
    for d in (30, 50, 70): BENCH.setdefault((L, 20, d, 1.0), set()).add(100)
for L in ("New_Orleans", "TKY"):                            # C4: share of the instances
    for f in (0.25, 0.5, 0.75, 1.0): BENCH.setdefault((L, 20, 50, f), set()).add(100)
for (L, k, d, f), Ks in sorted(BENCH.items()):
    inp, env, aft = _src(L)
    job(f"bench_{L}_k{k}_d{d}_f{int(round(100 * f))}", "ext/bench.py", [L, str(k), str(d), str(f)] + [str(K) for K in sorted(Ks)],
        env, inp + PSO + CQS, after=aft, exclusive=True, group="bench", subdirs=["out_bench"], deps=EXT)

# the three laws of the true log-rates at K = 100, 300, 1000 (main table of VI-B), hold rates of the three methods on
# the 13 real datasets, and the rating simulator with the three methods
for L in ("Philadelphia", "NYC"):
    inp, env, aft = _src(L)
    job(f"acc_{L}_law", "ext/acc.py", [L, "law", "lognormal,gamma,mixture", SEMI_SEEDS],
        {**env, "KS": "100,300,1000", **({"NCHK": "0"} if L == "Philadelphia" else {})},
        inp + PSO + CQS, after=aft, group="acc", subdirs=["out_acc"], deps=EXT)
for L in ["NYC", "TKY"] + [c.replace(" ", "_") for c in YCITIES]:
    inp, env, aft = _src(L)
    job(f"hold_{L}", "ext/hold.py", [L], env, inp + PSO, after=aft, group="hold", subdirs=["out_hold"], deps=EXT)
job("rsim", "ext/rsim.py", ["100"], inputs=PSO + CQS, group="rsim", subdirs=["out_rsim"], deps=EXT)

# ---- diagnostics for the weak results of VI-B (root causes) ------------------------------------------------------------
# diag_acc: calibration of V, detection by true margin, FDP-detection curves of the lower-bound and point rules, the
# posterior expected FDP rule, and PSO-MUCP ranked by its own score, on the default accuracy setting (K = 100)
# hold_curve: hold rates of the three methods at matched answer sizes on the real datasets
for L in ("Philadelphia", "New_Orleans", "Tampa"):
    inp, env, aft = _src(L)
    job(f"diag_{L}", "ext/diag_acc.py", [L, SEMI_SEEDS], env, inp + PSO, after=aft, group="diag", subdirs=["out_diag"], deps=EXT)
for L in [c.replace(" ", "_") for c in YCITIES]:
    inp, env, aft = _src(L)
    job(f"holdcurve_{L}", "ext/hold_curve.py", [L], env, inp + PSO, after=aft, group="diag", subdirs=["out_hold"], deps=EXT)
for L in ("NYC", "TKY"):                                    # Foursquare: diagnostics and matched-size hold rates
    inp, env, aft = _src(L)
    job(f"diag_{L}", "ext/diag_acc.py", [L, SEMI_SEEDS], env, inp + PSO, after=aft, group="diag", subdirs=["out_diag"], deps=EXT)
    job(f"holdcurve_{L}", "ext/hold_curve.py", [L], env, inp + PSO, after=aft, group="diag", subdirs=["out_hold"], deps=EXT)
# New York at k = 40 alone (the 12 h session stopped acc_NYC_k after two seeds); CQ-HUCPM does not finish there, so the
# authors' code is not run
inp, env, aft = _src("NYC")
job("acc_NYC_k40", "ext/acc.py", ["NYC", "k", "40", SEMI_SEEDS], {**env, "NCHK": "0"}, inp + PSO + CQS, after=aft, group="acc",
    subdirs=["out_acc"], deps=EXT)

# ---- scalability: synthetic check-in data (ext/syngen.py) from 20,000 to 120,000 instances, k = 20, d = 50, K = 100;
# the area grows with n so that the mean number of neighbours stays the same; each run of the authors' code is capped
# at one hour and a program that did not finish is reported as not finished
for nsyn in (20000, 40000, 60000, 80000, 100000, 120000):
    job(f"bench_syn_n{nsyn}", "ext/bench.py", ["syn", str(nsyn), "100"], {"CAP": "3600"}, PSO + CQS, exclusive=True,
        group="bench", subdirs=["out_bench"], deps=EXT + ["ext/syngen.py"])
