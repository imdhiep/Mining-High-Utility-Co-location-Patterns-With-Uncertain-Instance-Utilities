# Mining High-Utility Co-location Patterns With Uncertain Instance Utilities

Source code of **UHUCPM** (uncertain high-utility co-location pattern mining; the code keeps the earlier name `uhucm`), a sound and complete miner of high-utility co-location patterns whose instance utilities are
posterior distributions (mean and variance) estimated from user-generated evidence, together with the **EviGen**
synthetic generator and the scripts that produce every experiment of the paper.

A pattern `P` is reported when its lower-bound score `(M(P) - z*sqrt(V(P))) / Z` reaches the threshold, where `M` and `V`
are the summed posterior means and variances of the participating atoms (instances and neighboring pairs).

## Layout

| File | Content |
|---|---|
| `uhucm/uhucm.py` | Reference implementation and brute force (definitions, bounds, exact miner) |
| `uhucm/fastmine.py` | Vectorized UHUCM used in all experiments: prefix-optimal exact bound, chord bound, FUB, PCB, suffix bound located by binary search, robust (equicorrelated) bound, distance-banded non-negative covariance |
| `uhucm/test_fastmine.py`, `uhucm/test_bands.py` | `fastmine` vs. brute force on random instances (mixed signs, variances, interactions, rho; banded covariance) |
| `uhucm/evigen.py`, `uhucm/synth.py` | EviGen generator; E1 exactness, E2 calibration (incl. misspecification), scalability, clique stress test |
| `uhucm/ceb.py` | Compound empirical Bayes posterior for counts: per-feature NPMLE prior (zero/left-truncation allowed), moment center, expected normalizer, conditional (count) bootstrap for the total variance, optimal set selection |
| `uhucm/fsq_loader.py` | Foursquare TSMC2014 -> atoms (`model="ceb"`: compound empirical Bayes; `model="gamma"`: earlier Gamma-Poisson utility) and interaction pairs |
| `uhucm/run_suite_ceb.py`, `uhucm/run_bands_ceb.py`, `uhucm/run_inter_ceb.py`, `uhucm/run_ext_ceb.py`, `uhucm/run_timing_ceb.py` | D4/D5 experiments with the compound EB posterior and total variance (quality and hold-out, banded covariance, interactions and bootstrap sensitivity, external baselines, repeated timing) |
| `uhucm/real_ceb.py` | Real data: compound EB vs. Gamma-Poisson answer sets, posterior expected FDP, optimal selection |
| `uhucm/semisyn/` | Semi-synthetic count data on the real layouts of D4, D5 and Las Vegas with known utilities (`semisyn_fsq.py`; methods `np` with `cboot`, `oracle`, `gamma`), center studies, unit tests |
| `uhucm/make_semisyn_table.py`, `uhucm/stats_rev3.py`, `uhucm/stats_timing_ceb.py` | Semi-synthetic table, hold-out statistics and timing statistics for the compound EB runs |
| `uhucm/yelp_full.py` | Yelp Open Dataset -> atoms (Normal-Normal rating posterior) and interaction pairs |
| `uhucm/suite.py` | Experiment suite shared by all real datasets (quality, hold-out, filters, sensitivity, dependence, efficiency) |
| `uhucm/run_suite_fsq.py`, `uhucm/run_fsq_d.py` | Suite on Foursquare (D4 New York, D5 Tokyo) at d = 50 m and at other distances |
| `uhucm/kexp2.py`, `uhucm/kexp4.py` | Kaggle drivers for Yelp (D1 Philadelphia, D2 New Orleans, D3 Tampa) |
| `uhucm/ext_baselines.py`, `uhucm/run_ext_fsq.py`, `uhucm/cq_rerun.py` | External baselines with the authors' PSO-MUCP and CQ-HUCPM code |
| `uhucm/d8/` | Comparison with the authors' PSO-MUCP code on their public data (called D6 in the paper) and the Strategy-4 audit |
| `uhucm/rev_extra.py`, `uhucm/scale_mem.py`, `uhucm/scale_mem_fsq.py` | Ablation with all-negative features, interaction sensitivity, scalability and peak memory |
| `uhucm/run_timing_k.py`, `uhucm/run_timing_k2.py` | Repeated timing (10 runs) on D4/D5 with k = 20-80 features, ablation, d = 70 m |
| `uhucm/run_bands_fsq.py`, `uhucm/run_mc_fsq.py` | Distance-banded residual correlation and mining (E8); Monte Carlo check of the normal approximation on D4/D5 |
| `uhucm/results_db.py`, `uhucm/stats_rev.py`, `uhucm/stats_rev2.py`, `uhucm/stats_tests.py`, `uhucm/stats_timing2.py` | Collect results; Wilson intervals, score matching, Friedman, Wilcoxon with Holm, Nemenyi, Welch tests on timings |
| `uhucm/make_rev_tables.py`, `uhucm/make_rev3_tables.py`, `uhucm/make_figures.py`, `uhucm/figstyle.py` | Tables and figures, generated only from the result files |
| `uhucm/out_*`, `uhucm/results_kaggle/`, `uhucm/d8/d8_results.jsonl` | Result files behind the reported numbers (`out_semisyn`, `out_suite_ceb`, `out_ceb`, `out_timing_ceb`: compound EB version) |

Yelp runs were executed in a Kaggle notebook; their outputs were copied into `uhucm/results_kaggle/`
(`summary_transcribed.txt`, `k4_digest.txt`), and `parse_digest.py` converts and consistency-checks them.

## Requirements

Python 3.10+ and the packages in `requirements.txt` (`pip install -r requirements.txt`).

## Data

Raw data are not redistributed.

- Foursquare TSMC2014 (NYC and Tokyo check-ins): place `dataset_TSMC2014_NYC.txt` and `dataset_TSMC2014_TKY.txt` in
  `data/dataset_tsmc2014/`, or set `FSQ_DIR`.
- Yelp Open Dataset: pass the folder with `yelp_academic_dataset_*.json` to the Kaggle drivers.
- PSO-MUCP code and public data: clone https://github.com/yuanshuaikang/PSO-MUCP into `external/PSO-MUCP`, or set
  `PSOMUCP_DIR`.

## Usage

All commands run from `uhucm/`.

```bash
python test_fastmine.py                                  # exactness check against brute force
python synth.py e1                                       # E1 on 2,000 random datasets
python synth.py e2 && python synth.py e2b                # E2 calibration on EviGen
python test_bands.py                                     # banded covariance vs. enumeration
python synth.py e2c && python synth.py e2d               # activity + spatial settings, banded covariance
python scale_mem.py && python synth.py stress            # scalability, peak memory, clique stress test
python run_suite_fsq.py $FSQ_DIR/dataset_TSMC2014_NYC.txt NYC out_suite
python run_fsq_d.py    $FSQ_DIR/dataset_TSMC2014_NYC.txt NYC out_suite_d 30
python run_ext_fsq.py NYC                                # external baselines (needs PSO-MUCP)
(cd d8 && python run_d8_all.py cal && python run_d8_all.py lv)
python kexp4.py YELP_ROOT OUT 2019-01-01 2020-01-01 "Philadelphia;New Orleans;Tampa"   # Yelp, on Kaggle
python run_timing_k.py NYC && python run_timing_k2.py TKY 40 60 80   # repeated timing (also run_timing_k.py TKY for k = 20)
python run_bands_fsq.py NYC && python run_mc_fsq.py NYC             # E8 banded covariance, Monte Carlo check
(cd d8 && python rerun_d6_time.py)                                 # D6 timing, three runs
python stats_rev2.py && python stats_timing2.py && python make_rev3_tables.py && python make_figures.py
```

Compound empirical Bayes version (Section IV-A of the paper; D4/D5 and semi-synthetic data). `D6_DIR` is the `DATA`
folder of the PSO-MUCP clone.

```bash
python semisyn/test_ceb.py && python semisyn/test_trunc.py && python semisyn/test_boot_exact.py
FR=1.0 KS=100,300,1000 python semisyn/semisyn_fsq.py NYC 3 lognormal np zpost cboot diag momc   # also TKY, LasVegas; gamma, mixture
FR=1.0 KS=100,300,1000 python semisyn/semisyn_fsq.py NYC 3 lognormal oracle zpost diag momc
FR=1.0 KS=100,300,1000 python semisyn/semisyn_fsq.py NYC 3 lognormal gamma diag                 # parametric reference, own target
python make_semisyn_table.py
python run_suite_ceb.py $FSQ_DIR/dataset_TSMC2014_NYC.txt NYC out_suite_ceb quality
python run_bands_ceb.py NYC && python run_inter_ceb.py NYC && python run_ext_ceb.py NYC && python real_ceb.py NYC
python run_timing_ceb.py NYC && python stats_rev3.py && python stats_timing_ceb.py
```

Seeds are fixed in the scripts (EviGen calibration seeds 1000-1099, E1 seeds 0-1999, timing seeds 7-9, D6 utilities seed 2026, bootstrap/permutation and Monte Carlo seed 0).
Reported runs used Python 3.13 with NumPy 2.1/SciPy 1.16 (Kaggle) and NumPy 2.5/SciPy 1.18 (local).
