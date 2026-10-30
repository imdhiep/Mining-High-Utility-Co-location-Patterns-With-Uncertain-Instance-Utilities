# Mining High-Utility Co-location Patterns With Uncertain Instance Utilities

Source code of **UHUCM**, a sound and complete miner of high-utility co-location patterns whose instance utilities are
posterior distributions (mean and variance) estimated from user-generated evidence, together with the **EviGen**
synthetic generator and the scripts that produce every experiment of the paper.

A pattern `P` is reported when its lower-bound score `(M(P) - z*sqrt(V(P))) / Z` reaches the threshold, where `M` and `V`
are the summed posterior means and variances of the participating atoms (instances and neighboring pairs).

## Layout

| File | Content |
|---|---|
| `uhucm/uhucm.py` | Reference implementation and brute force (definitions, bounds, exact miner) |
| `uhucm/fastmine.py` | Vectorized UHUCM used in all experiments: prefix-optimal exact bound, chord bound, FUB, PCB, suffix bound, robust (equicorrelated) bound |
| `uhucm/test_fastmine.py` | `fastmine` vs. brute force on random instances (mixed signs, variances, interactions, rho) |
| `uhucm/evigen.py`, `uhucm/synth.py` | EviGen generator; E1 exactness, E2 calibration (incl. misspecification), scalability, clique stress test |
| `uhucm/fsq_loader.py` | Foursquare TSMC2014 -> atoms (Gamma-Poisson relative-performance utility) and interaction pairs |
| `uhucm/yelp_full.py` | Yelp Open Dataset -> atoms (Normal-Normal rating posterior) and interaction pairs |
| `uhucm/suite.py` | Experiment suite shared by all real datasets (quality, hold-out, filters, sensitivity, dependence, efficiency) |
| `uhucm/run_suite_fsq.py`, `uhucm/run_fsq_d.py` | Suite on Foursquare (D4 New York, D5 Tokyo) at d = 50 m and at other distances |
| `uhucm/kexp2.py`, `uhucm/kexp4.py` | Kaggle drivers for Yelp (D1 Philadelphia, D2 New Orleans, D3 Tampa) |
| `uhucm/ext_baselines.py`, `uhucm/run_ext_fsq.py`, `uhucm/cq_rerun.py` | External baselines with the authors' PSO-MUCP and CQ-HUCPM code |
| `uhucm/d8/` | Comparison with the authors' PSO-MUCP code on their public data (called D6 in the paper) and the Strategy-4 audit |
| `uhucm/rev_extra.py`, `uhucm/scale_mem.py`, `uhucm/scale_mem_fsq.py` | Ablation with all-negative features, interaction sensitivity, scalability and peak memory |
| `uhucm/results_db.py`, `uhucm/stats_rev.py`, `uhucm/stats_tests.py` | Collect results; Wilson intervals, Friedman/Nemenyi, Holm, Wilcoxon |
| `uhucm/make_rev_tables.py`, `uhucm/make_figures.py`, `uhucm/figstyle.py` | Tables and figures, generated only from the result files |
| `uhucm/out_*`, `uhucm/results_kaggle/`, `uhucm/d8/d8_results.jsonl` | Result files behind the reported numbers |

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
python synth.py scale_n && python synth.py scale_k && python synth.py stress
python run_suite_fsq.py $FSQ_DIR/dataset_TSMC2014_NYC.txt NYC out_suite
python run_fsq_d.py    $FSQ_DIR/dataset_TSMC2014_NYC.txt NYC out_suite_d 30
python run_ext_fsq.py NYC                                # external baselines (needs PSO-MUCP)
(cd d8 && python run_d8_all.py cal && python run_d8_all.py lv)
python kexp4.py YELP_ROOT OUT 2019-01-01 2020-01-01 "Philadelphia;New Orleans;Tampa"   # Yelp, on Kaggle
python stats_rev.py && python make_rev_tables.py && python make_figures.py
```

Seeds are fixed in the scripts (EviGen calibration seeds 1000-1099, timing seeds 7-9, D6 utilities seed 2026).
