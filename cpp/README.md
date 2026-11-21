# C++ implementation

Build: `make` (g++ 13, -O2, C++17). Programs in `bin/`:

| Program | What it does | Checked against |
|---|---|---|
| `fsq_prep` | Foursquare TSMC2014 -> instance table (venues, features, UTM coordinates, training and test counts) | `uhucm/fsq_loader.py`: same venues, features, counts; coordinates within 1e-8 m (New York, Tokyo) |
| `yelp_prep` | Yelp business/check-in/review files -> instance table, rating posteriors (eq. 4), bootstrap matrix of V_T, expected normalizer | `uhucm/yelp_full.py`: posteriors to 1e-12, same Z; bootstrap spread equal (`tests/verify_yelp.py`) |
| `post_counts` | Count posteriors (Definition 4: NPMLE prior, smoothing, local grids, moment center), bootstrap matrix of V_T, expected normalizer | `uhucm/ceb.py`: means and variances to 1e-14, same Z; bootstrap spread equal with B = 200 (`tests/verify_ceb.py`) |
| `uhucpm_mine` | UHUCPM miner (Algorithms 1 and 2) | brute force and `uhucm/fastmine.py`: 1600 random checks, New York (`tests/verify_miner.py`) |
| `pso_mucp` | Port of the authors' PSO-MUCP.py | the authors' code: 120 random datasets, the Strategy 4 counterexample, Las Vegas (`tests/verify_pso.py`) |

`cq/build_cq.sh` builds the authors' CQ-HUCPM (C++) on Linux: `original` (their mining code) and `total_fix`
(total-utility score and the missing 2-cliques added; see the script header).

Random streams (bootstrap of V_T) differ from numpy, so V_T agrees with the Python code in distribution only.
