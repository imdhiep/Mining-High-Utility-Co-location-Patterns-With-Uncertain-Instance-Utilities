# C++ implementation

## Build

```
make            # programs in bin/ (g++ >= 9, C++17, -O2); no dependency beyond the standard library
make test       # unit tests (bin/unit_tests)
```

or with CMake: `cmake -S . -B build && cmake --build build -j` (same programs in `bin/`, `ctest --test-dir build`).

## Layout

Each algorithm is one class; the programs in `apps/` only read arguments, call the classes and print one JSON line.

| Module | Classes | Role |
|---|---|---|
| `src/common` | `CsvTable`, `CsvWriter`, `JsonLineParser`, `JsonLineWriter`, `CommandLine`, `UtilityPosteriorFile`, `Stopwatch` | files, arguments, timing, peak memory |
| `src/geo` | `UtmProjection` | WGS84 -> UTM (Krueger series, matches pyproj to 1e-6 m) |
| `src/spatial` | `SpatialDataset`, `NeighborGraph`, `MaximalCliqueEnumerator`, `CliqueTable` | instances, neighbour relation, clique table (Section V-B) |
| `src/mining` | `InstanceUtilities`, `SubtreeBound`, `UhucpmMiner` | UHUCPM score, subtree bounds (Lemma 1), exact miner (Algorithms 1 and 2) |
| `src/posterior` | `PosteriorModel` (interface), `CountPosteriorModel`, `RatingPosteriorModel`, `Npmle`, `GridPosterior`, `RateCenter` | count utility (Definition 4) and rating utility (equation (4)) |
| `src/baselines` | `PsoMucpMiner` | port of the authors' PSO-MUCP.py |
| `src/data` | `FoursquareLoader`, `YelpLoader`, `FeatureLabels` | Foursquare TSMC2014 and Yelp files -> instances |

## Programs

| Program | Usage (header of `apps/<name>.cpp` for all options) |
|---|---|
| `uhucpm_mine` | `uhucpm_mine DATA_CSV POSTERIOR GAMMA D Z [--bound two\|exact\|mean\|none] [--rule global\|sum] [--deadline S] [--scores]` |
| `pso_mucp` | `pso_mucp DATA_CSV UTILITY MIN_UTILITY D [--norm NORM]` |
| `post_counts` | `post_counts DATA_CSV COUNT_COLUMN E OUT_POSTERIOR [--t T] [--B B] [--seed S] [--ep FILE]` |
| `fsq_prep` | `fsq_prep TSV UTM_ZONE NFEAT OUT_CSV` |
| `yelp_prep` | `yelp_prep BUSINESS CHECKIN REVIEW_DIR "City\|City" SPLIT TEST_END OUT_DIR [--nfeat K] [--B B] [--seed S]` |

`post_counts` reports `local_edge_mass_max`: the largest posterior mass on the edges of a local grid (Definition 4 uses
a local grid centred on the likelihood for large counts). It must stay negligible (below 1e-6); on New York and Tokyo
it is below 1e-7.

`cq/build_cq.sh` builds the authors' CQ-HUCPM (C++) on Linux: `original` (their mining code) and `total_fix`
(total-utility score and the missing 2-cliques added; see the script header).

## Checks

- `bin/unit_tests` (33 tests): every module against an independent reference.
  - Neighbour graph against all pairs; maximal cliques are cliques, maximal and cover every edge; participating
    instances against row instances found by brute force.
  - Subtree bounds against the best subset by enumeration; the miner against brute-force enumeration of all patterns
    for every bound test, z in {0, 1.645, 3} and four thresholds; edge cases (empty result, non-positive means, no
    neighbours, one feature).
  - Count posterior against the Gamma-Poisson closed form, on the grid and on local grids; moment center against the
    location of log-normal rates; rating posterior against a hand computation; E[u+] against numerical integration.
  - PSO-MUCP port against brute force with positive utilities (its pruning is safe there); with negative utilities it
    never reports a pattern below the threshold.
  - UTM against pyproj reference points.
- `tests/verify_cq.py CQ_BINARY`: the authors' CQ-HUCPM in total-utility mode against brute force. The fixed build
  agrees on every pattern decision (6,166 on 200 random datasets); the original build misses pairs.
- `tests/verify_*.py` (others): the programs against the earlier Python implementation (`uhucm/`) and the authors'
  PSO-MUCP code.

Random streams (bootstrap of V_T) differ from numpy, so V_T agrees with the Python code in distribution only.
