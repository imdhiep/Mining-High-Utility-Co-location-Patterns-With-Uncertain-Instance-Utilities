#!/bin/bash
# Running-time experiments (Section VI-C) on a Kaggle CPU notebook with Internet on and the same two inputs as
# run_accuracy.sh. Programs run one at a time; the notebook must run nothing else.
# Usage (from the repository root): bash experiments/kaggle/run_timing.sh [--counts] EXPERIMENT[:ARGS] ...
#   ARGS: further options of run_timing.py with commas for spaces, e.g. C4_las_vegas:--datasets,Las_Vegas_checkins
#   -> /kaggle/working/timing_raw.tar.gz and /kaggle/working/logs (the raw files are also kept under the repository
#   copy in /kaggle/working, so a session that reaches the time limit still keeps the points already written)
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EXP=$ROOT/experiments; WORK=/kaggle/working
nproc; grep -m1 'model name' /proc/cpuinfo; free -g | head -2
make -C "$ROOT/cpp" -j4 all bin/unit_tests >/dev/null && "$ROOT/cpp/bin/unit_tests" | tail -1
sh "$ROOT/cpp/cq/build_cq.sh" "$ROOT/cpp/cq_build" | tail -1
[ -d "$EXP/data/psomucp" ] || git clone -q --depth 1 https://github.com/yuanshuaikang/PSO-MUCP "$EXP/data/psomucp"
YELP=$(dirname "$(find /kaggle/input -name yelp_academic_dataset_business.json | head -1)")
python3 "$EXP/scripts/fsq_from_csv.py" /kaggle/input "$WORK/tsmc2014"
python3 "$EXP/scripts/prepare_layouts.py" --fsq-dir "$WORK/tsmc2014" --yelp-dir "$YELP" --ks 10,20,40,60,80 \
    --yelp-cities "Philadelphia|Tampa|New Orleans" > "$WORK/prepare.log"
mkdir -p "$WORK/logs"; cd "$EXP"
MODE=""; if [ "$1" = "--counts" ]; then MODE="--counts"; shift; fi   # --counts: untimed candidate counts instead of timing
for job in "$@"; do
  e=${job%%:*}; ARGS=""; [ "$job" != "$e" ] && ARGS=$(echo "${job#*:}" | tr ',' ' ')
  LOG="$WORK/logs/$(echo "$job" | tr ':,' '__')$MODE.log"
  python3 scripts/run_timing.py configs/timing.json "$e" $MODE $ARGS > "$LOG" 2>&1 && echo "finished $job" || { echo "failed $job"; tail -5 "$LOG"; }
done
tar czf "$WORK/timing_raw.tar.gz" -C "$EXP" results/raw
ls -la "$WORK"
