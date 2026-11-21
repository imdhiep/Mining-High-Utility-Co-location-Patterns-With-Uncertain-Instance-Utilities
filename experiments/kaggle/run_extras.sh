#!/bin/bash
# Untimed runs of Section VI on a Kaggle CPU notebook (Internet on, the same two inputs as run_accuracy.sh):
#   candidate counts of every program (pruning ratio) at the points of C.1 and C.4, counts on the real datasets (local
#   rules, answer sizes by K and delta), the case study of D.2. Usage: bash experiments/kaggle/run_extras.sh [--no-c4]
#   -> /kaggle/working/extras_raw.tar.gz
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EXP=$ROOT/experiments; WORK=/kaggle/working
nproc; grep -m1 'model name' /proc/cpuinfo; free -g | head -2
make -C "$ROOT/cpp" -j4 all bin/unit_tests >/dev/null && "$ROOT/cpp/bin/unit_tests" | tail -1
sh "$ROOT/cpp/cq/build_cq.sh" "$ROOT/cpp/cq_build" 2>/dev/null | tail -1
python3 "$ROOT/cpp/tests/verify_cq.py" "$ROOT/cpp/cq_build/total_fix/cq_hucpm" 50 | tail -1
git clone -q --depth 1 https://github.com/yuanshuaikang/PSO-MUCP "$EXP/data/psomucp"
YELP=$(dirname "$(find /kaggle/input -name yelp_academic_dataset_business.json | head -1)")
python3 "$EXP/scripts/fsq_from_csv.py" /kaggle/input "$WORK/tsmc2014"
python3 "$EXP/scripts/prepare_layouts.py" --fsq-dir "$WORK/tsmc2014" --yelp-dir "$YELP" --ks 10,20,40,60,80 \
    --yelp-cities "Philadelphia|Tampa|New Orleans" > "$WORK/prepare.log"
python3 "$EXP/scripts/check_layouts.py" "$EXP/data/layouts" "$EXP/kaggle/layout_reference.json" | grep -v OK || true
cd "$EXP"
python3 scripts/run_counts_real.py configs/timing.json && echo "finished counts on real data" || echo "failed counts on real data"
python3 scripts/run_case.py configs/hold.json && echo "finished case study" || echo "failed case study"
COUNTS="C1_features C4_las_vegas C4_toronto C4_california"
[ "$1" = "--no-c4" ] && COUNTS="C1_features"   # the C.4 counts are slow at large d: run them elsewhere
for e in $COUNTS; do
  python3 scripts/run_timing.py configs/timing.json "$e" --counts > "$WORK/counts_$e.log" 2>&1 && echo "finished counts $e" || { echo "failed counts $e"; tail -5 "$WORK/counts_$e.log"; }
done
tar czf "$WORK/extras_raw.tar.gz" -C "$EXP" results/raw
ls -la "$WORK"
