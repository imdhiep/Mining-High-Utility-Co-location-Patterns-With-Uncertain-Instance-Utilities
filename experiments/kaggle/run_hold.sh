#!/bin/bash
# Hold rates on real data (Section VI-D.1) on a Kaggle CPU notebook with Internet on and the same two inputs as
# run_accuracy.sh. Usage (from the repository root): bash experiments/kaggle/run_hold.sh
#   -> /kaggle/working/hold_raw.tar.gz
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EXP=$ROOT/experiments; WORK=/kaggle/working
nproc; grep -m1 'model name' /proc/cpuinfo; free -g | head -2
make -C "$ROOT/cpp" -j4 all bin/unit_tests >/dev/null && "$ROOT/cpp/bin/unit_tests" | tail -1
sh "$ROOT/cpp/cq/build_cq.sh" "$ROOT/cpp/cq_build" | tail -1
YELP=$(dirname "$(find /kaggle/input -name yelp_academic_dataset_business.json | head -1)")
python3 "$EXP/scripts/fsq_from_csv.py" /kaggle/input "$WORK/tsmc2014"
python3 "$EXP/scripts/prepare_layouts.py" --fsq-dir "$WORK/tsmc2014" --ks 20 > "$WORK/prepare.log"
python3 "$EXP/scripts/check_layouts.py" "$EXP/data/layouts" "$EXP/kaggle/layout_reference.json" | grep -v OK || true
python3 "$EXP/scripts/prepare_hold.py" --yelp-dir "$YELP" \
    --cities "Philadelphia|New Orleans|Tampa|Tucson|Indianapolis|Nashville|Reno|Saint Louis|Edmonton|Santa Barbara|Boise"
cd "$EXP" && python3 scripts/run_hold.py configs/hold.json
tar czf "$WORK/hold_raw.tar.gz" -C "$EXP" results/raw
ls -la "$WORK"
