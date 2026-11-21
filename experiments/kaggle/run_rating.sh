#!/bin/bash
# Rating scenario of Section VI-B.2 (semi-synthetic star ratings on the Tampa layout, rating posterior (4)) on a Kaggle
# CPU notebook with the same two inputs as run_accuracy.sh. Usage: bash experiments/kaggle/run_rating.sh
#   -> /kaggle/working/rating_raw.tar.gz
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EXP=$ROOT/experiments; WORK=/kaggle/working
nproc; grep -m1 'model name' /proc/cpuinfo; free -g | head -2
make -C "$ROOT/cpp" -j4 all bin/unit_tests >/dev/null && "$ROOT/cpp/bin/unit_tests" | tail -1
sh "$ROOT/cpp/cq/build_cq.sh" "$ROOT/cpp/cq_build" 2>/dev/null | tail -1
YELP=$(dirname "$(find /kaggle/input -name yelp_academic_dataset_business.json | head -1)")
python3 "$EXP/scripts/fsq_from_csv.py" /kaggle/input "$WORK/tsmc2014"
python3 "$EXP/scripts/prepare_layouts.py" --fsq-dir "$WORK/tsmc2014" --yelp-dir "$YELP" --ks 20 \
    --yelp-cities "Philadelphia|Tampa|New Orleans" > "$WORK/prepare.log"
cd "$EXP"
python3 scripts/run_accuracy.py configs/accuracy.json B2_rating && echo "finished B2_rating" || echo "failed B2_rating"
tar czf "$WORK/rating_raw.tar.gz" -C "$EXP" results/raw
ls -la "$WORK"
