#!/bin/bash
# Accuracy experiments (Section VI-B) on a Kaggle CPU notebook with Internet on and two inputs:
#   the Yelp Open Dataset (yelp-dataset/yelp-dataset) and Foursquare TSMC2014 (chetanism/foursquare-nyc-and-tokyo-checkin-dataset).
# Usage (from the repository root): bash experiments/kaggle/run_accuracy.sh [JOBS]
#   -> /kaggle/working/accuracy_raw.tar.gz (results/raw of every experiment) and the logs in /kaggle/working/logs
set -e
JOBS=${1:-4}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EXP=$ROOT/experiments; WORK=/kaggle/working
nproc; grep -m1 'model name' /proc/cpuinfo; free -g | head -2
make -C "$ROOT/cpp" -j"$JOBS" all bin/unit_tests >/dev/null && "$ROOT/cpp/bin/unit_tests" | tail -1
sh "$ROOT/cpp/cq/build_cq.sh" "$ROOT/cpp/cq_build" | tail -1
python3 "$ROOT/cpp/tests/verify_cq.py" "$ROOT/cpp/cq_build/total_fix/cq_hucpm" 50 | tail -1
YELP=$(dirname "$(find /kaggle/input -name yelp_academic_dataset_business.json | head -1)")
python3 "$EXP/scripts/fsq_from_csv.py" /kaggle/input "$WORK/tsmc2014"
python3 "$EXP/scripts/prepare_layouts.py" --fsq-dir "$WORK/tsmc2014" --yelp-dir "$YELP" --ks 10,20,30,40 \
    --yelp-cities "Philadelphia|Tampa|New Orleans" > "$WORK/prepare.log"
python3 "$EXP/scripts/check_layouts.py" "$EXP/data/layouts" "$EXP/kaggle/layout_reference.json" || { echo "Foursquare layouts differ from the original files"; exit 1; }
mkdir -p "$WORK/logs"
# one job per (experiment, layout), longest first
python3 - "$EXP/configs/accuracy.json" > "$WORK/jobs.txt" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
for name, e in cfg["experiments"].items():
    for layout in e["layouts"]:
        print(name, layout)
PY
cd "$EXP"
xargs -P "$JOBS" -L 1 bash -c 'python3 scripts/run_accuracy.py configs/accuracy.json "$0" --layouts "$1" --skip-done > /kaggle/working/logs/"$0"_"$1".log 2>&1; echo "finished $0 $1 ($?)"' < "$WORK/jobs.txt"
tar czf "$WORK/accuracy_raw.tar.gz" -C "$EXP" results/raw
ls -la "$WORK"
