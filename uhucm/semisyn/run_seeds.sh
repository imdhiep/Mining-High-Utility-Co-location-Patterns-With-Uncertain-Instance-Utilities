#!/bin/bash
# Semi-synthetic calibration, seeds 3-9 (seeds 0-2: OUTD=out_semisyn with SEED0=0 and 3 seeds). Run from uhucm/ with
# FSQ_DIR (Foursquare TSMC2014) and D6_DIR (DATA folder of the PSO-MUCP clone) set.
# Usage: semisyn/run_seeds.sh NYC TKY LasVegas
export OMP_NUM_THREADS=1 FR=1.0 KS=100,300,1000 SEED0=${SEED0:-3}
OUT=${OUT:-out_semisyn_s3_9}; mkdir -p $OUT
export OUTD=${OUTD:-json}; mkdir -p semisyn/$OUTD     # per-run JSON copies
for C in "$@"; do for T in lognormal gamma mixture; do
  python3 semisyn/semisyn_fsq.py $C 10 $T np zpost cboot diag momc > $OUT/log_${C}_${T}_np_cboot.txt 2>&1
  python3 semisyn/semisyn_fsq.py $C 10 $T oracle zpost diag momc > $OUT/log_${C}_${T}_oracle.txt 2>&1
  python3 semisyn/semisyn_fsq.py $C 10 $T gamma diag > $OUT/own_${C}_${T}_gamma.txt 2>&1
done; done
