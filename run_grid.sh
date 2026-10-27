#!/bin/bash
# Run the study grid in parallel: one process per (mode, channel, half).
# Each process writes its own JSON row; a merge step combines them.
set -u
cd /data/ammelsayed/fight_overtraining
source ~/root/bin/thisroot.sh
export OMP_NUM_THREADS=4

OUT=results_json
mkdir -p "$OUT"

MODES=${MODES:-"baseline mcw reg bagging smote reg_bagging reg_smote reg_flatten reg_clip"}
CHANNELS=${CHANNELS:-"3L_eee 3L_eeeJ 4L_eeee"}
HALVES=${HALVES:-"1 2"}
MAXJOBS=${MAXJOBS:-6}

n=0
for mode in $MODES; do
  for ch in $CHANNELS; do
    for h in $HALVES; do
      f="$OUT/${mode}__${ch}__DF${h}.json"
      [ -f "$f" ] && continue
      n=$((n+1))
      /data/ammelsayed/venv/bin/python study_one.py --mode "$mode" --channel "$ch" --half "$h" \
          --out "$f" > "$OUT/${mode}__${ch}__DF${h}.log" 2>&1 &
      while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 5; done
    done
  done
done
wait
echo "ALL DONE: launched $n jobs"
