#!/bin/bash
cd /data/ammelsayed/fight_overtraining
source ~/root/bin/thisroot.sh
for m in flat_bagging flat_smote; do
  for ch in 3L_eee 3L_eeeJ 4L_eeee; do
    for h in 1 2; do
      /data/ammelsayed/venv/bin/python study_one.py --mode "$m" --channel "$ch" --half "$h" \
        --out "results_json/${m}__${ch}__DF${h}.json" \
        > "results_json/${m}__${ch}__DF${h}.log" 2>&1 &
      while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
    done
  done
done
wait
echo FLAT_DONE
