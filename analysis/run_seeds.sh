#!/bin/bash
cd /data/ammelsayed/fight_overtraining
source ~/root/bin/thisroot.sh
# 4L_eeee both directions (the crux), and 3L_eeeJ as a cross-check
for spec in "4L_eeee 1" "4L_eeee 2" "3L_eeeJ 1" "3L_eeeJ 2"; do
  set -- $spec
  /data/ammelsayed/venv/bin/python seeds.py --channel "$1" --half "$2" \
      --out "/tmp/seeds_$1_$2.json" > "/tmp/seeds_$1_$2.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo SEEDS_DONE
