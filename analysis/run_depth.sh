#!/bin/bash
cd /data/ammelsayed/fight_overtraining
source ~/root/bin/thisroot.sh
for spec in "4L_eeee 1" "4L_eeee 2" "3L_eeeJ 1" "3L_eeeJ 2"; do
  set -- $spec
  /data/ammelsayed/venv/bin/python depth_test.py --channel "$1" --half "$2" \
      --out "/tmp/depth_$1_$2.json" > "/tmp/depth_$1_$2.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
echo DEPTH_DONE
