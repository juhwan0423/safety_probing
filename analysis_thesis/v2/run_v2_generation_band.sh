#!/usr/bin/env bash
# Boundary-band parts of ForkSafe-CF v2 (drive command 18-22, where the payload decides the outcome).
# Two simulator lanes; the second starts when parts 4 and 5 of the main set have finished (GPU memory is shared with training).
set -uo pipefail
SIM=/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project
LOG=/home/jude/isaac_ws/safety_probing/run_logs/forksafe_v2_generation.log
gen() {
  cd "$SIM"
  echo "[v2] $(date +%F\ %T) start $1" >> $LOG
  ../../../env_isaaclab/bin/python -m forklift_dataset.generator --config forklift_dataset/forksafe_v2/$1.json --headless > run_logs/$1.log 2>&1
  echo "[v2] $(date +%F\ %T) done $1 exit=$? pairs=$(ls dataset_root/$1/pairs 2>/dev/null | wc -l)" >> $LOG
}
after() { until grep -q "done $1" $LOG 2>/dev/null; do sleep 30; done; }
( gen forksafe_v2_band_01; gen forksafe_v2_band_03; gen forksafe_v2_band_05 ) &
( after forksafe_v2_part_04; after forksafe_v2_part_05; gen forksafe_v2_band_02; gen forksafe_v2_band_04 ) &
wait
echo "[v2] $(date +%F\ %T) BAND COMPLETE" >> $LOG
