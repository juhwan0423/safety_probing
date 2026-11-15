#!/usr/bin/env bash
# Boundary-sweep parts of ForkSafe-CF v2 (regular grid over payload mass / lateral CoM offset and drive command); one simulator lane.
set -uo pipefail
SIM=/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project
LOG=/home/jude/isaac_ws/safety_probing/run_logs/forksafe_v2_generation.log
gen() {
  cd "$SIM"
  echo "[v2] $(date +%F\ %T) start $1" >> $LOG
  ../../../env_isaaclab/bin/python -m forklift_dataset.generator --config forklift_dataset/forksafe_v2/$1.json --headless > run_logs/$1.log 2>&1
  echo "[v2] $(date +%F\ %T) done $1 exit=$? pairs=$(ls dataset_root/$1/pairs 2>/dev/null | wc -l)" >> $LOG
}
gen forksafe_v2_sweep_01; gen forksafe_v2_sweep_02
echo "[v2] $(date +%F\ %T) SWEEP COMPLETE" >> $LOG
