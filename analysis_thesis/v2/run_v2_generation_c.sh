#!/usr/bin/env bash
# Parallel v2 generation: part_01 is already running (PID $1); two more simulator instances generate parts 2/4 and 3/5.
set -uo pipefail
SIM=/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project
LOG=/home/jude/isaac_ws/safety_probing/run_logs/forksafe_v2_generation.log
gen() {
  cd "$SIM"
  echo "[v2] $(date +%F\ %T) start $1" >> $LOG
  ../../../env_isaaclab/bin/python -m forklift_dataset.generator --config forklift_dataset/forksafe_v2/$1.json --headless > run_logs/$1.log 2>&1
  echo "[v2] $(date +%F\ %T) done $1 exit=$? pairs=$(ls dataset_root/$1/pairs 2>/dev/null | wc -l)" >> $LOG
}
( while kill -0 "$1" 2>/dev/null; do sleep 20; done; echo "[v2] $(date +%F\ %T) done forksafe_v2_part_01 pairs=$(ls $SIM/dataset_root/forksafe_v2_part_01/pairs 2>/dev/null | wc -l)" >> $LOG ) &
( gen forksafe_v2_part_02; gen forksafe_v2_part_04 ) &
( sleep 45; gen forksafe_v2_part_03; gen forksafe_v2_part_05 ) &
wait
echo "[v2] ALL DONE" >> $LOG
