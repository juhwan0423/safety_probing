#!/usr/bin/env bash
# Continuation of the v2 generation chain: wait for the running part_01 generator, then parts 2-5 (single camera).
# The paused LTX train rollouts are resumed separately, when no other GPU job is running.
set -uo pipefail
SIM=/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
echo "[v2] $(date +%F\ %T) done forksafe_v2_part_01 exit=? pairs=$(ls $SIM/dataset_root/forksafe_v2_part_01/pairs 2>/dev/null | wc -l)"
gen() {
  cd "$SIM"
  echo "[v2] $(date +%F\ %T) start $1"
  ../../../env_isaaclab/bin/python -m forklift_dataset.generator --config forklift_dataset/forksafe_v2/$1.json --headless > run_logs/$1.log 2>&1
  echo "[v2] $(date +%F\ %T) done $1 exit=$? pairs=$(ls dataset_root/$1/pairs 2>/dev/null | wc -l)"
}
gen forksafe_v2_part_02
gen forksafe_v2_part_03
gen forksafe_v2_part_04
gen forksafe_v2_part_05
echo "[v2] ALL DONE"
