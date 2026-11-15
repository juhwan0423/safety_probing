#!/usr/bin/env bash
# ForkSafe-CF v2 generation: parts 1-3, then resume the paused LTX train rollouts, then parts 4-5.
set -uo pipefail
SIM=/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project
SP=/home/jude/isaac_ws/safety_probing
gen() {
  cd "$SIM"
  echo "[v2] $(date +%F\ %T) start $1"
  ../../../env_isaaclab/bin/python -m forklift_dataset.generator --config forklift_dataset/forksafe_v2/$1.json --headless > run_logs/$1.log 2>&1
  echo "[v2] $(date +%F\ %T) done $1 exit=$? pairs=$(ls dataset_root/$1/pairs 2>/dev/null | wc -l)"
}
gen forksafe_v2_part_01
gen forksafe_v2_part_02
gen forksafe_v2_part_03
cd "$SP"
echo "[v2] $(date +%F\ %T) resuming LTX train rollouts"
PYTHONUNBUFFERED=1 PYTHONPATH=src external/LTX-2/.venv/bin/python -m safety_probing.ltx_batch --experiment configs/experiments/ltx_2_3_continuation_prompt_v1.json \
  --shard full --split train --seed 17 --output-root outputs/ltx_v2/full_prompt_v1_train > run_logs/ltx_full_train_seed17_run3.log 2>&1
echo "[v2] $(date +%F\ %T) LTX rollouts exit=$? $(grep -A3 status_counts outputs/ltx_v2/full_prompt_v1_train/batch_summary.json | tr -d '\n ')"
gen forksafe_v2_part_04
gen forksafe_v2_part_05
echo "[v2] ALL DONE"
