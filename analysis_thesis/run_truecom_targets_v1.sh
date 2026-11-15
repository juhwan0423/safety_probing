#!/usr/bin/env bash
# Rebuild the v1 target caches from the true multi-body CoM/ZMP (truecom.py): five observation lengths + the LTX probe targets.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
export PYTHONPATH=src
for F in 65 47 24 36 56; do
  if [ $F = 65 ]; then EXP=configs/experiments/vjepa_safety_query.json; else EXP=configs/experiments/thesis_vjepa_obs$F.json; fi
  .venv/bin/python analysis_thesis/truecom_build_targets.py --experiment $EXP --shard full --output artifacts/targets/thesis_v1_obs${F}_truecom.pt > run_logs/truecom_targets_obs$F.log 2>&1 &
done
.venv/bin/python analysis_thesis/truecom_build_targets.py --experiment configs/experiments/ltx_safety_query.json --shard full --output artifacts/targets/ltx_safety_query_full_truecom.pt > run_logs/truecom_targets_ltx.log 2>&1 &
wait
ls -la --time-style=+%H:%M artifacts/targets | grep truecom.pt
