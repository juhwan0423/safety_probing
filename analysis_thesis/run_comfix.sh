#!/usr/bin/env bash
# Corrected-label pipeline: targets -> V-JEPA PSQ -> LTX PSQ, for two label variants. New files only.
set -euo pipefail
cd /home/jude/isaac_ws/safety_probing
PY=.venv/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
for V in comfix comfix_med5; do
  if [ "$V" = comfix_med5 ]; then export COMFIX_MED5=1; else export COMFIX_MED5=0; fi
  $PY analysis_thesis/comfix_build_targets.py --experiment configs/experiments/vjepa_safety_query.json --shard full --output artifacts/targets/safety_query_full_$V.pt
  $PY analysis_thesis/comfix_build_targets.py --experiment configs/experiments/ltx_safety_query.json --shard full --output artifacts/targets/ltx_safety_query_full_$V.pt
  $PY -m safety_probing.safety_query_probe --feature-cache artifacts/features/vjepa_2_1_full_observation_spatial12.pt \
      --target-cache artifacts/targets/safety_query_full_$V.pt --output-root outputs/safety_query/full_$V
  $PY -m safety_probing.ltx_safety_query --real-cache artifacts/features/ltx_2_3_full_real_future.pt \
      --rollout-cache artifacts/features/ltx_2_3_full_rollout_future.pt \
      --target-cache artifacts/targets/ltx_safety_query_full_$V.pt --output-root outputs/ltx_safety_query/full_prompt_v1_$V
  echo "[comfix] variant $V done"
done
echo "[comfix] ALL DONE"
