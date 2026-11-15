#!/usr/bin/env bash
# Paper-split table (observed video / real-future oracle / generated future) on the true-CoM targets.
set -uo pipefail
cd /home/jude/isaac_ws/safety_probing
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/python -m safety_probing.safety_query_probe --feature-cache artifacts/features/vjepa_2_1_full_observation_spatial12.pt \
    --target-cache artifacts/targets/thesis_v1_obs65_truecom.pt --output-root outputs/safety_query/full_truecom
.venv/bin/python -m safety_probing.ltx_safety_query --real-cache artifacts/features/ltx_2_3_full_real_future.pt \
    --rollout-cache artifacts/features/ltx_2_3_full_rollout_future.pt \
    --target-cache artifacts/targets/ltx_safety_query_full_truecom.pt --output-root outputs/ltx_safety_query/full_prompt_v1_truecom
echo "[paper-split] done"
