#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
LTX_PYTHON="${LTX_PYTHON:-$ROOT/external/LTX-2/.venv/bin/python}"
PHASE="${1:-help}"

verify_inputs() {
  "$PYTHON" -m safety_probing.full_shard_preflight --deep --strict
  "$PYTHON" -m safety_probing.protocol_lock verify
}

case "$PHASE" in
  preflight)
    "$PYTHON" -m safety_probing.full_shard_preflight --deep --strict
    ;;
  audit)
    verify_inputs
    "$PYTHON" -m safety_probing.cli all --config configs/local_dataset.json
    ;;
  vjepa)
    verify_inputs
    "$PYTHON" -m safety_probing.vjepa_spatial_features \
      --shard full --batch-size "${BATCH_SIZE:-4}"
    "$PYTHON" -m safety_probing.safety_query_targets --shard full
    "$PYTHON" -m safety_probing.safety_query_probe \
      --feature-cache artifacts/features/vjepa_2_1_full_observation_spatial12.pt \
      --target-cache artifacts/targets/safety_query_full.pt \
      --output-root outputs/safety_query/full
    "$PYTHON" -m safety_probing.safety_query_controls \
      --feature-cache artifacts/features/vjepa_2_1_full_observation_spatial12.pt \
      --target-cache artifacts/targets/safety_query_full.pt \
      --checkpoint-root outputs/safety_query/full \
      --output outputs/safety_query/full/controls.json
    ;;
  vjepa-baselines)
    verify_inputs
    "$PYTHON" -m safety_probing.vjepa_features \
      --shard full --batch-size "${BATCH_SIZE:-8}" \
      --output artifacts/features/vjepa_2_1_full_observation.pt
    "$PYTHON" -m safety_probing.vjepa_linear_probe \
      --feature-cache artifacts/features/vjepa_2_1_full_observation.pt \
      --output-root outputs/vjepa_linear_probe/full
    "$PYTHON" -m safety_probing.vjepa_transformer_probe \
      --feature-cache artifacts/features/vjepa_2_1_full_observation.pt \
      --linear-result outputs/vjepa_linear_probe/full/result.json \
      --output-root outputs/vjepa_transformer_probe/full
    ;;
  ltx-real)
    verify_inputs
    PYTHONPATH=src "$LTX_PYTHON" -m safety_probing.ltx_real_latents --shard full
    "$PYTHON" -m safety_probing.safety_query_targets \
      --experiment configs/experiments/ltx_safety_query.json \
      --shard full \
      --output artifacts/targets/ltx_safety_query_full.pt
    ;;
  ltx-rollout)
    verify_inputs
    PYTHONPATH=src "$LTX_PYTHON" -m safety_probing.ltx_batch \
      --experiment configs/experiments/ltx_2_3_continuation_prompt_v1.json \
      --shard full --split val --split test \
      --output-root outputs/ltx_v2/full_prompt_v1
    "$PYTHON" -m safety_probing.ltx_rollout_cache \
      --real-cache artifacts/features/ltx_2_3_full_real_future.pt \
      --rollout-root outputs/ltx_v2/full_prompt_v1 \
      --output artifacts/features/ltx_2_3_full_rollout_future.pt
    ;;
  ltx-query)
    verify_inputs
    "$PYTHON" -m safety_probing.ltx_safety_query \
      --real-cache artifacts/features/ltx_2_3_full_real_future.pt \
      --rollout-cache artifacts/features/ltx_2_3_full_rollout_future.pt \
      --target-cache artifacts/targets/ltx_safety_query_full.pt \
      --output-root outputs/ltx_safety_query/full_prompt_v1
    ;;
  help|*)
    cat <<'EOF'
Usage: scripts/run_full_grid_completion.sh PHASE

Phases must be run in order:
  preflight    Verify all 1,008 pairs, logs, and RGB frames.
  audit        Rebuild leakage-safe manifests and full-grid statistics.
  vjepa        Build full V-JEPA cache and train/evaluate the frozen protocol.
  vjepa-baselines
               Run full-cohort linear, mean-MLP, and temporal probe controls.
  ltx-real     Build the full LTX real-future oracle cache and targets.
  ltx-rollout  Generate and consolidate full prompt-v1 val/test rollouts.
  ltx-query    Train on real latents and evaluate the full generated rollout cache.

Part 02/03 are workload shards, not independent confirmatory test sets.
EOF
    [[ "$PHASE" == "help" ]] || exit 2
    ;;
esac
