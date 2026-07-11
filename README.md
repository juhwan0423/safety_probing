# Safety Probing

`safety_probing` is the analysis and frozen-encoder probing workspace for the
ExoFork-CF lateral tip-over dataset. It treats simulator outputs as read-only inputs.

## Current Scope

- audit all dataset shards and generate reproducible statistics
- build split manifests without repeated physical-condition leakage
- emit pair-level and run-level JSONL manifests for feature extraction
- freeze the dataset specification and temporal probing protocol in Markdown
- manage pinned LTX-2.3, Qwen3.6, and V-JEPA checkpoints
- prepare deterministic CCTV prefix videos for model smoke tests

## Quick Start

```bash
cd ~/isaac_ws/safety_probing
uv sync --locked \
  --extra vision-audit --extra model-prep --extra inference-core

.venv/bin/safety-probing all --config configs/local_dataset.json
```

Without installation:

```bash
PYTHONPATH=src python3 -m safety_probing all \
  --config configs/local_dataset.json
```

For a fast inventory that does not scan every dense CSV row:

```bash
safety-probing analyze --config configs/local_dataset.json --metadata-only
```

## Outputs

- `artifacts/splits/leakage_safe_splits.json`: condition-connected probing splits
- `artifacts/current/dataset_statistics.json`: machine-readable statistics
- `artifacts/current/manifests/pairs.jsonl`: counterfactual pair manifest
- `artifacts/current/manifests/runs.jsonl`: run/clip manifest and targets
- `artifacts/current/manifests/canonical_runs.jsonl`: one deterministic run per physical condition
- `docs/DATASET_STATISTICS_KO.md`: generated human-readable report

`artifacts/current` is intentionally ignored because it can be regenerated. The split
manifest is retained because it is part of the experimental protocol.

## Adding Shards

Place or mount `eccv_workshop_grid_part_02` and `part_03` at the paths listed in
`configs/local_dataset.json`, then rerun `safety-probing all`. Missing roots are reported,
not treated as empty completed shards.

## Documents

- [Dataset Card](docs/DATASET_CARD_KO.md)
- [Schema Reference](docs/SCHEMA_REFERENCE.md)
- [Generated Statistics](docs/DATASET_STATISTICS_KO.md)
- [Probing Protocol](docs/PROBING_PROTOCOL_KO.md)
- [Model Experiment Plan](docs/MODEL_EXPERIMENT_PLAN_KO.md)
- [Model Runtime Status](docs/MODEL_RUNTIME_STATUS_KO.md)
- [DGX Spark Runbook](docs/DGX_SPARK_RUNBOOK_KO.md)
- [LTX Prompt Audit](docs/LTX_PROMPT_AUDIT_KO.md)

## Model Preparation

```bash
.venv/bin/safety-models status --registry configs/model_registry.json
.venv/bin/safety-models download --registry configs/model_registry.json
.venv/bin/safety-smoke-inputs

# LTX uses the official repository's separately locked runtime.
cd external/LTX-2
uv sync --frozen
cd ../..

.venv/bin/safety-model-smoke vjepa_2
.venv/bin/safety-model-smoke vjepa_2_1
PYTHONPATH=src external/LTX-2/.venv/bin/python -m safety_probing.ltx_smoke
```

Production workloads do not use CPU offload. LTX-2.3 and V-JEPA run on the RTX
5090; Qwen3.6 runs on the DGX Spark through NVIDIA's ARM64 vLLM container.

```bash
# Run on the DGX Spark after cloning this repository and downloading Qwen.
scripts/start_qwen_vllm.sh

# Run on this workstation. Replace the hostname with the DGX Spark address.
export QWEN_VLLM_SERVER_URL=http://dgx-spark.local:8016
.venv/bin/safety-qwen-vllm-smoke
```

Video requests use base64 data URIs, so the client and DGX Spark do not need a
shared filesystem. See `configs/hardware_profiles.json` for the enforced model
assignment and the DGX Spark runbook for setup details.

## LTX Shard 01 Batch

The batch runner loads LTX once, generates all main-split validation/test
rollouts for the local shard, and resumes from per-rollout JSON sidecars.
The completed 800-rollout `prompt_v1` baseline is preserved at
`outputs/ltx_v2/part_01`. The active action-aware prompt is still a candidate;
do not launch another full batch until the oracle prompt smoke is reviewed.

```bash
# Reproduce the archived prompt_v1 selection without loading the model.
PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_batch \
  --experiment configs/experiments/ltx_2_3_continuation_prompt_v1.json \
  --shard part_01 --split val --split test --dry-run

# Balanced six-rollout pilot for the active prompt candidate.
PYTHONUNBUFFERED=1 PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_batch \
  --shard part_01 --split val --split test \
  --max-per-outcome 2 --seed 17 \
  --output-root outputs/ltx_prompt_v2_pilot
```

Outputs are keyed by `split/condition_id/seed`, with a consolidated
`batch_manifest.jsonl` and `batch_summary.json` under the output root.

The shared CCTV observation contract is 65 frames at 24 FPS (2.7083 seconds).
LTX uses an internal 121-frame sequence because its temporal VAE requires `8n+1`
frames; only the matching 55 future frames are scored.
