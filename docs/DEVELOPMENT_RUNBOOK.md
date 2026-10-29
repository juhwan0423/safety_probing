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
- evaluate the released frozen V-JEPA predictor with future-latent retrieval

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

- [Dataset Card](../docs/DATASET_CARD_KO.md)
- [Schema Reference](../docs/SCHEMA_REFERENCE.md)
- [Generated Statistics](../docs/DATASET_STATISTICS_KO.md)
- [Probing Protocol](../docs/PROBING_PROTOCOL_KO.md)
- [Model Experiment Plan](../docs/MODEL_EXPERIMENT_PLAN_KO.md)
- [Model Runtime Status](../docs/MODEL_RUNTIME_STATUS_KO.md)
- [DGX Spark Runbook](../docs/DGX_SPARK_RUNBOOK_KO.md)
- [Qwen Direct-Semantic Protocol](../docs/qwen_protocol.md)
- [LTX Prompt Audit](../docs/LTX_PROMPT_AUDIT_KO.md)
- [Summary to J0](../docs/Summary_to_J0.md)
- [Summary to J1](../docs/Summary_to_J1.md)
- [Summary to J2](../docs/Summary_to_J2.md)
- [Physical Safety Query Summary](../docs/Summary_to_Safety_Query.md)
- [LTX Physical Safety Query Protocol](../docs/LTX_SAFETY_QUERY_PROTOCOL_KO.md)
- [LTX Physical Safety Query Summary](../docs/Summary_to_LTX_Safety_Query.md)

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
PYTHONPATH=src .venv/bin/python -m safety_probing.vjepa_native --split val
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

## V-JEPA Three-Stage Protocol

- J0 native: released V-JEPA 2 encoder/predictor frozen, full future mask, paired
  candidate latent-energy retrieval
- J1 linear: frozen observed-prefix tokens, fixed mean pooling, linear safety heads
- J2 transformer: frozen observed-prefix tokens, two-layer temporal Transformer heads

Run the deterministic J0 validation Gold-pair smoke:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_native \
  --split val \
  --output outputs/vjepa_native/smoke_result.json \
  --latent-output outputs/vjepa_native/smoke_latents.pt
```

J0 is a paired candidate-retrieval test, not a standalone tip classifier. For each
query, its observed prefix is held fixed while only the candidate A/B future is
swapped. J1/J2 provide the standalone safety probabilities used for AUROC and
calibration.

Run the resumable J0 batch over shard 01 validation and test Gold pairs. The model
is loaded once for all 27 pairs.

```bash
mkdir -p run_logs
set -o pipefail
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_native_batch \
  --shard part_01 \
  --split val --split test \
  --save-latents \
  --output-root outputs/vjepa_native_batch/part_01 \
  2>&1 | tee run_logs/vjepa_j0_part_01.log
```

Pair results are written under `split/pair_id/result.json`. Re-running the same
command skips completed pairs and refreshes `batch_manifest.jsonl` and
`batch_summary.json`.

Run J1 frozen feature extraction and the three-seed linear probe:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_features \
  --shard part_01 --batch-size 8 \
  --output artifacts/features/vjepa_2_1_part_01_observation.pt

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_linear_probe \
  --feature-cache artifacts/features/vjepa_2_1_part_01_observation.pt \
  --output-root outputs/vjepa_linear_probe/part_01
```

Shard 01 J1 achieved `0.960 +/- 0.008` test AUROC, while the primary Gold
same-command pair accuracy was `0.50 +/- 0.10`. See the J1 summary for the
global-versus-counterfactual interpretation.

Run J2 and its reversed-order/mean-pool-MLP controls from the same frozen cache:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_transformer_probe \
  --feature-cache artifacts/features/vjepa_2_1_part_01_observation.pt \
  --linear-result outputs/vjepa_linear_probe/part_01/result.json \
  --output-root outputs/vjepa_transformer_probe/part_01
```

J2 reached `0.972 +/- 0.002` AUROC and `0.667 +/- 0.058` Gold-pair accuracy.
The smaller mean-pool MLP matched pair accuracy and reached `0.978 +/- 0.003`
AUROC; reversing temporal tokens did not change J2 pair decisions. The J2
summary therefore attributes the gain over J1 to nonlinear readout capacity,
not temporal-order reasoning.

## Physical Safety Query

The physical Safety Query keeps the V-JEPA 2.1 encoder frozen but preserves a
`10 x 12 x 12` spatiotemporal token grid. A small cross-attention decoder queries
hidden load mass/CoM, observed and future dynamics, ZMP/CoM margins, five contact
points, support polygons, and future tip outcome in physical units.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_spatial_features --shard part_01 --batch-size 4

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_targets --shard part_01

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_probe \
  --output-root outputs/safety_query/part_01

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_controls \
  --checkpoint-root outputs/safety_query/part_01
```

On the exploratory shard 01 split, load mass MAE was `14.44 +/- 1.17 kg`,
lateral CoM MAE was `0.0331 +/- 0.0055 m`, and future minimum ZMP-margin MAE
was `0.0457 +/- 0.0053 m`. Zeroing or permuting encoder features collapsed the
results, while reversing temporal token order did not. See the physical Safety
Query summary for the full target contract, support-polygon metrics, controls,
and interpretation limits.

## Full-grid completion

`part_01/02/03` are workload shards of one 1,008-pair grid, not independent
scientific test shards. After copying part 02/03 into their configured dataset
roots, begin with the strict integrity check:

```bash
scripts/run_full_grid_completion.sh preflight
```

The staged audit, V-JEPA, LTX real-oracle, rollout, and query commands are in
`docs/FULL_GRID_COMPLETION_RUNBOOK_KO.md`. Every long phase verifies the frozen
protocol lock before running.

## LTX Physical Safety Query

The LTX experiment trains a 450,075-parameter physical query only on frozen VAE
latents from real simulator futures, then freezes it and evaluates the 800 saved
prompt-v1 rollout latents. This keeps the generative evaluation inside the LTX
latent family without asking an external VLM to judge the videos.

The real-future oracle reached `11.93 kg` mass MAE, `0.0213 m` CoM MAE, and
`0.9996` tip AUROC. On generated rollouts those became `105.00 kg`, `0.1845 m`,
and `0.7230`. Mass-pair direction remained `0.800`, but Gold ZMP-risk ordering
fell to `0.333`, showing that coarse outcome signal did not imply physically
consistent stability generation. See the LTX Safety Query summary for the full
oracle/rollout comparison and prompt-v1 limitation.
