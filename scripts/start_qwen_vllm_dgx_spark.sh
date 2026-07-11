#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="$(uname -m)"

if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  printf '%s\n' \
    "Qwen3.6 production inference is assigned to DGX Spark (ARM64)." \
    "Run this launcher on the DGX Spark; CPU offload on the RTX 5090 is disabled."
  exit 2
fi

MODEL_DIR="${QWEN_MODEL_DIR:-$ROOT/models/qwen_3_6}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.06-py3}"
PORT="${QWEN_VLLM_PORT:-8016}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  printf 'Qwen checkpoint not found at %s\n' "$MODEL_DIR" >&2
  printf '%s\n' \
    "Download it first with: .venv/bin/safety-models download --model qwen_3_6" >&2
  exit 2
fi

exec docker run --rm \
  --name safety-probing-qwen \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -p "$PORT:$PORT" \
  -v "$MODEL_DIR:/models/qwen_3_6:ro" \
  "$VLLM_IMAGE" \
  vllm serve /models/qwen_3_6 \
  --served-model-name qwen3.6-27b-fp8 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.80 \
  --cpu-offload-gb 0 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --limit-mm-per-prompt '{"video": 1}'
