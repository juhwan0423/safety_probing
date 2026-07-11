# Model Runtime Status

## Pinned assets

| Arm | Asset | Local size | Status |
|---|---|---:|---|
| Pixel-generative | LTX-2.3 distilled FP8 | 29.53 GB | complete |
| Pixel-generative | LTX spatial upscaler | 1.00 GB | complete |
| Pixel-generative | Gemma 3 12B text encoder | 24.41 GB | complete |
| Direct semantic | Qwen3.6-27B-FP8 | 30.89 GB | complete |
| Latent predictive | V-JEPA 2 ViT-L | 5.13 GB | complete |
| Latent predictive | V-JEPA 2.1 ViT-L | 5.15 GB | complete |

정확한 repository revision과 파일 검증 기준은 `configs/model_registry.json` 및
`models/download_manifest.json`에 기록한다.

## Verified smoke results

### V-JEPA

| Model | Input | Output latent | Finite | Peak VRAM |
|---|---|---|---|---:|
| V-JEPA 2 | `[1,3,20,256,256]` | `[1,2560,1024]` | yes | 1.40 GB |
| V-JEPA 2.1 | `[1,3,20,384,384]` | `[1,5760,1024]` | yes | 1.58 GB |

두 모델 모두 65-frame CCTV prefix에서
`[0,3,7,10,13,17,20,24,27,30,34,37,40,44,47,51,54,57,61,64]`에 해당하는
20개 균등 프레임을 사용해 다시 검증했다.

### Qwen3.6

- Historical compatibility smoke: vLLM 0.20.0, CUDA 13.0, RTX 5090, eager mode
- Model weights on GPU: 20.4 GiB
- CPU-offloaded parameters: 8.09 GiB
- Free KV cache: 5.73 GiB / 22,736 tokens
- CCTV request: 20 frames, 1,168 prompt tokens, 27 completion tokens
- Structured JSON generation: valid for both recovered and tip smoke clips
- Recovered smoke prediction: `observed_tip`, probability 0.95, incorrect
- Tip smoke prediction: `observed_tip`, probability 0.95, correct

두 결과가 같은 것은 runtime failure가 아니라 zero-shot model behavior다. 본 실험에서는 전체 validation/test
분포의 AUROC, calibration, pair accuracy로 판단한다. 위 5090 smoke는 API와 FP8 kernel의 호환성만
확인한 과거 기록이며 production 결과에는 사용하지 않는다.

Production profile은 DGX Spark의 NVIDIA vLLM `26.06-py3` ARM64 container, Qwen full model load,
`cpu_offload_gb=0`이다. 설정과 launcher는 준비됐지만 DGX Spark에서의 실제 smoke는 아직 실행 전이다.

Transformers 5.13 + Accelerate CPU offload는 모든 layer의 일부 FP8 scale을 무시했고 반복 난문자를
생성했다. 같은 현상은 upstream model discussion에서도 보고되었으므로 이 backend는 제외한다:
<https://huggingface.co/Qwen/Qwen3.6-27B-FP8/discussions/3>

### LTX-2.3

- Prefix contract preflight: passed
- Prefix: 65 frames; internal sequence: 121 frames
- Evaluated future: 55 frames; one generated tail frame discarded
- Resolution: 576x320, valid for the two-stage 64-pixel divisibility constraint
- Full continuation smoke: passed with official locked environment
- Output: 121 frames, 24 FPS, 5.0417 seconds, H.264 yuv420p
- Final VAE latent: `[1,128,16,10,18]`
- Runtime: `fp8-scaled-mm`, no offload, peak VRAM 25.43 GB
- 65-frame full continuation smoke: passed, seed 17
- Output: `outputs/smoke/ltx_confirmed_recovered_65f.mp4`
- Representative frames at 64, 80, 96, 120 were nonblank and decodable
- The vehicle remained fully visible at frame 120; the old 57-frame rollout had nearly exited the lower-right frame

## Commands

```bash
cd ~/isaac_ws/safety_probing

.venv/bin/safety-models status --registry configs/model_registry.json
.venv/bin/safety-model-smoke vjepa_2
.venv/bin/safety-model-smoke vjepa_2_1

PYTHONPATH=src external/LTX-2/.venv/bin/python -m safety_probing.ltx_smoke
```

DGX Spark에서 Qwen server:

```bash
scripts/start_qwen_vllm.sh
```

5090 workstation에서 remote smoke client:

```bash
export QWEN_VLLM_SERVER_URL=http://dgx-spark.local:8016
.venv/bin/safety-qwen-vllm-smoke --outcome observed_tip
```
