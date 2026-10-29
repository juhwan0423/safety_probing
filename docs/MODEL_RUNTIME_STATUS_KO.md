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

#### J0 native future retrieval

- Model: V-JEPA 2 ViT-L/16, released encoder and predictor frozen
- Input: 20 observation frames + 20 fully masked future frames
- Predicted future: `[2,2560,1024]`
- Context-controlled A/B candidate targets: `[2,2,2560,1024]`
- Smoke pair: validation Gold `A_mass_s0_left_vmedium_smedium_comcenter_m125_to_m150`
- A query: tip candidate selected, cosine margin `0.006216`
- B query: recovered candidate selected, cosine margin `0.002078`
- Identity/outcome retrieval: `2/2`
- Peak GPU allocation: 2,079,602,176 bytes, no CPU offload

이는 한 pair의 runtime 및 protocol smoke 결과이며 dataset 성능 추정치가 아니다. 후보 A/B 미래가
주어진 retrieval이므로 standalone tip classification AUROC로 해석하지 않는다.

Resumable batch runner도 별도 output root에서 1-pair 실제 실행 후 같은 command를 재실행해
`Complete=1, pending=0` skip을 확인했다. 현재 shard 01 batch 대상은 validation 17 + test 10 Gold
pairs다.

#### J1 frozen linear probe

- Model: V-JEPA 2.1 ViT-L/16 384, encoder frozen
- Canonical conditions: 670
- Cache: `[670,10,1024]` float16, finite
- Feature extraction: 219.714 s
- Peak GPU allocation: 3,493,543,936 bytes, no CPU offload
- Binary test: 99 definitive runs
- Gold test: 10 matched pairs
- Test AUROC: `0.960 +/- 0.008`, three probe seeds
- Gold pair accuracy: `0.50 +/- 0.10`
- Margin MAE: `0.0705 +/- 0.0030 m`

Runtime과 standalone probing pipeline은 정상이다. Global AUROC와 primary causal pair metric의 차이는
`docs/Summary_to_J1.md`에서 별도로 해석한다.

#### J2 temporal Transformer probe

- Input cache: `[670,10,1024]`, J1 cache 재사용
- Transformer trainable parameters/task: 1,319,937
- Mean-pool MLP control parameters/task: 263,169
- Seeds: 17, 29, 43
- J2 test AUROC: `0.972 +/- 0.002`
- J2 Gold pair accuracy: `0.667 +/- 0.058`
- Reversed-order Gold pair accuracy: `0.667 +/- 0.058`
- Mean-pool MLP AUROC: `0.978 +/- 0.003`
- Mean-pool MLP Gold pair accuracy: `0.667 +/- 0.058`
- J2 mean runtime/seed: 2.07 s
- J2 peak GPU allocation: 149,779,968 bytes

Transformer는 reversed order와 같은 결정을 냈고 mean-pool MLP를 넘지 못했다. Runtime은 정상이나
temporal-order advantage는 관찰되지 않았다. 상세 해석은 `docs/Summary_to_J2.md`에 기록한다.

#### Physical Safety Query

- Encoder: V-JEPA 2.1 frozen, foundation trainable parameters 0
- Spatial feature cache: `[670,10,12,12,1024]` float16, 1.9 GB
- Feature extraction: 219.349 s
- Encoder extraction peak GPU allocation: 3,540,402,176 bytes
- Dense target cache: 670 conditions, maximum alignment error 0.006944 s
- Query decoder: 570,928 trainable parameters
- Query training: seeds 17/29/43, mean 113.73 s/seed
- Query training peak GPU allocation: 약 156.6 MB
- Test mass MAE: `14.44 +/- 1.17 kg`
- Test lateral CoM MAE: `0.0331 +/- 0.0055 m`
- Future minimum ZMP-margin MAE: `0.0457 +/- 0.0053 m`
- Future tip AUROC: `0.9996 +/- 0.0004`
- All-tier test pair delta direction: mass `0.994`, CoM `0.961`

Zero-memory control은 tip AUROC 0.5와 pair direction 0.5로 붕괴했고, split 내부 condition permutation도
mass/CoM MAE를 각각 182.77 kg/0.213 m로 악화시켰다. Spatial/temporal mean 제거도 크게 악화됐지만
reverse-time은 원본과 사실상 같았다. 전체 결과와 제한은 `docs/Summary_to_Safety_Query.md`에 기록한다.

### Qwen3.6

Workshop main experiment에서는 2026-07-12 일정 결정에 따라 deferred다. 아래 내용은 runtime compatibility
기록이며 production benchmark 결과가 아니다.

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
- Shard 01 `prompt_v1` batch: 800/800 complete, no runtime failures
- Visual audit of 40 sampled `observed_tip` final frames: zero visible rollovers; frequent duplication, text, and ground artifacts
- Corrected action-aware neutral prompt pilot: 6/6 complete, zero visible rollovers; artifact modes persisted
- Full rerun with the corrected prompt is deferred until forced-tip/forced-stable controllability smoke

#### Physical Safety Query

- Real VAE encoder parameters: 318,936,897, all frozen
- Real future cache: `[670,7,10,18,128]` float16, 207 MB
- Exact cache resume run: 475 encoded + 195 reused in 627.6 s
- VAE encode peak GPU allocation: 8,397,758,464 bytes
- Consolidated prompt-v1 rollout cache: `[800,7,10,18,128]` float16, 247 MB
- Query decoder: 450,075 trainable parameters
- Query training source: real future latents only
- Generated latents used for training/model selection: no
- Query seeds: 17, 29, 43
- Mean query runtime: 33.19 s/seed
- Query peak GPU allocation: 94,252,544 bytes

| Metric | Real-future oracle | Generated 4-rollout ensemble |
|---|---:|---:|
| Mass MAE | `11.93 +/- 0.83 kg` | `105.00 +/- 5.92 kg` |
| CoM MAE | `0.0213 +/- 0.0015 m` | `0.1845 +/- 0.0086 m` |
| Future roll MAE | `2.32 +/- 0.35 deg` | `16.28 +/- 2.18 deg` |
| Min-ZMP summary MAE | `0.0313 +/- 0.0021 m` | `0.1163 +/- 0.0140 m` |
| Tip AUROC | `0.9996 +/- 0.0007` | `0.7230 +/- 0.0538` |
| Mass pair direction | `0.994` | `0.800` |
| CoM pair direction | `1.000` | `0.598` |

Real zero-memory control은 tip AUROC와 pair direction이 모두 0.5로 붕괴했다. Generated Gold ZMP-risk
ordering은 `0.333`이어서 coarse tip ranking이 물리적으로 일관된 ZMP rollout을 뜻하지 않는다. 상세 결과는
`docs/Summary_to_LTX_Safety_Query.md`에 기록한다.

## Commands

```bash
cd ~/isaac_ws/safety_probing

.venv/bin/safety-models status --registry configs/model_registry.json
.venv/bin/safety-model-smoke vjepa_2
.venv/bin/safety-model-smoke vjepa_2_1

PYTHONPATH=src .venv/bin/python -m safety_probing.vjepa_native \
  --split val \
  --output outputs/vjepa_native/smoke_result.json

PYTHONPATH=src external/LTX-2/.venv/bin/python -m safety_probing.ltx_smoke

PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_real_latents --shard part_01
PYTHONPATH=src .venv/bin/python -m safety_probing.ltx_rollout_cache
PYTHONPATH=src .venv/bin/python -m safety_probing.ltx_safety_query \
  --output-root outputs/ltx_safety_query/part_01_prompt_v1
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
