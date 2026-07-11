# CCTV Future Safety Model Comparison

## 비교 정의

| 비교군 | 모델 | CCTV prefix 이후 하는 일 |
|---|---|---|
| Pixel-generative | LTX-2.3 | 미래 RGB를 continuation하고 rollout별 위험도를 집계 |
| Direct semantic | Qwen3.6 multimodal | 미래 safety outcome과 probability를 JSON으로 직접 출력 |
| Latent predictive | V-JEPA 2/2.1 | frozen latent의 미래를 예측한 뒤 safety head로 판정 |

세 방법은 native output이 다르므로 공통 최종 평가는 `tip_probability`와 pairwise risk score로 맞춘다.
생성 영상 품질, JSON parse failure, latent prediction loss는 각 family 전용 보조 지표로 별도 보고한다.
LTX의 safety score는 다른 foundation model에 재질문하지 않는다. Frozen LTX VAE로 실제 future RGB를
encode해 작은 temporal safety readout을 학습하고, 같은 readout을 LTX rollout의 최종 future latent에
적용한다. 따라서 pixel rollout과 safety decision은 LTX 계열 안에서 닫혀 있다.

## 고정 checkpoint

- LTX: `Lightricks/LTX-2.3-fp8`, distilled FP8 checkpoint
- LTX text encoder: `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized`
- Qwen: `Qwen/Qwen3.6-27B-FP8`
- Latent primary: V-JEPA 2.1 ViT-L/16 384
- Latent ablation: V-JEPA 2 ViT-L/16 256

정확한 repository revision은 `configs/model_registry.json`에 pin한다. Model weight와 외부 code는 각각
`models/`, `external/`에 두며 Git에는 포함하지 않는다.

## 공통 데이터 계약

- Camera: `main_fixed_northwest`
- Observation: 0.0-2.7083 s, source 24 FPS, 65 frames
- Target: 2.7083-5.0 s
- Positive: `observed_tip`
- Negative: `confirmed_recovered`
- Binary excluded: `right_censored`
- Primary split: endpoint-connected leakage-safe main split
- Primary causal metric: same-command Gold pair accuracy

Qwen과 LTX prompt에는 mass, CoM, route label, 정답 label을 넣지 않는다. V-JEPA predictor에도 RGB와
공통 action ablation 이외의 physics metadata를 넣지 않는다.

65프레임은 LTX video VAE의 시간축 조건 `(frames - 1) % 8 == 0`을 만족한다. LTX는 내부적으로
121프레임을 만들고, prefix 뒤의 56 generated frames 중 마지막 1프레임을 버려 실제 데이터의 남은
55프레임과 비교한다. Qwen과 V-JEPA도 같은 65프레임까지만 볼 수 있다. 이 boundary는 2.0초 steering
onset 이후 약 0.7초를 포함하지만 가장 빠른 definitive event 2.958초보다 앞선다.

## 실행 단계

```bash
cd ~/isaac_ws/safety_probing
source .venv/bin/activate

safety-models status --registry configs/model_registry.json
safety-models download --registry configs/model_registry.json
safety-smoke-inputs
```

실험 장비 배정은 `configs/hardware_profiles.json`으로 고정한다.

| Host | Assigned workload | Offload |
|---|---|---|
| RTX 5090 32GB | LTX-2.3 continuation, V-JEPA 2/2.1 feature extraction | none |
| DGX Spark 128GB unified memory | Qwen3.6 multimodal inference | none |

Qwen은 NVIDIA ARM64 vLLM container를 DGX Spark에서 실행한다. 5090에서 수행했던 8GB CPU-offload
smoke는 backend 호환성 확인 기록일 뿐 본 실험 결과에는 사용하지 않는다. Transformers 5.13의
fine-grained FP8 + Accelerate 경로도 scale tensor를 일부 무시해 반복 난문자를 생성했으므로 제외한다.
LTX의 primary runtime은 official FP8 checkpoint와 `fp8-scaled-mm`, no-offload이며 실제 peak가
25.43GB라 5090에 들어간다. 향후 32GB를 넘는 설정은 CPU offload로 우회하지 않고 별도 대용량 hardware
profile과 launcher를 먼저 정의한다.

## 실험 순서

1. 세 outcome의 deterministic smoke prefix로 decoder와 parser를 검증한다.
2. Validation 16 runs에서 OOM, JSON validity, generated frame continuity를 확인한다.
3. 전체 embedding/inference 결과를 run ID 단위로 cache한다.
4. AUROC/AUPRC/ECE와 Gold pair accuracy를 계산한다.
5. left-to-right, right-to-left mirror generalization을 각각 평가한다.

### Shard 01 LTX rollout batch

현재 Computer 1의 canonical manifest에서는 main split validation 98 conditions, test 102 conditions가
선택된다. 네 seed를 적용하면 800 rollouts이며 right-censored 3 conditions도 provenance와 survival 보조
분석을 위해 생성하되 binary metric에서는 제외한다. Batch runner는 pipeline을 한 번만 load하고
rollout별 atomic output과 sidecar를 기록한다. 같은 command를 재실행하면 완료된 rollout을 건너뛴다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_batch \
  --shard part_01 --split val --split test \
  --output-root outputs/ltx_v2/part_01
```

Qwen server와 smoke client는 서로 다른 host에서 실행한다.

```bash
# DGX Spark
scripts/start_qwen_vllm.sh

# RTX 5090 workstation
export QWEN_VLLM_SERVER_URL=http://dgx-spark.local:8016
.venv/bin/safety-qwen-vllm-smoke --outcome confirmed_recovered
```

MP4 prefix는 base64 data URI로 전송하므로 두 host 사이에 동일한 dataset 경로나 shared filesystem이
필요하지 않다. 상세 절차는 `docs/DGX_SPARK_RUNBOOK_KO.md`를 따른다.
