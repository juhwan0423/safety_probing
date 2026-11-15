# CCTV Future Safety Model Comparison

## 비교 정의

| 비교군 | 모델 | CCTV prefix 이후 하는 일 |
|---|---|---|
| Pixel-generative | LTX-2.3 | 미래 RGB를 continuation하고 rollout별 위험도를 집계 |
| Direct semantic, deferred | Qwen3.6 multimodal | 미래 safety outcome과 probability를 JSON으로 직접 출력 |
| Latent predictive J0 | V-JEPA 2 | released frozen predictor로 미래 latent를 예측하고 paired candidate energy를 비교 |
| Representation probe J1/J2 | V-JEPA 2/2.1 | frozen observation latent에서 linear/transformer probe로 안전성을 판정 |
| Physical Safety Query SQ | V-JEPA 2.1 | frozen spatial-temporal latent에서 물성·동역학·ZMP·접촉 geometry를 질의 |
| Generated Physical SQ | LTX-2.3 | real-only query로 generated future latent의 물리 정보 보존을 평가 |

2026-07-12 workshop scope에서 Qwen production batch는 일정상 보류한다. Main empirical comparison은
V-JEPA observation-prefix Physical Safety Query와 LTX generated-rollout Physical Safety Query다. Qwen
checkpoint, launcher와 protocol은 후속 실험용으로 보존하지만 본문 결과표의 빈 행으로 두지 않는다.

세 방법의 native output은 서로 다르다. LTX는 RGB, Qwen은 semantic JSON, V-JEPA J0는 미래 latent를
출력한다. 따라서 J0의 직접 지표는 candidate retrieval accuracy와 latent energy이며 독립적인
`tip_probability`를 주장하지 않는다. 공통 AUROC/AUPRC는 J1/J2 probe 또는 명시적인 non-parametric
readout에서만 계산한다. 생성 영상 품질, JSON parse failure, latent prediction energy는 각 family 전용
지표로 별도 보고한다.
LTX의 safety score는 다른 foundation model에 재질문하지 않는다. Frozen LTX VAE로 실제 future RGB를
encode해 작은 temporal safety readout을 학습하고, 같은 readout을 LTX rollout의 최종 future latent에
적용한다. 따라서 pixel rollout과 safety decision은 LTX 계열 안에서 닫혀 있다.

## 고정 checkpoint

- LTX: `Lightricks/LTX-2.3-fp8`, distilled FP8 checkpoint
- LTX text encoder: `Lightricks/gemma-3-12b-it-qat-q4_0-unquantized`
- Qwen: `Qwen/Qwen3.6-27B-FP8`
- Native latent prediction J0: V-JEPA 2 ViT-L/16 256
- Probe primary J1/J2: V-JEPA 2.1 ViT-L/16 384
- Probe ablation J1/J2: V-JEPA 2 ViT-L/16 256

V-JEPA 2.1 ViT-L release는 1024-dimensional student encoder와 1664-dimensional teacher-space predictor를
함께 제공한다. 동일 checkpoint 안에 그 teacher target encoder가 없으므로 target energy를 임의 projection으로
맞추지 않는다. J0는 predictor와 target encoder의 차원이 일치하는 V-JEPA 2로 고정하고, V-JEPA 2.1은
encoder probing에 사용한다.

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

## V-JEPA 3-stage protocol

### J0. Native frozen future prediction

관측 65 source frames에서 20 frames, 미래 55 source frames에서 20 frames를 각각 균등 sampling한다.
V-JEPA의 2-frame tubelet 경계가 observation/future 사이를 가로지르지 않도록 총 40 model frames를
`20 context + 20 future`로 구성한다. Context token만 encoder에 남기고 future token 2,560개를 모두
mask한 뒤 released predictor로 채운다. Encoder와 predictor의 trainable parameter count는 0이다.

Gold pair의 각 query마다 관측 prefix는 고정하고 candidate future만 A 또는 B로 교체한다. 이 두 hybrid
clip을 동일한 frozen target encoder로 encode한 뒤 predicted future와의 mean token cosine distance를
비교한다. 따라서 A/B의 과거 identity 차이로 candidate를 고르는 것을 막는다. 이 평가는 candidate가
주어진 multiple-choice retrieval이며 standalone tip classification은 아니다.

### J1. Frozen linear probe

관측 prefix의 frozen encoder token을 고정 mean pooling하고 linear head만 학습한다. 미래 predictor는
사용하지 않는다. 안전 정보가 선형적으로 접근 가능한지를 측정한다.

### J2. Frozen transformer probe

동일한 관측 token sequence에 2-layer, 4-head temporal Transformer를 학습한다. J1보다 유의하게 좋아질
때만 temporal interaction의 이점을 주장한다.

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
| DGX Spark 128GB unified memory | Qwen3.6 multimodal inference, deferred | none |

Qwen은 NVIDIA ARM64 vLLM container를 DGX Spark에서 실행한다. 5090에서 수행했던 8GB CPU-offload
smoke는 backend 호환성 확인 기록일 뿐 본 실험 결과에는 사용하지 않는다. Transformers 5.13의
fine-grained FP8 + Accelerate 경로도 scale tensor를 일부 무시해 반복 난문자를 생성했으므로 제외한다.
LTX의 primary runtime은 official FP8 checkpoint와 `fp8-scaled-mm`, no-offload이며 실제 peak가
25.43GB라 5090에 들어간다. 향후 32GB를 넘는 설정은 CPU offload로 우회하지 않고 별도 대용량 hardware
profile과 launcher를 먼저 정의한다.

## 실험 순서

1. 세 outcome의 deterministic smoke prefix로 decoder와 parser를 검증한다.
2. J0 한 Gold pair에서 frozen future-mask retrieval을 검증한다.
3. Validation Gold pairs 전체에서 J0 energy와 retrieval accuracy를 계산한다.
4. J1/J2 frozen feature cache와 probe 학습을 수행한다.
5. AUROC/AUPRC/ECE와 Gold pair accuracy를 계산한다.
6. left-to-right, right-to-left mirror generalization을 각각 평가한다.
7. Real LTX future latent에서 physical query를 학습한다.
8. 같은 frozen query로 네 seed의 generated rollout을 평가한다.

J0 smoke command:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_native \
  --split val \
  --output outputs/vjepa_native/smoke_result.json \
  --latent-output outputs/vjepa_native/smoke_latents.pt
```

Shard 01의 main validation 17 + test 10 Gold pairs를 한 번의 model load로 평가한다. Pair별 sidecar가
완료 단위이므로 중단 후 동일 command를 다시 실행하면 이어서 진행한다.

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

J1은 V-JEPA 2.1 encoder를 한 번만 실행해 canonical observation feature를 cache하고, 이후 세 seed의
linear classifier와 ZMP-margin regressor를 cache 위에서 학습한다.

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

Shard 01 exploratory 결과는 test AUROC `0.960 +/- 0.008`, Gold pair accuracy
`0.50 +/- 0.10`이다. 일반 위험도는 잘 분리하지만 same-command hidden intervention은 아직 분리하지
못했다. 상세 해석은 `docs/Summary_to_J1.md`에 고정한다.

J2는 같은 cache에서 10개 temporal token을 보존한 2-layer Transformer를 학습한다. Reversed-order와
mean-pool MLP control을 같은 실행에서 계산한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_transformer_probe \
  --feature-cache artifacts/features/vjepa_2_1_part_01_observation.pt \
  --linear-result outputs/vjepa_linear_probe/part_01/result.json \
  --output-root outputs/vjepa_transformer_probe/part_01
```

Shard 01 J2는 AUROC `0.972 +/- 0.002`, Gold pair `0.667 +/- 0.058`이었다. 그러나 reverse-order에서도
pair 결과가 같고 mean-pool MLP가 pair `0.667`, AUROC `0.978`을 기록했다. 따라서 J1 대비 향상은
temporal reasoning이 아니라 nonlinear readout 효과로 해석한다. 상세 결과는 `docs/Summary_to_J2.md`에
고정한다.

### Physical Safety Query

J1/J2의 label 중심 출력과 별도로, V-JEPA 2.1의 `10 x 12 x 12` frozen token grid에 123개의 learned
physical query를 cross-attend한다. Query decoder는 mass, lateral CoM, observed/future 18-channel state,
다섯 contact point, support polygon과 future stability summary를 동시에 복원한다. Foundation parameter는
계속 0개 trainable이며 query decoder만 570,928 parameters다.

Shard 01 exploratory 결과는 mass MAE `14.44 +/- 1.17 kg`, lateral CoM MAE
`0.0331 +/- 0.0055 m`, future minimum ZMP-margin MAE `0.0457 +/- 0.0053 m`다. Test의 all-tier
intervention pair에서 mass delta direction은 `0.994`, CoM delta direction은 `0.961`이었다. Zero/permuted
feature에서는 결과가 붕괴했지만 reverse-time에는 거의 불변이므로 causal temporal-order reasoning은
아직 주장하지 않는다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_spatial_features --shard part_01 --batch-size 4
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_targets --shard part_01
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_probe --output-root outputs/safety_query/part_01
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_controls --checkpoint-root outputs/safety_query/part_01
```

전체 target/coordinate/mask 계약과 해석 제한은 `docs/Summary_to_Safety_Query.md`에 고정한다.

### Shard 01 LTX rollout batch

현재 Computer 1의 canonical manifest에서는 main split validation 98 conditions, test 102 conditions가
선택된다. 네 seed를 적용하면 800 rollouts이며 right-censored 3 conditions도 provenance와 survival 보조
분석을 위해 생성하되 binary metric에서는 제외한다. Batch runner는 pipeline을 한 번만 load하고
rollout별 atomic output과 sidecar를 기록한다. 같은 command를 재실행하면 완료된 rollout을 건너뛴다.

`prompt_v1` 800 rollouts은 이미 완료됐으며 `outputs/ltx_v2/part_01`에 보존한다. 이 prompt는 실제
blue grid를 checkerboard라고 잘못 기술했다. 수정한 action-aware neutral prompt는 6-rollout balanced
pilot에서 의미상 정합성은 좋아졌지만 문자, 부품 복제, 배경 변형이 계속됐고 visible tip은 0건이었다.
따라서 두 번째 800-rollout batch는 oracle forced-tip/forced-stable smoke가 끝날 때까지 실행하지 않는다.
또한 action-aware prompt를 main으로 채택하면 Qwen과 V-JEPA에도 동일한 future command를 제공한다.
그렇지 않으면 해당 LTX 결과는 video-only main table이 아니라 action-conditioned ablation으로 분리한다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_batch \
  --shard part_01 --split val --split test \
  --max-per-outcome 2 --seed 17 \
  --output-root outputs/ltx_prompt_v2_pilot
```

### LTX Physical Safety Query

실제 simulator 120 frames의 마지막 frame을 한 번 반복해 LTX VAE가 요구하는 121 frames로 만들고, frozen
VAE encoder의 future latent `[7,10,18,128]`를 cache했다. 450,075-parameter query는 real train/validation
latent에서만 학습·선택한 뒤, 기존 prompt-v1 800 rollout latent에 그대로 적용했다.

Real-future oracle은 mass MAE `11.93 kg`, CoM MAE `0.0213 m`, tip AUROC `0.9996`이었다. Generated
four-rollout ensemble은 각각 `105.00 kg`, `0.1845 m`, `0.7230`으로 하락했다. Mass intervention pair
direction은 `0.800`으로 일부 남았으나 CoM pair는 `0.598`, Gold ZMP-risk ordering은 `0.333`이었다.

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_real_latents --shard part_01
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.ltx_rollout_cache
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.ltx_safety_query \
  --output-root outputs/ltx_safety_query/part_01_prompt_v1
```

이 결과는 `prompt-v1 archived baseline`이다. Real-future oracle은 prediction 결과가 아니며, generated
latent의 ZMP는 simulator force에서 직접 계산된 값이 아니라 frozen real-latent readout의 estimate다. 전체
계약과 결과는 `docs/LTX_SAFETY_QUERY_PROTOCOL_KO.md` 및 `docs/Summary_to_LTX_Safety_Query.md`에 고정한다.

### Deferred Qwen protocol

Qwen server와 smoke client는 서로 다른 host에서 실행한다. 아래 경로는 후속 실험용이며 현재 workshop
main result에는 포함하지 않는다.

Selection, prompt, metric, pair scarcity 해석은 `docs/qwen_protocol.md`에 별도 동결한다.

```bash
# DGX Spark
scripts/start_qwen_vllm.sh

# RTX 5090 workstation
export QWEN_VLLM_SERVER_URL=http://dgx-spark.local:8016
.venv/bin/safety-qwen-vllm-smoke --outcome confirmed_recovered
```

MP4 prefix는 base64 data URI로 전송하므로 두 host 사이에 동일한 dataset 경로나 shared filesystem이
필요하지 않다. 상세 절차는 `docs/DGX_SPARK_RUNBOOK_KO.md`를 따른다.
