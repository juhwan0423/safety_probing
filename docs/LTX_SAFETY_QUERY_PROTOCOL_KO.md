# LTX Physical Safety Query Protocol

## 1. 목적

LTX-2.3이 생성한 미래 영상 latent에 simulator의 물리적 미래가 보존되는지를 측정한다. 외부 VLM으로
생성 영상을 다시 판정하지 않고, frozen LTX VAE latent 공간 안에서 평가를 닫는다.

```text
Real simulator future RGB
-> frozen LTX VAE encoder
-> physical query decoder training

CCTV prefix
-> frozen LTX-2.3 continuation
-> generated final LTX VAE future latent
-> frozen physical query decoder
-> mass / CoM / dynamics / ZMP / contact / tip
```

Query decoder는 real future latent에서만 학습한다. Generated rollout latent는 validation/test inference에만
사용하며 loss, normalization, checkpoint selection에 들어가지 않는다.

## 2. 비교 질문

1. 실제 future RGB의 LTX VAE latent에서 물리 상태를 복원할 수 있는가?
2. 동일 readout을 생성 future latent에 적용했을 때 그 정보가 얼마나 유지되는가?
3. 생성 rollout이 hidden mass/CoM intervention의 변화 방향과 future stability ordering을 보존하는가?
4. 실패가 VAE representation의 한계인지 generative rollout의 한계인지 구분할 수 있는가?

첫 질문은 `real-future oracle`, 두 번째 이후는 `generated rollout`으로 보고한다. Oracle은 미래 영상을 직접
보므로 prediction 성능이 아니라 frozen VAE + query의 측정 상한이다.

## 3. Frozen latent contract

| 항목 | 값 |
|---|---|
| Model | LTX-2.3 distilled FP8 checkpoint의 video VAE |
| Foundation trainable parameters | 0 |
| Source simulator RGB | frames 0-119, 24 FPS |
| Internal VAE geometry | 121 frames, source frame 119를 한 번 tail-repeat |
| Resize | official PyTorch bilinear aspect-fill + center crop |
| Pixel range | RGB `[0,255]` to `[-1,1]` |
| Full latent | `[128,16,10,18]` |
| Prefix latent | temporal indices 0-8, 9 frames |
| Future latent | temporal indices 9-15, `[7,10,18,128]` |

LTX temporal VAE는 `1 + 8k` frame을 요구한다. Dataset은 120 frames이므로 마지막 source frame을 한 번
반복해 121로 맞춘다. 생성 pipeline도 121 internal frames를 만들고 마지막 generated pixel frame 하나만
평가에서 제외하므로 latent geometry가 동일하다.

VAE encoder는 full 121-frame sequence를 본 뒤 future slice를 만든다. 따라서 future token에는 causal prefix
context가 남을 수 있으며, real과 generated 양쪽에 정확히 같은 slicing을 적용한다. 이를 isolated
future-only image encoding으로 해석하지 않는다.

Real cache는 official LTX encoder lifecycle로 video encoder만 로드한다. Gemma text encoder와 22B diffusion
transformer는 real-oracle cache 생성에 필요하지 않다. Condition별 part를 atomic 저장한 뒤 manifest 순서로
consolidate하므로 중단 후 재개할 수 있다.

## 4. Physical query contract

LTX future latent에서 다음을 동시에 질의한다.

- Static: load mass, lateral load CoM
- Future 10 times: roll/pitch, IMU angular velocity, body velocity, IMU acceleration
- Future 10 times: system CoM, valid ZMP, outward ZMP/CoM margin
- Five contacts: presence와 chassis-local `(x,y)`
- Support polygon: active contact convex hull area와 centroid
- Future summary: minimum ZMP/CoM margin, maximum absolute roll, minimum contact count
- Binary: future tip-over, right-censored mask 적용

Query layout은 static 2 + future state 10 + future contacts 50 + summary 1, 총 63 tokens다. 2-layer,
4-head, 128-dimensional Transformer decoder가 `[7,10,18,128]` memory에 cross-attend한다. Continuous
normalization과 class weight는 real train split에서만 계산하고 model selection은 real validation loss만 쓴다.

## 5. Rollout protocol

현재 첫 baseline은 이미 완료된 `prompt_v1` batch다.

| 항목 | 값 |
|---|---|
| Root | `outputs/ltx_v2/part_01` |
| Conditions | validation 98 + test 102 |
| Rollout seeds | 17, 29, 43, 71 |
| Jobs | 800 |
| Saved latent | 각 job `[1,128,16,10,18]` |
| Primary aggregation | continuous prediction mean, probability mean |

Condition별 네 rollout을 각각 평가하고, primary condition score는 네 prediction의 평균이다. Probability는
logit 평균이 아니라 probability 평균을 사용한다. Rollout seed는 독립 condition 표본처럼 세지 않는다.

`prompt_v1`은 ground를 checkerboard라고 잘못 기술한 archived baseline이다. 수정된 action-aware prompt는
현재 6-rollout pilot만 존재하므로 main table과 섞지 않는다. Prompt를 바꾼 full rerun은 별도 protocol version,
output root와 명시적인 compute 결정이 필요하다.

## 6. Metrics

V-JEPA Physical Safety Query와 겹치는 metric을 그대로 사용한다.

- Mass/CoM physical-unit MAE와 nearest-level accuracy
- All-tier intervention pair의 mass/CoM delta direction accuracy
- Future state channel별 MAE/RMSE/Spearman
- Future contact F1, active-position MAE
- Support polygon coverage, area MAE, centroid error
- Future minimum ZMP/CoM margin, maximum roll, minimum contact count
- Tip AUROC/AUPRC, balanced accuracy, Brier, ECE
- Gold/Silver/Bronze/Reject별 future margin ordering

모든 metric을 `real-future oracle`과 `generated rollout ensemble`에 동일하게 계산한다. 두 결과의 차이가
generative rollout information loss다.

## 7. Controls

Real-future query에 다음 test-time feature intervention을 적용한다.

- zero latent values
- condition permutation within split
- spatial mean repeated over the latent grid
- temporal mean repeated over seven latent frames
- temporal reverse

Generated rollout에는 네 seed 개별 결과와 ensemble, temporal-reverse ensemble을 보고한다. Static 및 min/max
summary는 본질적으로 order-invariant이므로 reverse 결과는 trajectory query와 분리해 해석한다.

## 8. 실행

```bash
cd ~/isaac_ws/safety_probing

# LTX 전용 dense target
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_targets \
  --experiment configs/experiments/ltx_safety_query.json \
  --shard part_01 \
  --output artifacts/targets/ltx_safety_query_part_01.pt

# Frozen LTX VAE real-future cache. Official LTX environment 사용.
PYTHONUNBUFFERED=1 PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_real_latents --shard part_01

# 기존 800 generated rollout latent consolidate
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.ltx_rollout_cache

# Real-only query training + real oracle/generated rollout evaluation
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.ltx_safety_query \
  --output-root outputs/ltx_safety_query/part_01_prompt_v1
```

## 9. 해석 제한

1. Real-future oracle은 미래 영상 접근 상한이지 미래 prediction 결과가 아니다.
2. Generated latent에는 simulator force나 진짜 ZMP가 존재하지 않는다. 출력은 real latent로 학습한 frozen
   readout이 생성 dynamics에 부여한 estimated physical state다.
3. Query가 generated artifact에 out-of-distribution일 수 있으므로 oracle gap과 영상 failure rate를 함께 본다.
4. Prompt-v1 결과를 corrected action-aware prompt 결과처럼 부르면 안 된다.
5. Shard 01은 exploratory이며 최종 claim에는 untouched shard 검증이 필요하다.

## 10. Source files

- Experiment: `configs/experiments/ltx_safety_query.json`
- Real VAE cache: `src/safety_probing/ltx_real_latents.py`
- Rollout cache: `src/safety_probing/ltx_rollout_cache.py`
- Query/evaluation: `src/safety_probing/ltx_safety_query.py`
- Physical targets: `src/safety_probing/safety_query_targets.py`
- V-JEPA comparison: `docs/Summary_to_Safety_Query.md`
