# Summary to Physical Safety Query

## 1. 무엇을 바꿨나

기존 J2는 CCTV prefix에서 최종 tip label과 단일 ZMP summary를 예측하는 분류/회귀 probe였다. 이것만으로는
우리가 원한 Safety Query, 즉 영상 latent에서 물리 상태를 구조적으로 꺼낼 수 있는지를 답하지 못한다.

이번 SQ 실험은 질문을 다음처럼 바꿨다.

> Frozen V-JEPA의 CCTV representation에서 숨겨진 load property, 현재와 미래의 동역학 상태, 접촉점과
> support polygon, ZMP/CoM stability margin을 물리 단위로 복원할 수 있는가?

V-JEPA 2.1 encoder는 완전히 freeze하고, 공간·시간 token을 읽는 작은 query decoder만 supervised training했다.
Tip AUROC는 여러 출력 중 하나일 뿐이며, 주된 증거는 물리량별 오차와 counterfactual pair의 변화 방향이다.

## 2. 입력과 시간 계약

| 항목 | 값 |
|---|---|
| Camera | `main_fixed_northwest` |
| Observation | source frame 0-64, 0.0-2.7083 s |
| Future target | source frame 65-119, 약 2.75-5.0 s |
| RGB sampling | observation/future 각각 20 frames |
| Encoder | frozen V-JEPA 2.1 ViT-L/16 384 |
| Encoder output | 10 tubelets x 24 x 24 x 1024 |
| Cached feature | fixed 2x2 pooling 후 `[670,10,12,12,1024]`, float16 |
| Split | condition-connected main split, 470/98/102 conditions |

RGB frame `i`의 실제 capture time은 `(i + 1) / 24`초다. 두 source frame을 한 tubelet으로 묶고 그 두
capture time의 평균 시각에서 가장 가까운 72 Hz dense row를 target으로 사용했다. 전체 cache의 최대
시간 정렬 오차는 `0.006944 s`, 즉 physics step 절반이다.

Query decoder 입력에는 RGB-derived V-JEPA feature만 들어간다. Mass, CoM, route, dense log, safety label은
encoder 입력이나 decoder memory로 제공하지 않는다.

## 3. 물리 target 계약

### Static queries

- Load mass: kg
- Load lateral CoM offset: m, load-local lateral axis

현재 shard에는 mass 5단계 `240/360/480/600/720 kg`와 lateral CoM 3단계
`-0.25/0.0/+0.25 m`가 있다. 계획에 있던 load-height intervention은 이 데이터에 실제로 존재하지 않으므로
height query는 만들지 않았다.

### Observed and future state queries

각 구간의 10 tubelet 시각마다 다음 18채널을 예측한다.

| Group | Channels | Unit/frame |
|---|---|---|
| Attitude | roll, pitch | degree |
| IMU angular velocity | x, y, z | rad/s, sensor-native frame |
| Linear velocity | x, y, z | m/s, chassis-yaw-local frame |
| IMU linear acceleration | x, y, z | m/s2, sensor-native frame |
| System CoM | x, y, z | m, chassis-yaw-local origin/frame |
| ZMP | x, y | m, chassis-yaw-local origin/frame |
| Stability margin | outward ZMP, outward CoM | m, turn-direction-invariant signed margin |

World position/vector인 velocity, CoM, ZMP와 contact point는 매 시각 forklift yaw를 사용해 chassis-local로
변환했다. IMU 채널은 logger가 기록한 sensor-native frame을 유지한다.

ZMP finite-difference가 전도 후 발산하는 구간을 target으로 학습하지 않도록
`outward_lateral_margin_valid`가 참인 row에서만 ZMP 좌표와 두 margin을 supervise한다. 결측·right-censored
label은 mask로 loss와 해당 metric에서 제외한다.

### Contact and support queries

다섯 contact body 각각에 대해 active 여부와 chassis-local `(x,y)`를 예측한다.

- `back_wheel`
- `roller_back_left`, `roller_back_right`
- `roller_front_left`, `roller_front_right`

예측 active point의 convex hull로 support polygon을 재구성하고, area와 centroid 오차를 계산한다.

### Future summary query

- Future minimum outward ZMP margin
- Future minimum outward CoM margin
- Future maximum absolute roll
- Future minimum contact count
- Future tip-over probability

## 4. Query decoder

```text
frozen V-JEPA memory [B,10,12,12,1024]
-> Linear(1024,128) + learned time/space positions
-> 123 learned physical query tokens
-> 2-layer, 4-head Transformer decoder cross-attention
-> unit-specific regression/presence heads
```

Query 구성은 static 2, observed state 10, future state 10, observed contacts 50, future contacts 50,
future summary 1개다. Trainable parameter는 `570,928`, foundation-model trainable parameter는 `0`이다.
모든 continuous channel은 train split 통계로만 normalize하며 checkpoint는 validation total normalized loss로
선택한다. Seeds는 17, 29, 43이다.

## 5. Shard 01 핵심 결과

아래 값은 3 seeds test mean +/- sample SD다.

### Hidden load properties

| Query | MAE | Discrete level accuracy | Test pair direction |
|---|---:|---:|---:|
| Load mass | `14.44 +/- 1.17 kg` | `0.987 +/- 0.011` | `0.994 +/- 0.010`, 60 pairs |
| Lateral CoM | `0.0331 +/- 0.0055 m` | `0.951 +/- 0.017` | `0.961 +/- 0.017`, 34 pairs |

Pair direction은 Gold만 사용하지 않는다. Test의 mass/CoM intervention pair 94개를 quality tier와 outcome에
상관없이 사용해 `sign(pred_B - pred_A) == sign(target_B - target_A)`를 평가했다. 이 task에서는 두 endpoint가
서로 다른 outcome일 필요가 없으므로 Reject/Silver/Bronze 데이터도 버리지 않는다.

### Observed physical state

| Query | Model MAE | Train-mean baseline MAE |
|---|---:|---:|
| Roll | `0.0089 deg` | `0.0184 deg` |
| Yaw angular velocity | `0.0265 rad/s` | `0.1529 rad/s` |
| Body velocity x / y | `0.115 / 0.024 m/s` | `0.926 / 0.133 m/s` |
| IMU acceleration x / y | `0.446 / 0.710 m/s2` | `1.239 / 1.712 m/s2` |
| System CoM x / y / z | `0.0106 / 0.0041 / 0.0092 m` | `0.0444 / 0.0122 / 0.0537 m` |
| ZMP x / y | `0.0181 / 0.0333 m` | `0.0553 / 0.0792 m` |
| ZMP / CoM outward margin | `0.0544 / 0.0430 m` | `0.2115 / 0.1759 m` |

Observed roll은 시작 구간에서 거의 0이라 Spearman이 낮고 absolute error도 작다. 반면 body velocity x,
ZMP margin, CoM 위치 등은 train-mean baseline보다 명확히 좋아졌다.

### Future physical state

| Query | Model MAE | Train-mean baseline MAE | Spearman |
|---|---:|---:|---:|
| Roll | `3.080 deg` | `15.276 deg` | `0.914` |
| Body velocity x / y | `0.186 / 0.110 m/s` | `1.012 / 0.599 m/s` | `0.967 / 0.971` |
| System CoM x / y / z | `0.012 / 0.049 / 0.036 m` | `0.045 / 0.248 / 0.199 m` | `0.938 / 0.932 / 0.936` |
| Valid ZMP x / y | `0.063 / 0.093 m` | `0.094 / 0.292 m` | `0.725 / 0.903` |
| ZMP / CoM outward margin | `0.102 / 0.047 m` | `0.198 / 0.107 m` | `0.813 / 0.816` |

### Support polygon

| Query | Result |
|---|---:|
| Observed active contact position MAE | `0.0191 +/- 0.0021 m` |
| Observed polygon area MAE | `0.0359 +/- 0.0062 m2` |
| Observed polygon centroid error | `0.0187 +/- 0.0019 m` |
| Future contact F1 | `0.9753 +/- 0.0016` |
| Future active contact position MAE | `0.0258 +/- 0.0018 m` |
| Future valid polygon coverage | `0.9847 +/- 0.0072` |
| Future polygon area MAE | `0.0923 +/- 0.0104 m2` |
| Future polygon centroid error | `0.0500 +/- 0.0036 m` |

Observed contact F1 `1.0`은 성과로 내세우지 않는다. Sampled observation 시각의 5,100 contact label이 모두
active라서 constant-positive predictor도 얻을 수 있는 값이다. 위치와 polygon geometry 오차가 더 유효한
observed metric이다.

### Future stability summary and outcome

| Query | MAE | Train-mean MAE | Spearman |
|---|---:|---:|---:|
| Minimum ZMP outward margin | `0.0457 +/- 0.0053 m` | `0.1019 m` | `0.904` |
| Minimum CoM outward margin | `0.0129 +/- 0.0005 m` | `0.0539 m` | `0.933` |
| Maximum absolute roll | `4.55 +/- 0.55 deg` | `36.37 deg` | `0.939` |
| Minimum contact count | `0.301 +/- 0.031` | `1.462` | `0.959` |

Future tip AUROC는 `0.9996 +/- 0.0004`, balanced accuracy는 `0.9846 +/- 0.0022`다. 그러나 binary
결과는 grid 내 궤적 regime을 분리하기 쉬울 수 있으므로 단독 결론으로 사용하지 않는다.

Gold test 10 pairs에서 future minimum ZMP-margin ordering은 seed 평균 `0.900`, CoM-margin ordering은
`0.867`이었다. Silver/Bronze/Reject까지 나누어 결과에 보존했으며 censored tier는 exploratory로만 본다.

## 6. Feature intervention controls

학습된 checkpoint를 바꾸지 않고 test-time feature 구조만 개입했다. 이 수치는 retrained baseline이 아니라
trained decoder가 어떤 정보에 의존하는지 보는 out-of-distribution diagnostic이다.

| Input intervention | Mass MAE kg | CoM MAE m | Future min-ZMP MAE m | Tip AUROC | Mass pair dir. | CoM pair dir. |
|---|---:|---:|---:|---:|---:|---:|
| Identity | **14.44** | **0.033** | **0.046** | **1.000** | **0.994** | **0.961** |
| Zero encoder values | 184.59 | 0.204 | 0.116 | 0.500 | 0.500 | 0.500 |
| Conditions permuted within split | 182.77 | 0.213 | 0.148 | 0.588 | 0.506 | 0.657 |
| Spatial mean repeated | 156.19 | 0.186 | 0.109 | 0.814 | 0.872 | 0.500 |
| Temporal mean repeated | 188.74 | 0.192 | 0.137 | 0.622 | 0.844 | 0.549 |
| Temporal order reversed | 14.68 | 0.034 | 0.046 | 1.000 | 0.994 | 0.961 |

해석은 세 가지다.

1. Zero와 condition permutation에서 대부분의 물리 query가 붕괴하므로 query/position prior나 split label
   빈도만으로 얻은 결과는 아니다.
2. Spatial mean은 특히 lateral CoM pair를 chance로 떨어뜨린다. 물체와 차량의 공간적 배치가 필요하다.
3. Temporal mean은 크게 해롭지만 temporal reverse는 거의 영향이 없다. 현재 probe는 여러 시각의 변화와
   분포를 사용하지만, 시간의 화살표나 causal order를 활용한다는 증거는 없다.

## 7. 현재 주장 가능한 것

1. Shard 01에서 frozen V-JEPA 2.1 feature에는 hidden mass와 lateral CoM을 supervised query로 복원할 수
   있는 정보가 있다.
2. 동일 feature에서 현재와 미래의 CoM, ZMP, velocity, roll 및 contact geometry가 train-mean보다 훨씬
   낮은 오차로 접근 가능하다.
3. Gold-only binary table과 달리 physical-property query는 test intervention pair 94개를 활용할 수 있다.
4. 공간 구조와 temporal variation을 제거하면 결과가 크게 나빠진다.

## 8. 아직 주장하면 안 되는 것

1. V-JEPA가 zero-shot으로 ZMP를 계산한다. 여기서는 57만 parameter query decoder를 physics label로
   supervised training했다.
2. Temporal Transformer가 causal dynamics를 학습했다. Reverse-time control에 사실상 불변이다.
3. Tip AUROC 1.0에 가깝다는 사실만으로 보편적인 물리 이해가 증명된다.
4. Shard 01 결과가 다른 camera, texture, forklift, command distribution으로 일반화된다.
5. Load height를 추론할 수 있다. 현재 dataset에는 height intervention이 없다.

특히 part 01을 보며 architecture와 target을 정했으므로 이 결과는 exploratory다. 같은 config와 metric을
동결한 뒤 untouched shard 02/03에서 재검증해야 confirmatory claim이 된다.

## 9. 재현 명령

```bash
cd ~/isaac_ws/safety_probing

# 1. Frozen spatial feature cache
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_spatial_features \
  --shard part_01 --batch-size 4

# 2. Dense-log physical target cache
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_targets \
  --shard part_01

# 3. Three-seed physical Safety Query training/evaluation
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_probe \
  --output-root outputs/safety_query/part_01

# 4. Checkpoint-preserving feature intervention diagnostics
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.safety_query_controls \
  --checkpoint-root outputs/safety_query/part_01
```

## 10. 산출물

- Experiment contract: `configs/experiments/vjepa_safety_query.json`
- Spatial cache: `artifacts/features/vjepa_2_1_part_01_observation_spatial12.pt`
- Physical target cache: `artifacts/targets/safety_query_part_01.pt`
- Main result: `outputs/safety_query/part_01/result.json`
- Control result: `outputs/safety_query/part_01/controls.json`
- Seed checkpoints: `outputs/safety_query/part_01/seed_{0017,0029,0043}.pt`

Feature/target cache에는 condition order, preprocessing hash, manifest hash, model code revision과 experiment hash가
기록된다. Identity control은 세 checkpoint 모두 저장된 prediction과 최대 절대차 `0.0`으로 일치했다.
