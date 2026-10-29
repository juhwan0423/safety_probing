# J1/J2/SQ Frozen Encoder Probing Protocol

## 1. 목적

Frozen visual encoder가 forklift의 미래 횡방향 안전성을 표현하고 있는지 작은 temporal transformer로
측정한다. Encoder fine-tuning 성능이 아니라 **고정된 representation의 linearly/temporally accessible
safety information**을 평가한다.

이 문서는 J1 linear probe, J2 transformer probe와 physical Safety Query(SQ)를 정의한다. Released
V-JEPA predictor 자체를 학습 없이 평가하는 J0 native future retrieval은
`MODEL_EXPERIMENT_PLAN_KO.md`의 별도 protocol을 따른다.

## 2. Data Unit

### Run-level task

- 입력: 한 run의 `main_fixed_northwest` RGB clip
- label: 미래 tip-over, 미래 최소 outward ZMP margin
- 중복 제거: `canonical_runs.jsonl`에서 동일 `condition_id`당 deterministic representative 하나만 사용
- run-level task는 pair quality tier로 filter하지 않는다. Tier는 pair context의 속성이기 때문이다.

### Pair-level task

- 입력: 동일 command의 A/B clip embedding
- label: 어느 run이 더 위험한가, 또는 `risk_B - risk_A`
- primary subset: `gold`
- secondary subset: `silver`, `bronze`
- 제외: `reject`

위 filter는 outcome-risk ordering task에만 적용한다. Mass나 CoM의 실제 intervention delta를 묻는 SQ
property task는 endpoint target이 존재하면 quality tier 및 stable/tip 조합과 무관하게 모든 pair를 사용한다.
즉 `reject`는 binary near-boundary benchmark에는 부적합해도 물성 변화 방향 supervision에는 유효하다.

## 3. Leakage-Safe Splits

기존 pair-level split에는 같은 physical endpoint가 인접 pair를 통해 train/test에 반복되는 문제가 있다.
예를 들어 `240→360 kg`의 B와 `360→480 kg`의 A는 동일 route/seed/CoM의 360kg 상태다.

새 `leakage_safe_splits.json`은 다음 순서로 만든다.

1. 각 pair의 A/B를 realized `(route, seed, mass, CoM, height, geometry)` endpoint로 변환
2. endpoint를 공유하는 pair를 graph edge로 연결
3. main protocol에서는 left/right mirror counterpart도 연결
4. connected component 전체를 train, validation, test 중 하나에만 배정

Main split 목표는 70/15/15이며 현재 full-grid 배정은 710/148/150 pairs다. Validation/test는 각각
slow 1, medium 2, fast 1 component를 포함하도록 action strata를 맞춘다. Cross-direction protocol은
left-to-right와 right-to-left를 각각 따로 제공한다.

## 4. Primary Temporal Window

| 항목 | 값 |
|---|---|
| Camera | `main_fixed_northwest` |
| Observation | 0.0-2.7083 s, 65 source frames |
| Source FPS | 24 |
| Probe sampling | 20 uniformly spaced frames |
| Input tokens before encoder pooling | 20 frames |
| Future target | 2.7083-5.0 s |

Steering pulse는 2.0초에 시작한다. 65-frame boundary의 마지막 영상 timestamp는 약 2.667초이므로
모델은 약 17개의 조향 구간 RGB frame과 명확한 초기 yaw response를 본다. 현재 데이터의 가장 빠른
definitive event는 2.958초이므로 event 자체는 observation에 들어가지 않는다. 57-frame 계약은 마지막
timestamp가 약 2.333초라 yaw 변화가 2-3도에 그쳐 LTX가 직진 continuation을 생성하는 문제가 있어
폐기했다.
현재 합쳐진 데이터의 가장 빠른 post-hoc failure는 2.958초이므로 최소 약 0.583초의 prediction lead
time이 있다. 57프레임 경계는 LTX temporal VAE의 `8n+1` 조건도 만족한다.

추가 ablation:

- pre-turn: 0.0-2.0 s
- shorter early-turn: 0.0-2.25 s
- later observation: 0.0-2.75 s
- top-view-only와 two-camera fusion

Observation window가 failure time을 넘는 sample은 해당 protocol에서 제외하거나 별도 late-prediction
평가로 분리한다.

## 5. Frozen Encoder Contract

Encoder 선택은 별도 benchmark 표에서 고정하되 모든 encoder에 다음 계약을 적용한다.

1. pretrained checkpoint와 preprocessing version을 명시한다.
2. encoder parameter와 normalization statistics를 freeze한다.
3. frame/clip embedding을 먼저 cache한다.
4. probe 학습 중 RGB augmentation으로 encoder representation을 바꾸지 않는다.
5. encoder별 embedding dimension 차이는 trainable input projection 하나로 맞춘다.

Cache record 권장 schema:

```text
run_id
camera_name
frame_indices[T]
timestamps_s[T]
features[T, D] float16
encoder_name
checkpoint_revision
preprocess_hash
```

첫 비교는 image encoder frame features를 동일한 temporal probe에 넣는 방식으로 시작하고, 이후 native
video encoder를 별도 family로 비교한다. 구체 checkpoint는 다운로드 전에 license, resolution,
preprocessing을 함께 검토해 고정한다.

## 6. Temporal Probe

기본 probe는 작게 유지한다.

```text
cached frozen features [B,T,D]
→ trainable projection
→ learned temporal position embedding
→ 2-layer Transformer Encoder, 4 heads
→ masked temporal pooling
→ task heads
```

- Encoder: frozen
- Probe layers: 2
- Attention heads: 4
- Dropout: 0.1
- Model selection: validation split only
- Early stopping metric: task별 primary metric
- Random seeds: 최소 3개

Linear pooling probe와 temporal mean-pooling MLP를 반드시 하위 baseline으로 둔다. Transformer가 이보다
좋지 않으면 temporal reasoning 주장을 하지 않는다.

## 7. Tasks and Targets

### T1. Future tip-over classification

- Target: observation end 이후 `observed_tip` event
- `confirmed_recovered`는 negative, `right_censored`는 binary loss/metric에서 제외
- Metrics: AUROC, AUPRC, balanced accuracy, F1, ECE
- Report: overall, direction, speed, steering, quality tier

### T2. Future minimum outward ZMP margin

- Target: `[2.7083, 5.0]`에서 `outward_lateral_margin_valid=true`인 row의 최소값
- Metrics: MAE, RMSE, Spearman correlation
- Missing target: valid row가 하나도 없으면 regression loss에서 제외하고 count 보고
- `margin < 0`을 tip-over threshold로 사용하지 않음. Threshold calibration이 필요하면 train split만 사용
- Secondary target: future minimum projected-CoM outward margin

### T3. Counterfactual pairwise risk

- Gold target: tip run이 stable run보다 위험
- Same-outcome target: 미래 최소 outward margin이 작은 run이 위험
- Metrics: pairwise accuracy, rank correlation, tie-aware accuracy

### T4. Mirror generalization

- left-to-right와 right-to-left protocol 별도 학습/평가
- Raw roll sign 대신 direction-invariant risk가 유지되는지 측정

### T5. Censor-aware time-to-event

- Target: `followup_time_s`, `event_observed`
- Metrics: concordance index와 time-dependent AUROC
- 5초 종료 시 unresolved인 run을 stable negative로 강제하지 않음

### T6. Physical Safety Query

- V-JEPA input: frozen observation token grid `[B,10,12,12,1024]`
- LTX oracle input: frozen real-future VAE latent `[B,7,10,18,128]`
- LTX rollout input: frozen generated-future latent with the same geometry
- Static targets: load mass, lateral load CoM
- Time-indexed targets: roll/pitch, IMU, body velocity, system CoM, ZMP, outward margins
- Contact targets: five contact presences and chassis-local positions
- Derived geometry: support polygon area and centroid
- Future summaries: minimum ZMP/CoM margin, maximum absolute roll, minimum contact count
- Metrics: physical-unit MAE/RMSE, Spearman, contact F1/position error, polygon geometry error
- Pair metrics: all eligible intervention pairs에서 mass/CoM delta direction, tier별 future-margin ordering

ZMP 좌표와 margin은 `outward_lateral_margin_valid=true`인 dense row만 supervise한다. Missing target과
right-censored binary label은 mask하고, observed contact F1처럼 label variation이 없는 metric은 primary
evidence로 사용하지 않는다. 상세 query/coordinate 계약은 `Summary_to_Safety_Query.md`를 따른다.

LTX query는 real future latent에서만 학습하고 generated latent에는 freeze해 적용한다. Oracle은 prediction이
아니라 representation/readout ceiling으로 보고하며, LTX generated output과의 gap을 rollout information
loss로 측정한다. LTX 전용 계약은 `LTX_SAFETY_QUERY_PROTOCOL_KO.md`, 결과는
`Summary_to_LTX_Safety_Query.md`를 따른다.

## 8. Training Rules

- 모든 normalization은 train split에서만 계산
- class weight 또는 sampler는 train label로만 결정
- pair A/B는 같은 mini-batch에 둘 수 있지만 split은 반드시 동일
- checkpoint selection에 test 사용 금지
- hyperparameter budget을 encoder마다 동일하게 유지
- run-level task에서 duplicate `condition_id`를 반복 sampling하지 않음
- `reject` 포함/제외는 사전에 정한 ablation으로만 변경
- Physical-property delta는 사전에 정의한 all-tier protocol을 사용하고 outcome-risk Gold metric과 섞지 않음
- SQ는 zero/permuted/spatial-mean/temporal-mean/reverse-time feature intervention을 함께 보고

## 9. Reporting Table

각 encoder에 대해 다음을 한 행으로 보고한다.

| Encoder | Params | Input | Probe | AUROC | AUPRC | Margin MAE | Pair Acc | L→R | R→L |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|

추가 표에는 quality tier별 pair accuracy와 observation horizon별 성능을 둔다. 평균과 표준편차는 probe
seed 3개 이상에서 계산한다.

## 10. 다음 구현 순서

1. sampled RGB pixel audit와 canonical run manifest 생성
2. frozen encoder registry와 preprocessing hash 구현
3. feature cache writer/reader 구현
4. mean-pool MLP와 linear baseline
5. temporal transformer probe
6. main 및 cross-direction evaluation runner
7. spatial Safety Query target/cache/query decoder 및 intervention controls
8. LTX real-future oracle과 generated-rollout frozen-query comparison
