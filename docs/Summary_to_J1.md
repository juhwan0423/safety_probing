# Summary to J1

> 후속 J2와 temporal controls까지 완료했다. 최종 비교는 `docs/Summary_to_J2.md`를 참조한다.

## 1. 한 줄 결론

V-JEPA 2.1 frozen observation representation의 J1 linear probe를 shard 01에서 완료했다.

> 일반 episode의 미래 tip 분류는 test AUROC **0.960 +/- 0.008**이지만, 같은 command의 Gold
> counterfactual pair에서는 **0.50 +/- 0.10**으로 chance 수준이다.

따라서 현재 결과는 V-JEPA 표현에 속도와 조향을 포함한 coarse future-risk 정보가 선형적으로 존재함을
보여준다. 그러나 논문의 핵심인 hidden payload intervention의 A/B 미래 차이가 선형적으로 읽힌다는
증거는 아직 없다.

## 2. J1이 답하는 질문

J0는 실제 미래 candidate A/B를 evaluator에게 제공하는 closed-set retrieval이었다. J1은 candidate
future를 전혀 제공하지 않는다.

```text
observation RGB only
-> frozen V-JEPA 2.1 encoder
-> fixed mean pooling
-> trainable linear head
-> future tip score / future ZMP-margin estimate
```

J1의 질문은 다음과 같다.

1. 관측 prefix만으로 미래 tip outcome이 frozen representation에 선형적으로 존재하는가?
2. 미래 최소 outward ZMP margin의 순서가 선형적으로 복원되는가?
3. 같은 command에서 hidden mass/CoM intervention만 다른 Gold pair도 올바르게 rank하는가?

세 번째가 본 연구의 primary causal metric이다. 첫 번째 AUROC만 높다고 counterfactual safety reasoning을
주장하지 않는다.

## 3. 고정된 입력과 모델 계약

| 항목 | 값 |
|---|---|
| Encoder | V-JEPA 2.1 ViT-L/16 384 |
| Repository revision | `204698b45b3712590f06245fbfba32d3be539812` |
| Camera | `main_fixed_northwest` |
| Observation | source frame 0-64, 0.0-2.7083 s |
| Source RGB | 65 frames at 24 FPS |
| Encoder sample | 20 uniformly sampled frames |
| Encoder output | 10 temporal tubelets x 576 spatial tokens x 1024 |
| Cached representation | spatial mean, `[N,10,1024]` |
| Linear input | fixed temporal mean, `[N,1024]` |
| Encoder trainable parameters | 0 |
| Predictor used | no |
| Probe | one linear layer per task |

샘플링 frame index는 다음과 같다.

```text
[0, 3, 7, 10, 13, 17, 20, 24, 27, 30,
 34, 37, 40, 44, 47, 51, 54, 57, 61, 64]
```

모든 feature normalization은 train condition 470개에서만 계산했다. Binary class weight도 train의
positive/negative count만 사용했고 decision threshold는 validation balanced accuracy로 고정했다.

## 4. 평가 데이터

동일 physical condition의 반복 run은 `condition_id`별 deterministic canonical run 하나만 사용했다.

| Split | Canonical runs | Binary eligible | Tip | Recovered | Censored | Margin target |
|---|---:|---:|---:|---:|---:|---:|
| Train | 470 | 461 | 254 | 207 | 9 | 470 |
| Validation | 98 | 98 | 58 | 40 | 0 | 98 |
| Test | 102 | 99 | 60 | 39 | 3 | 102 |

`right_censored` 12개는 binary loss와 metric에서 제외했다. ZMP margin regression에는 target이 존재하는
전체 670개를 사용했다.

Gold pair 수는 train 31, validation 17, test 10이다. 모든 Gold endpoint의 pair label과 canonical clip
label 일치를 전수 확인했고 mismatch는 0개였다. Shared physical endpoint와 left/right mirror component가
split을 넘는 leakage도 없다.

## 5. Feature cache 실행 결과

| 항목 | 값 |
|---|---:|
| Conditions | 670 |
| Cache shape | `[670,10,1024]` |
| Cache dtype | float16 |
| Non-finite values | 0 |
| Zero-variance pooled dimensions | 0 |
| Extraction time | 219.714 s |
| Peak allocated GPU memory | 3,493,543,936 bytes |
| Cache file size | 약 14 MiB |

Feature extraction이 끝난 뒤 probe는 cache만 읽으므로 encoder를 다시 load하지 않는다.

## 6. J1 classification 결과

세 seed `17, 29, 43`의 평균과 sample standard deviation이다.

| Test metric | J1 linear probe |
|---|---:|
| AUROC | **0.9603 +/- 0.0080** |
| AUPRC | **0.9676 +/- 0.0044** |
| Balanced accuracy | **0.8968 +/- 0.0200** |
| F1 | **0.9232 +/- 0.0187** |
| Brier score | **0.0724 +/- 0.0146** |
| ECE, 10 bins | **0.0570 +/- 0.0132** |

Seed별 test AUROC는 `0.9530, 0.9590, 0.9688`이다. AUROC와 AUPRC는 threshold-free이고, 나머지
classification metric은 validation에서 선택한 threshold를 test에 그대로 적용했다.

## 7. 핵심 Gold pair 결과

| Metric | Train | Validation | Test |
|---|---:|---:|---:|
| Tip classifier pair accuracy | 1.000 | 0.941 | **0.400 / 0.500 / 0.600** |
| ZMP regressor pair accuracy, mean | 0.731 | 0.686 | **0.467** |

Test Gold pair는 10개뿐이다. 세 classifier seed는 각각 4, 5, 6개를 맞혔고 평균은 **0.50 +/- 0.10**이다.
Seed 변화보다 어떤 pair를 틀리는지가 상당히 일관적이었다.

Classifier의 test subgroup은 다음 범위였다.

- Left: seed별 `0.2, 0.4, 0.4`
- Right: `0.6, 0.6, 0.8`
- Load mass: `0.429, 0.571, 0.714`
- CoM shift: 세 seed 모두 `0.333`

표본이 left 5, right 5, mass 7, CoM 3이므로 subgroup 차이를 일반화하면 안 된다. 특히 CoM shift
3쌍은 결론을 내리기에 너무 작다.

Validation 16/17과 test 평균 5/10의 큰 차이는 validation Gold pair accuracy로 checkpoint를 고른 영향,
작은 pair 표본, held-out action component의 차이를 함께 반영할 수 있다. Validation 수치를 모델의
일반화 성능으로 인용하지 않는다.

## 8. ZMP margin regression

| Test metric | 결과 |
|---|---:|
| MAE | **0.0705 +/- 0.0030 m** |
| RMSE | **0.0895 +/- 0.0036 m** |
| Spearman | **0.7556 +/- 0.0151** |
| Gold pair risk accuracy | **0.4667 +/- 0.0577** |

전체 run 사이의 margin ordering은 어느 정도 복원되지만, matched counterfactual pair의 작은 차이는
복원하지 못한다. 또한 stable과 tip 모두 future ZMP margin이 음수가 될 수 있으므로 margin sign을 binary
tip label로 사용하지 않는다.

## 9. Dataset-severity control

J1의 높은 global AUROC가 pair 결과와 왜 다른지 확인하기 위해 train label prevalence만 사용하는
metadata diagnostic을 함께 계산했다. 이 control은 영상 모델과 공정한 경쟁 모델이 아니라 dataset
confounding을 측정하는 도구다.

| Train-only prevalence control | Test AUROC | Gold pair accuracy |
|---|---:|---:|
| Speed only | 0.8654 | 0.500, all ties |
| Direction + speed + steering | 0.8942 | 0.500, all ties |
| Frozen V-JEPA J1 | 0.9603 +/- 0.0080 | 0.500 +/- 0.100 |

속도만 알아도 test AUROC가 0.865다. V-JEPA linear probe는 이보다 높으므로 RGB dynamics에서 추가
신호를 읽고 있다. 하지만 same-command pair에서는 speed, steering, direction이 완전히 같기 때문에 이
coarse signal은 쓸 수 없고 성능이 chance로 내려간다.

## 10. J0와 함께 보는 해석

| Stage | 미래 candidate 제공 | Shard 01 결과 | 주 해석 |
|---|---|---:|---|
| J0 native retrieval | yes | 66.7% query accuracy | boundary identity shortcut 지배 |
| J1 standalone classification | no | AUROC 0.960 | coarse risk가 선형적으로 accessible |
| J1 Gold pair ranking | no | 0.50 pair accuracy | hidden intervention contrast 미검증 |

J0와 J1은 다른 task이므로 숫자를 직접 우열 비교하지 않는다. J0의 66.7%는 A/B 실제 미래가 주어진
multiple choice이고, J1 AUROC는 관측 prefix만으로 전체 run을 rank한 값이다. 현재 가장 논문 질문에
가까운 숫자는 J1 Gold pair accuracy다.

## 11. 현재 주장 가능한 것

1. Frozen V-JEPA 2.1 observation feature cache와 linear probing pipeline이 end-to-end로 동작한다.
2. 일반 run의 미래 tip outcome과 ZMP margin에 관한 coarse information이 representation에 선형적으로
   존재한다.
3. 이 global signal의 상당 부분은 speed와 maneuver severity로 설명된다.
4. 현재 J1은 same-command hidden mass/CoM counterfactual을 test Gold pair에서 chance 이상으로 분리하지
   못했다.

## 12. 아직 주장하면 안 되는 것

1. AUROC 0.96이 hidden load mass 또는 CoM reasoning을 증명한다.
2. Validation Gold 94.1%가 일반화된 causal pair 성능이다.
3. Test 10 pair만으로 V-JEPA가 counterfactual 정보를 전혀 갖지 않는다고 결론낸다.
4. Shard 01 결과를 전체 1,008-pair dataset 결과로 간주한다.
5. Margin regression Spearman 0.756이 tip classification criterion을 대체한다.

## 13. 다음 단계 J2

J2는 같은 cache의 `[N,10,1024]` temporal token을 사용한다. feature extraction을 다시 할 필요가 없다.

```text
[B,10,1024]
-> trainable projection
-> learned temporal position embedding
-> 2-layer, 4-head Transformer
-> classification and margin heads
```

J2의 success criterion은 global AUROC 상승이 아니라 다음 순서다.

1. Test Gold pair accuracy가 J1의 같은 10 pair에서 일관되게 개선되는가?
2. Untouched shard 02/03에서 개선이 반복되는가?
3. Left-to-right와 right-to-left mirror generalization에서도 유지되는가?
4. Last-frame, optical-flow, route-severity baseline을 넘는가?

J2도 pair accuracy가 chance라면 현재 observation horizon에서 hidden intervention effect가 시각적으로
충분히 드러나지 않거나, pretrained representation이 그 작은 동역학 차이를 보존하지 않는다는 두
가능성을 분리해야 한다. Observation horizon ablation과 optical-flow baseline이 그 구분에 필요하다.

## 14. 재현 커맨드

```bash
cd ~/isaac_ws/safety_probing

# 1. Frozen feature cache. 기존 contract와 일치하는 cache가 있으면 재사용한다.
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_features \
  --shard part_01 --batch-size 8 \
  --output artifacts/features/vjepa_2_1_part_01_observation.pt

# 2. Three-seed J1 probe.
PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_linear_probe \
  --feature-cache artifacts/features/vjepa_2_1_part_01_observation.pt \
  --output-root outputs/vjepa_linear_probe/part_01

# 3. Unit tests.
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## 15. 주요 산출물

- J1 config: `configs/experiments/vjepa_linear_probe.json`
- Feature cache: `artifacts/features/vjepa_2_1_part_01_observation.pt`
- Cache sidecar: `artifacts/features/vjepa_2_1_part_01_observation.json`
- Aggregate result: `outputs/vjepa_linear_probe/part_01/result.json`
- Seed checkpoints and predictions: `outputs/vjepa_linear_probe/part_01/seed_*.pt`
- Feature extractor: `src/safety_probing/vjepa_features.py`
- Probe runner: `src/safety_probing/vjepa_linear_probe.py`

`artifacts/features`와 `outputs`는 재생성 가능한 대용량/실험 산출물이므로 Git에서 제외한다. Config,
runner, tests, summary 문서가 재현 계약이다.
