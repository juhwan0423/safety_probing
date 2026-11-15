# Summary to J2

## 1. 결론

Shard 01의 frozen V-JEPA 2.1 feature로 J2 temporal Transformer probe를 완료했다.

> J2는 J1 linear probe보다 global AUROC와 Gold pair accuracy가 높았다. 그러나 tubelet 순서를
> 뒤집어도 결과가 그대로였고, 5배 작은 mean-pool MLP가 J2와 같은 pair accuracy 및 더 높은 AUROC를
> 기록했다. 따라서 현재 향상은 temporal-order reasoning이 아니라 nonlinear readout capacity의
> 효과로 해석해야 한다.

핵심 결과는 다음과 같다.

| Probe | Params/task | Test AUROC | Gold pair acc | Margin MAE | Margin pair acc |
|---|---:|---:|---:|---:|---:|
| J1 linear | 1,025 | 0.9603 +/- 0.0080 | 0.500 +/- 0.100 | 0.0705 +/- 0.0030 m | 0.467 +/- 0.058 |
| Mean-pool MLP control | 263,169 | **0.9778 +/- 0.0031** | **0.667 +/- 0.058** | **0.0701 +/- 0.0012 m** | **0.633 +/- 0.115** |
| J2 Transformer | 1,319,937 | 0.9721 +/- 0.0020 | **0.667 +/- 0.058** | 0.0717 +/- 0.0013 m | 0.433 +/- 0.115 |
| J2, reversed order | same | 0.9718 +/- 0.0019 | **0.667 +/- 0.058** | 0.0716 +/- 0.0011 m | 0.433 +/- 0.115 |

현재 가장 좋은 frozen-feature readout은 Transformer가 아니라 mean-pool MLP다. 하지만 Gold test가
10 pair뿐이라 `7/10`을 definitive counterfactual result로 주장할 수는 없다.

## 2. J2 연구 질문

J1은 10개 temporal tubelet을 먼저 평균내고 linear head를 학습했다. J2는 tubelet 순서를 보존한 채
작은 Transformer를 학습한다.

```text
frozen V-JEPA features [B,10,1024]
-> train-only channel normalization
-> 1024-to-256 projection
-> learned temporal position embedding
-> 2-layer, 4-head Transformer
-> temporal mean pooling
-> task head
```

J2가 답하려는 질문은 다음과 같다.

1. Frozen representation의 시간별 변화가 linear temporal mean보다 유용한가?
2. Temporal interaction이 hidden mass/CoM Gold pair를 더 잘 구분하는가?
3. 향상이 단순 nonlinear capacity가 아니라 실제 temporal order에 의존하는가?

세 번째 질문 때문에 reversed-order와 mean-pool MLP control을 사전에 포함했다.

## 3. 고정된 실험 계약

### Frozen input

| 항목 | 값 |
|---|---|
| Encoder | V-JEPA 2.1 ViT-L/16 384 |
| Encoder trainable params | 0 |
| Predictor | 사용하지 않음 |
| Observation | 0.0-2.7083 s, source frame 0-64 |
| Sampled RGB | 20 frames |
| Cached features | `[670,10,1024]` float16 |
| Spatial pooling | tubelet별 576 token fixed mean |
| Normalization | train runs x temporal tokens의 channel z-score |

Feature cache와 split, label, canonical condition은 J1과 정확히 같다. J2 때문에 encoder를 다시 실행하지
않았다.

### Transformer

| 항목 | 값 |
|---|---:|
| Layers | 2 |
| Attention heads | 4 |
| Model dimension | 256 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| Activation | GELU |
| Position embedding | learned, length 10 |
| Pooling | encoded temporal token mean |
| Trainable params/task | 1,319,937 |

### Mean-pool MLP control

```text
[B,10,1024]
-> fixed mean over time
-> Linear(1024,256)
-> GELU + Dropout + LayerNorm
-> task head
```

이 control은 temporal order를 볼 수 없고 trainable params는 task당 263,169개다.

### Training

| 항목 | 값 |
|---|---:|
| Seeds | 17, 29, 43 |
| Batch size | 64 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Max epochs | 300 |
| Patience | 40 |

Classifier는 train-only class weight를 사용한다. Checkpoint 선택은 validation Gold pair accuracy, AUROC,
loss 순서이며 test는 선택에 사용하지 않는다. Margin regressor는 validation RMSE로 선택한다.

## 4. 평가 표본

| Split | Canonical runs | Binary eligible | Margin targets | Gold pairs |
|---|---:|---:|---:|---:|
| Train | 470 | 461 | 470 | 31 |
| Validation | 98 | 98 | 98 | 17 |
| Test | 102 | 99 | 102 | 10 |

Right-censored 12 runs은 binary loss와 metric에서 제외했다. 모든 Gold endpoint와 canonical clip label의
일치 여부를 확인했고 mismatch는 0개다.

## 5. J2 classification 결과

| Test metric | J1 linear | J2 Transformer |
|---|---:|---:|
| AUROC | 0.9603 +/- 0.0080 | **0.9721 +/- 0.0020** |
| AUPRC | 0.9676 +/- 0.0044 | **0.9796 +/- 0.0015** |
| Balanced accuracy | 0.8968 +/- 0.0200 | 0.8998 +/- 0.0104 |
| F1 | 0.9232 +/- 0.0187 | 0.9224 +/- 0.0147 |
| Brier | **0.0724 +/- 0.0146** | 0.0759 +/- 0.0139 |
| ECE, 10 bins | **0.0570 +/- 0.0132** | 0.0811 +/- 0.0134 |

J2는 ranking metric인 AUROC/AUPRC는 올랐지만 thresholded F1과 calibration은 좋아지지 않았다.

Seed별 J2 AUROC는 다음과 같다.

```text
0.9705, 0.9714, 0.9744
```

## 6. Gold pair 결과

### Seed별 결과

| Probe | Seed 17 | Seed 29 | Seed 43 | Mean +/- SD |
|---|---:|---:|---:|---:|
| J1 linear | 4/10 | 5/10 | 6/10 | 0.500 +/- 0.100 |
| Mean-pool MLP | 7/10 | 7/10 | 6/10 | 0.667 +/- 0.058 |
| J2 Transformer | 7/10 | 6/10 | 7/10 | 0.667 +/- 0.058 |
| J2 reversed | 7/10 | 6/10 | 7/10 | 0.667 +/- 0.058 |

Seed-majority 결과는 J1 `5/10`, MLP `7/10`, J2 `7/10`이다.

### J2 subgroup

| Subgroup | Pair count | Seed별 accuracy |
|---|---:|---|
| Left | 5 | 0.6, 0.4, 0.6 |
| Right | 5 | 0.8, 0.8, 0.8 |
| Load mass | 7 | 0.857, 0.714, 0.857 |
| CoM shift | 3 | 0.333, 0.333, 0.333 |

향상은 주로 load-mass pair에서 나왔다. CoM shift는 세 seed 모두 1/3이며, left-turn도 안정적이지 않다.
각 subgroup 표본이 매우 작으므로 차이의 원인을 모델 능력으로 확정하지 않는다.

## 7. Temporal-order control

학습된 J2에 test-time으로 tubelet 순서를 `0..9`에서 `9..0`으로 뒤집었다. Retraining은 하지 않았다.

| Metric | Chronological | Reversed |
|---|---:|---:|
| AUROC | 0.97208 | 0.97179 |
| Gold pair accuracy | 0.6667 | 0.6667 |
| Margin MAE | 0.07171 m | 0.07155 m |
| Margin pair accuracy | 0.4333 | 0.4333 |

Chronological와 reversed classifier logits의 Pearson correlation은 seed별로 모두 `>0.99999`였다.
Mean absolute logit 변화도 `0.006-0.017`에 불과했다.

Position embedding이 구현상 존재하고 unit test에서 순서 변화에 반응함을 확인했다. 따라서 동일 결과는
구현이 순서를 못 보는 문제가 아니라, 학습된 probe가 이 task에서 position/order 신호를 거의 사용하지
않았다는 뜻이다.

## 8. Nonlinear-capacity control

Mean-pool MLP는 시간 순서를 제거한 뒤 nonlinear head만 학습한다.

| Metric | Mean-pool MLP | J2 Transformer | J2 - MLP |
|---|---:|---:|---:|
| AUROC | **0.97778** | 0.97208 | -0.00570 |
| AUPRC | **0.98394** | 0.97959 | -0.00435 |
| Gold pair accuracy | 0.6667 | 0.6667 | 0.0000 |
| Margin MAE | **0.07009 m** | 0.07171 m | +0.00162 m |
| Margin Spearman | **0.76855** | 0.74451 | -0.02404 |
| Margin pair accuracy | **0.6333** | 0.4333 | -0.2000 |

Transformer는 어떤 primary metric에서도 MLP를 명확히 넘지 못했다. 따라서 J1에서 J2로 오른 부분은
`temporal interaction`이 아니라 `nonlinear readout`으로 이름 붙이는 것이 정확하다.

MLP의 balanced accuracy는 `0.9353`, F1은 `0.9467`, Brier는 `0.0531`, ECE는 `0.0549`로 이
threshold/calibration 지표에서도 J2보다 좋았다.

## 9. 통계적 불확실성

Gold test는 10 pair뿐이다. 각 probe seed를 독립 pair 표본처럼 합쳐 `20/30`으로 검정하면 안 된다.
세 seed는 같은 10 pair와 같은 frozen feature를 반복 평가했기 때문이다.

Seed 또는 majority 기준 `7/10`에 대해:

- Wilson 95% interval: 약 `0.397-0.892`
- Chance 0.5에 대한 one-sided exact binomial p-value: `0.1719`

`6/10`은 interval `0.313-0.832`, one-sided p-value `0.3770`이다. 즉 shard 01 결과는 유망한 방향을
보여주지만 chance 초과를 통계적으로 확정하지 못한다. Untouched shard 02/03이 필요하다.

## 10. Margin regression

J2 margin regression은 J1이나 MLP보다 낫지 않았다.

| Test metric | J1 | Mean-pool MLP | J2 |
|---|---:|---:|---:|
| MAE | 0.0705 m | **0.0701 m** | 0.0717 m |
| RMSE | 0.0895 m | 0.0916 m | 0.0914 m |
| Spearman | 0.7556 | **0.7686** | 0.7445 |
| Gold pair risk acc | 0.4667 | **0.6333** | 0.4333 |

J2 classification score의 pair improvement가 물리적 ZMP margin prediction 향상으로 이어지지 않았다.
따라서 현재 classifier가 어떤 latent cue를 사용하는지 추가 control 없이 `stability margin reasoning`으로
부르면 안 된다.

## 11. 실행 성능

Feature cache는 이미 존재하므로 아래 시간은 probe 학습만 포함한다.

| Probe | Mean runtime/seed | Peak allocated GPU memory |
|---|---:|---:|
| J2 classifier + regressor | 2.07 s | 149,779,968 bytes |
| MLP classifier + regressor | 0.57 s | J2보다 작음 |

Encoder extraction은 J1 단계에서 한 번 수행했으며 219.714초가 걸렸다. J2는 encoder를 재실행하지 않는다.

## 12. 현재 주장 가능한 것

1. Frozen V-JEPA 2.1 representation에서 future safety outcome은 linear보다 nonlinear readout으로 더 잘
   접근된다.
2. Mean-pool MLP와 Transformer 모두 shard 01 Gold pair에서 seed 평균 0.667을 기록했다.
3. J2는 temporal order reversal에 사실상 불변이고 mean-pool MLP를 넘지 못했다.
4. 따라서 현재 데이터에서 temporal Transformer의 고유한 이점은 관찰되지 않았다.
5. CoM-shift pair는 여전히 제대로 구분되지 않는다.

## 13. 아직 주장하면 안 되는 것

1. J2가 temporal dynamics reasoning으로 hidden load를 판정했다.
2. `7/10`이 chance보다 유의하게 높다.
3. J2가 J1보다 좋아졌으므로 Transformer가 필요하다.
4. High AUROC가 counterfactual pair reasoning을 증명한다.
5. Classification improvement가 ZMP 또는 CoM stability margin을 학습했다는 뜻이다.

## 14. 다음 실험 우선순위

### Priority 1: untouched shard validation

현재 config와 metric을 바꾸지 않고 shard 02/03을 cache한 뒤 confirmatory test를 수행한다. Shard 01을
보고 architecture나 threshold를 더 조정하면 part 01은 계속 exploratory로 남는다.

### Priority 2: observation-horizon ablation

- Pre-turn: 0.0-2.0 s
- Early turn: 0.0-2.25 s
- Current: 0.0-2.7083 s

MLP가 current horizon에서만 좋아진다면 초기 yaw/roll response의 aggregate cue를 읽는 것이다. Pre-turn에도
높으면 scene 또는 route shortcut을 의심해야 한다.

### Priority 3: motion baselines

- Last RGB frame linear/MLP
- Two-frame difference
- Optical flow summary
- Route/speed prevalence control

V-JEPA MLP가 optical-flow control을 넘을 때 representation의 추가 가치를 주장할 수 있다.

### Priority 4: direct semantic Qwen arm

J0/J1/J2 V-JEPA 단계의 protocol은 이제 닫혔다. 세 model-family 비교를 완성하려면 DGX Spark에서 Qwen
direct semantic batch를 실행하고, 동일 99 definitive test runs와 10 Gold pairs에서 평가해야 한다.

## 15. 재현 커맨드

```bash
cd ~/isaac_ws/safety_probing

PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python \
  -m safety_probing.vjepa_transformer_probe \
  --feature-cache artifacts/features/vjepa_2_1_part_01_observation.pt \
  --linear-result outputs/vjepa_linear_probe/part_01/result.json \
  --output-root outputs/vjepa_transformer_probe/part_01

PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## 16. 주요 산출물

- Config: `configs/experiments/vjepa_transformer_probe.json`
- Runner: `src/safety_probing/vjepa_transformer_probe.py`
- Tests: `tests/test_vjepa_transformer_probe.py`
- Aggregate: `outputs/vjepa_transformer_probe/part_01/result.json`
- Seed checkpoints/predictions: `outputs/vjepa_transformer_probe/part_01/seed_*.pt`
- J1 interpretation: `docs/Summary_to_J1.md`

Outputs와 feature cache는 Git ignore 대상이며 config, runner, tests, summary가 재현 가능한 실험 계약이다.
