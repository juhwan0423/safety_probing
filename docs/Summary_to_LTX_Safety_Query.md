# Summary to LTX Physical Safety Query

## 1. 결론

Shard 01의 LTX Physical Safety Query를 완료했다.

> Frozen LTX VAE는 실제 simulator future에서 물리 상태를 매우 잘 표현한다. 그러나 prompt-v1 LTX
> continuation이 만든 future latent에는 그 정보가 크게 손실됐다. Generated rollout은 hidden mass 변화
> 방향과 coarse tip ranking은 일부 보존했지만, lateral CoM과 ZMP-consistent stability ordering은 거의
> 보존하지 못했다.

핵심 결과는 다음과 같다.

| Metric | Real-future oracle | Generated rollout ensemble |
|---|---:|---:|
| Load mass MAE | `11.93 +/- 0.83 kg` | `105.00 +/- 5.92 kg` |
| Lateral CoM MAE | `0.0213 +/- 0.0015 m` | `0.1845 +/- 0.0086 m` |
| Future roll MAE | `2.32 +/- 0.35 deg` | `16.28 +/- 2.18 deg` |
| Future minimum ZMP-margin MAE | `0.0313 +/- 0.0021 m` | `0.1163 +/- 0.0140 m` |
| Tip AUROC | `0.9996 +/- 0.0007` | `0.7230 +/- 0.0538` |
| Mass-pair delta direction | `0.994 +/- 0.010` | `0.800 +/- 0.017` |
| CoM-pair delta direction | `1.000 +/- 0.000` | `0.598 +/- 0.103` |

Oracle은 실제 미래 영상을 보므로 prediction 결과가 아니다. 동일 frozen latent space와 query가 충분히
작동한다는 측정 상한이며, oracle과 rollout의 차이가 생성 단계의 information loss다.

## 2. 실험 계약

### Real-future oracle

```text
Simulator RGB frames 0-119 + repeated frame 119
-> 121 frames, 576x320
-> frozen LTX-2.3 video VAE encoder
-> future latent indices 9:16, [7,10,18,128]
-> physical query decoder training/evaluation
```

- Canonical conditions: 670
- Main split: train 470 / validation 98 / test 102
- VAE encoder parameters: 318,936,897, all frozen
- Real latent cache: `[670,7,10,18,128]` float16, 207 MB
- VAE encode peak allocation: 8,397,758,464 bytes
- Exact preprocessing: official PyTorch bilinear aspect-fill + center crop
- Source frame 119를 한 번 반복해 LTX temporal geometry `1 + 8k = 121`을 만족

PNG decoding은 ffmpeg RGB24를 사용하되 resize/crop/normalization은 official LTX 경로와 맞췄다. Reference
preprocessing과 normalized-pixel mean absolute difference는 약 `9.6e-8`이었다.

### Generated rollout

```text
CCTV prefix 65 frames
-> frozen LTX-2.3 prompt-v1 continuation
-> saved final latent [1,128,16,10,18]
-> future indices 9:16
-> real-only trained frozen query decoder
```

- Conditions: validation 98 + test 102
- Rollout seeds: 17, 29, 43, 71
- Jobs: 800, runtime failure 0
- Query training에 generated latent 사용: 0
- Primary condition output: 네 rollout의 continuous prediction 평균과 probability 평균

### Query decoder

```text
future latent [B,7,10,18,128]
-> learned temporal/spatial positions
-> 63 physical query tokens
-> 2-layer, 4-head Transformer decoder
-> physical-unit heads
```

Query는 static 2, future state 10, future contacts 50, future summary 1개다. Trainable parameter는
`450,075`, foundation parameter는 0이다. Seeds 17/29/43, checkpoint는 real validation normalized loss로만
고른다.

## 3. Real-future oracle

Oracle은 frozen LTX VAE representation과 query implementation이 물리 target을 읽을 수 있는지 확인한다.

### Hidden properties

| Query | MAE | Level accuracy | All-tier pair direction |
|---|---:|---:|---:|
| Mass | `11.93 kg` | `0.997` | `0.994`, 60 pairs |
| Lateral CoM | `0.0213 m` | `0.980` | `1.000`, 34 pairs |

Train-mean baseline MAE는 mass `144.25 kg`, CoM `0.1670 m`다.

### Future state

| Query | Oracle MAE | Train-mean MAE | Spearman |
|---|---:|---:|---:|
| Roll | `2.320 deg` | `15.276 deg` | `0.930` |
| Body velocity x / y | `0.177 / 0.090 m/s` | `1.012 / 0.599 m/s` | `0.977 / 0.974` |
| System CoM x / y / z | `0.0093 / 0.0358 / 0.0272 m` | `0.0454 / 0.2484 / 0.1990 m` | `0.973 / 0.944 / 0.951` |
| Valid ZMP x / y | `0.0543 / 0.0882 m` | `0.0937 / 0.2922 m` | `0.767 / 0.918` |
| ZMP / CoM outward margin | `0.0915 / 0.0386 m` | `0.1981 / 0.1067 m` | `0.836 / 0.832` |

### Contacts and summaries

| Query | Oracle result |
|---|---:|
| Future contact F1 | `0.9863 +/- 0.0031` |
| Active contact position MAE | `0.0251 +/- 0.0011 m` |
| Polygon valid coverage | `0.9926 +/- 0.0051` |
| Polygon area MAE | `0.0518 +/- 0.0072 m2` |
| Polygon centroid error | `0.0390 +/- 0.0057 m` |
| Minimum ZMP margin MAE / rho | `0.0313 m / 0.943` |
| Minimum CoM margin MAE / rho | `0.0085 m / 0.968` |
| Maximum absolute roll MAE / rho | `2.55 deg / 0.952` |
| Minimum contact count MAE / rho | `0.183 / 0.966` |

Oracle tip AUROC는 `0.9996`, balanced accuracy는 `0.9972`다. 이 결과는 VAE latent가 물리 readout에
충분하다는 뜻이지 LTX가 prefix에서 미래를 예측했다는 뜻은 아니다.

## 4. Generated rollout 결과

### Hidden properties and outcome

| Query | Generated | Train-mean baseline |
|---|---:|---:|
| Mass MAE | `105.00 +/- 5.92 kg` | `144.25 kg` |
| Mass level accuracy | `0.310 +/- 0.015` | 5-level chance `0.20` |
| Lateral CoM MAE | `0.1845 +/- 0.0086 m` | `0.1670 m` |
| CoM level accuracy | `0.333 +/- 0.017` | 3-level chance `0.33` |
| Tip AUROC | `0.7230 +/- 0.0538` | `0.50` |
| Tip balanced accuracy | `0.579 +/- 0.024` | `0.50` |

Mass dynamics는 일부 남지만 CoM absolute decoding은 train-mean보다도 나쁘다. Tip score도 chance보다 높지만
query seed에 따라 `0.682-0.784`로 흔들린다.

### Physical trajectory

| Query | Generated MAE | Train-mean MAE | Generated rho |
|---|---:|---:|---:|
| Roll | `16.28 deg` | `15.28 deg` | `0.492` |
| Body velocity x / y | `0.772 / 0.400 m/s` | `1.012 / 0.599 m/s` | `0.545 / 0.838` |
| System CoM x / y / z | `0.047 / 0.259 / 0.169 m` | `0.045 / 0.248 / 0.199 m` | `0.324 / 0.571 / 0.512` |
| Valid ZMP x / y | `0.106 / 0.191 m` | `0.094 / 0.292 m` | `0.198 / 0.777` |
| ZMP / CoM outward margin | `0.157 / 0.099 m` | `0.198 / 0.107 m` | `0.569 / 0.282` |

일부 direction/ranking 신호는 남지만 여러 absolute MAE가 constant train-mean과 비슷하거나 더 나쁘다.

### Contacts and stability summaries

| Query | Generated result | Oracle result |
|---|---:|---:|
| Future contact F1 | `0.828 +/- 0.018` | `0.986` |
| Contact position MAE | `0.0467 +/- 0.0015 m` | `0.0251 m` |
| Polygon coverage | `0.801 +/- 0.056` | `0.993` |
| Polygon area MAE | `0.280 +/- 0.014 m2` | `0.052 m2` |
| Polygon centroid error | `0.159 +/- 0.020 m` | `0.039 m` |
| Minimum ZMP-margin MAE / rho | `0.1163 m / 0.298` | `0.0313 m / 0.943` |
| Minimum CoM-margin MAE / rho | `0.0471 m / 0.349` | `0.0085 m / 0.968` |
| Maximum roll MAE / rho | `33.49 deg / 0.300` | `2.55 deg / 0.952` |
| Minimum contact count MAE / rho | `1.283 / 0.415` | `0.183 / 0.966` |

Gold test pair의 future ZMP-margin ordering은 `0.333 +/- 0.058`, CoM-margin ordering은
`0.433 +/- 0.058`이다. Tip AUROC가 0.723이라고 해서 generated rollout이 ZMP-consistent physics를
재현했다고 볼 수 없는 가장 직접적인 근거다.

## 5. Counterfactual pair 보존

| Pair query | Oracle | Generated ensemble |
|---|---:|---:|
| Mass delta direction, 60 pairs | `0.994` | `0.800` |
| CoM delta direction, 34 pairs | `1.000` | `0.598` |
| Gold ZMP-risk ordering, 10 pairs | `0.833` | `0.333` |
| Silver ZMP-risk ordering, 30 pairs | `0.811` | `0.422` |
| Bronze ZMP-risk ordering, 12 pairs | `0.917` | `0.472` |
| Reject ZMP-risk ordering, 38 pairs | `0.833` | `0.395` |

Generated mass pair는 세 query seed 모두 `0.783-0.817`로 비교적 안정적이다. CoM pair는
`0.500-0.706`으로 불안정하다. Generated model이 hidden intervention을 완전히 지우지는 않지만, 그 차이를
올바른 future stability consequence로 전개하지 못한다.

## 6. Rollout stochasticity와 ensemble

동일 query heads에서 rollout seed별 test tip AUROC 평균은 다음과 같다.

| Rollout seed | Tip AUROC | Mass MAE | CoM MAE |
|---:|---:|---:|---:|
| 17 | `0.764` | `123.32 kg` | `0.190 m` |
| 29 | `0.519` | `145.21 kg` | `0.195 m` |
| 43 | `0.640` | `114.34 kg` | `0.206 m` |
| 71 | `0.569` | `108.51 kg` | `0.210 m` |
| Four-rollout ensemble | **`0.723`** | **`105.00 kg`** | **`0.185 m`** |

단일 rollout 평균 AUROC는 약 `0.623`이다. Ensemble은 ranking과 mass MAE를 개선하지만 CoM과 ZMP
physics를 복구하지는 못한다.

## 7. Feature controls

Real-future oracle에 test-time intervention을 적용했다.

| Input | Mass MAE kg | CoM MAE m | Min-ZMP MAE m | Tip AUROC | Mass pair | CoM pair |
|---|---:|---:|---:|---:|---:|---:|
| Identity | **11.93** | **0.0213** | **0.0313** | **1.000** | **0.994** | **1.000** |
| Zero values | 156.71 | 0.2168 | 0.1783 | 0.500 | 0.500 | 0.500 |
| Condition permutation | 179.07 | 0.2212 | 0.1512 | 0.585 | 0.511 | 0.569 |
| Spatial mean repeated | 168.75 | 0.2076 | 0.1072 | 0.890 | 0.656 | 0.667 |
| Temporal mean repeated | 90.58 | 0.1392 | 0.0749 | 0.924 | 0.800 | 0.804 |
| Temporal reverse | 15.99 | 0.0271 | 0.0344 | 0.999 | 0.994 | 1.000 |

Zero/permutation에서 oracle이 무너지고 spatial mean이 특히 치명적이므로 VAE query가 learned prior만 출력한
것은 아니다. Temporal mean은 부분 신호를 남기며 reverse는 작은 영향만 준다.

Generated ensemble reverse-time도 tip AUROC `0.703`, mass MAE `110.7 kg`, ZMP summary MAE `0.116 m`로
chronological input과 비슷하다. Static/summary 중심 결과에 explicit arrow-of-time evidence는 없다.

## 8. V-JEPA와 비교

| Metric | V-JEPA prefix SQ | LTX real-future oracle | LTX generated rollout |
|---|---:|---:|---:|
| Trainable query params | 570,928 | 450,075 | same frozen LTX query |
| Mass MAE | `14.44 kg` | `11.93 kg` | `105.00 kg` |
| CoM MAE | `0.0331 m` | `0.0213 m` | `0.1845 m` |
| Future roll MAE | `3.08 deg` | `2.32 deg` | `16.28 deg` |
| Min-ZMP summary MAE | `0.0457 m` | `0.0313 m` | `0.1163 m` |
| Tip AUROC | `0.9996` | `0.9996` | `0.7230` |
| Mass pair direction | `0.994` | `0.994` | `0.800` |
| CoM pair direction | `0.961` | `1.000` | `0.598` |

V-JEPA는 observation prefix만 보고 future target을 예측한다. LTX oracle은 실제 future를 보므로 V-JEPA보다
좋다고 비교할 수 없다. 핵심 비교는 V-JEPA prefix query와 LTX generated rollout이다. 현재 prompt-v1에서
V-JEPA representation query가 LTX generated future보다 물리 target을 훨씬 잘 보존한다.

## 9. 현재 주장 가능한 것

1. Frozen LTX VAE latent는 실제 forklift future의 mass, CoM, dynamics, support geometry와 stability state를
   supervised query로 복원할 수 있다.
2. Prompt-v1 LTX continuation은 이 정보를 크게 손실하며 특히 CoM과 ZMP-based ordering이 약하다.
3. Four-rollout ensemble은 단일 rollout보다 낫지만 oracle gap을 닫지 못한다.
4. Generated rollout에는 mass intervention direction 신호가 일부 남아 있다.
5. V-JEPA observation latent query가 현재 LTX generated-rollout query보다 훨씬 강하다.

## 10. 아직 주장하면 안 되는 것

1. 모든 pixel-generative world model이 latent-predictive model보다 물리를 못한다.
2. Prompt-v1 결과가 최적 LTX prompting 또는 fine-tuned LTX의 상한이다.
3. Tip AUROC 0.723이 물리적으로 올바른 rollover rollout을 뜻한다.
4. Generated latent의 estimated ZMP가 simulator에서 직접 계산된 진짜 ZMP다.
5. Shard 01 exploratory 결과가 다른 camera, scene, vehicle과 untouched shard에 일반화된다.

Prompt-v1은 실제 blue square grid를 checkerboard라고 잘못 기술했다. Corrected prompt pilot도 visible tip을
만들지 못했지만 6 jobs뿐이다. 본 결과는 정확히 `prompt-v1 archived baseline`으로 이름 붙인다.

## 11. 산출물

- Protocol: `docs/LTX_SAFETY_QUERY_PROTOCOL_KO.md`
- Experiment: `configs/experiments/ltx_safety_query.json`
- Real latent cache: `artifacts/features/ltx_2_3_part_01_real_future.pt`
- Rollout latent cache: `artifacts/features/ltx_2_3_part_01_rollout_future.pt`
- Dense target cache: `artifacts/targets/ltx_safety_query_part_01.pt`
- Result: `outputs/ltx_safety_query/part_01_prompt_v1/result.json`
- Checkpoints/raw predictions: `outputs/ltx_safety_query/part_01_prompt_v1/seed_*.pt`

각 checkpoint에는 model state, train-only normalizer, real predictions, rollout-seed별 predictions, ensemble
prediction과 condition ID order가 저장된다. Generated latents는 query training에 사용되지 않았다.
