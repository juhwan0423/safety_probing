# Summary to J0

> 이 문서는 released predictor의 native future-retrieval 실험만 다룬다. 영상 latent에서 mass, CoM, ZMP,
> contact와 support polygon을 실제 물리 단위로 추출하는 후속 실험은
> `docs/Summary_to_Safety_Query.md`에 별도로 정리했다.

## 1. 이 문서의 결론

현재까지 simulator dataset, censor-aware label, leakage-safe split, LTX pixel rollout, V-JEPA J0 native
latent prediction pipeline을 구축하고 실행했다.

J0의 사전 정의된 per-query candidate retrieval은 54개 query 중 36개를 맞혀 **66.7%**였다. 그러나
동일한 candidate 접근 권한으로 마지막 관측 frame과 첫 미래 frame의 RGB MSE만 비교한 baseline이
**50/54 = 92.6%**를 기록했다. 따라서 현재 J0 결과는 zero-shot tip-over prediction이나 learned physics
understanding의 증거가 아니다. 더 정확한 결론은 다음과 같다.

> Frozen V-JEPA predictor는 A/B context 변화에 반응하지만, 현재 closed-set future-candidate retrieval
> protocol은 safety reasoning보다 clip boundary의 저수준 시각 연속성을 주로 측정한다.

J0는 유용한 pipeline 및 protocol diagnostic이지만 논문의 주 safety 결과로 사용하면 안 된다. 다음
핵심 실험은 candidate future를 제공하지 않는 J1 linear probe와 J2 temporal Transformer probe다.

## 2. 연구 질문과 범위

목표는 고정 CCTV prefix만 보고 hidden payload intervention이 일으키는 미래 횡방향 안정성을 예측하는
것이다.

- Hazard: turn-induced lateral tip-over
- Intervention: payload mass 또는 payload-local lateral CoM
- Maneuver: left/right finite steering pulse
- Excluded: straight braking, longitudinal pitch hazard
- Primary camera: `main_fixed_northwest`
- Observation: `0.0-2.7083 s`, 65 source frames at 24 FPS
- Future: `2.7083-5.0 s`, 55 source frames

조향은 2초에 시작해 3초에 0으로 돌아간다. 가장 빠른 definitive event는 2.958초이므로 observation은
초기 yaw response를 포함하지만 definitive failure 이전에 끝난다.

## 3. Dataset 구축 상태

### Simulation contract

| 항목 | 값 |
|---|---:|
| Episode duration | 5.0 s |
| Physics/log rate | 72 Hz |
| RGB rate | 24 FPS |
| Dense rows/run | 360 |
| RGB frames/camera/run | 120 |
| Cameras | fixed northwest + fixed top debug |
| Post-tip recording | 5초 끝까지 유지 |

Load는 spawn pose를 유지한 fixed joint로 fork에 연결했다. Ground는 square grid 환경이며 camera는 follow가
아닌 고정 CCTV다.

### 현재 접근 가능한 shard

현재 이 workstation에는 shard 01만 있으며 전체 계획의 59.92%다.

| 항목 | 현재 값 | 전체 계획 |
|---|---:|---:|
| Counterfactual pairs | 604 | 1,008 |
| Runs | 1,208 | 2,016 |
| Gold pairs | 58 | full shard 병합 후 재계산 |
| Dense rows scanned | 434,880 | full shard 병합 후 재계산 |

Shard 02/03이 아직 없으므로 모든 dataset 수치와 J0 결과는 preliminary다.

### Signal integrity

- Dense log 누락: 0
- Wrong row count: 0
- IMU valid: 100%
- System CoM available: 100%
- ZMP available: 100%
- Active contacts with complete x/y: 100%
- RGB frame-count mismatch: 0
- Sampled blank/decode/resolution failure: 0

`outward_lateral_margin_valid`가 28.38%인 것은 결측 문제가 아니다. 직진, 낮은 yaw rate, steering
transition grace, tip 이후를 의도적으로 제외한 validity mask다.

## 4. Label 현대화

Raw simulator label 대신 dense log를 다시 판정하는 `censor-aware-lateral-tip-v3`를 사용한다.

- `observed_tip`: `abs(roll) >= 20 deg` 또는 contact count `< 3`이 6 physics steps 지속
- `confirmed_recovered`: definitive event가 없고 terminal roll이 계속 outward-growing하지 않음
- `right_censored`: 아직 event는 없지만 5초 종료 시 roll이 계속 증가 중

현재 run 결과는 다음과 같다.

| Outcome | Runs |
|---|---:|
| Observed tip/instability | 687 |
| Confirmed recovered | 501 |
| Right-censored | 20 |

Source metadata상 stable이었지만 dense support-loss/roll evidence로 tip으로 승격된 run은 112개다.
Right-censored run은 binary classification에서 negative로 강제하지 않는다.

### ZMP 해석

Future minimum outward ZMP margin은 stable과 tip에서 모두 자주 음수다. `margin < 0`을 tip label로 쓰면
안 된다.

| Diagnostic | Tip AUROC |
|---|---:|
| `-future_min_outward_zmp_margin_m` | 0.853 |
| `-min_outward_com_margin_m` | 0.961 |

따라서 binary target은 censor-aware event label, ZMP/CoM margin은 continuous regression 또는 ranking
target으로 분리한다.

## 5. Split과 pair quality

인접 mass pair가 endpoint를 공유해 기존 pair-ID split에는 physical-condition leakage가 있었다. Shared
endpoint와 mirror counterpart의 connected component 전체를 한 split에 넣는 새 split을 만들었다.

- 기존 reference split의 duplicate physical-condition groups: 428
- 새 main split을 넘는 endpoint leakage: 0
- Local main split: train 426 / val 84 / test 94 pairs
- Gold: train 31 / val 17 / test 10 pairs

Pair tier의 의미는 다음과 같다.

- Gold: A/B definitive outcome이 다름
- Silver/Bronze: outcome은 같아도 consequence 차이가 있음
- Reject: 유용한 consequence contrast가 작음
- Censored: 적어도 한 run의 결과가 unresolved

## 6. Model comparison 설계

| Stage | Model | Native output 또는 readout |
|---|---|---|
| Pixel-generative | LTX-2.3 | prefix 이후 future RGB 생성 |
| Direct semantic | Qwen3.6 multimodal | future safety JSON 직접 출력 |
| J0 native latent | V-JEPA 2 | frozen predictor의 future latent |
| J1 probe | V-JEPA 2/2.1 | frozen observation latent + linear head |
| J2 probe | V-JEPA 2/2.1 | frozen observation tokens + 2-layer Transformer |

LTX, Qwen, V-JEPA의 native output은 서로 다르다. 따라서 native 결과를 억지로 동일 metric으로 부르지
않고, J1/J2에서 공통 standalone tip probability와 margin prediction을 비교한다.

## 7. LTX 단계 요약

### Prompt v1 batch

- 200 conditions x 4 seeds = 800/800 rollout 완료
- Prompt가 실제 blue grid를 `checkerboard`로 잘못 기술
- Sampled `observed_tip` rollout final frames 40개에서 visible rollover 0
- Forklift/part duplication, 문자, grid 변형, object fragmentation 발생

### Corrected action-aware prompt pilot

- Tip 2 + recovered 2 + censored 2 = 6/6 완료
- Scene description은 수정됐지만 visible rollover 0/6
- 기존 생성 artifact가 계속됨

현재 LTX에 대해 말할 수 있는 것은 runtime과 continuation pipeline이 작동한다는 점이다. Neutral prompt가
future tip을 재현한다고 말할 근거는 없다. Action-aware prompt는 future steering schedule을 text로
제공하므로 video-only V-JEPA J0와 직접 비교해서도 안 된다. Forced-tip/forced-stable controllability를
확인하기 전 corrected prompt 전체 rerun은 보류 상태다.

## 8. V-JEPA J0 protocol

### 왜 V-JEPA 2인가

Released V-JEPA 2 ViT-L predictor와 target encoder는 모두 1024-dimensional target space를 사용한다.
반면 현재 받은 distilled V-JEPA 2.1 ViT-L predictor는 1664-dimensional teacher space를 출력하지만 같은
checkpoint에 그 teacher target encoder가 없다. 임의 projection을 추가하지 않기 위해 J0는 V-JEPA 2로
고정했다. V-JEPA 2.1은 J1/J2 encoder probe에 사용한다.

### Input과 masking

- Observation 65 source frames에서 20 frames 균등 sampling
- Future 55 source frames에서 20 frames 균등 sampling
- Model input: `20 context + 20 masked future`
- Context tokens: 2,560
- Predicted future tokens: 2,560 x 1,024
- Encoder/predictor trainable parameters: 0

각 query A/B마다 query prefix는 고정하고 candidate future만 A 또는 B로 교체한 hybrid target을 만든다.
Frozen target encoder로 두 candidate latent를 만들고 predicted future와의 mean token cosine distance가 더
작은 candidate를 선택한다.

이 설정은 미래 후보 A/B를 모델 평가자에게 제공하는 closed-set multiple-choice retrieval이다. 관측
prefix만으로 standalone stable/tip을 출력하는 실험이 아니다.

## 9. J0 실행 결과

### Completion과 runtime

| 항목 | 값 |
|---|---:|
| Gold pairs | 27/27 complete |
| Validation | 17 pairs |
| Test | 10 pairs |
| Queries | 54 |
| Mean runtime/pair | 1.378 s |
| Peak GPU allocation | 2,079,602,176 bytes |
| Saved latent size | 849,405,447 bytes |

평가 subset은 medium speed 26/27, medium steering 25/27로 크게 치우쳐 있다. Intervention은 load mass
20, CoM shift 7이며 방향은 left 13, right 14다. 다양한 maneuver 일반화 결과로 해석하면 안 된다.

### 사전 정의된 per-query retrieval

| Subset | Correct | Accuracy |
|---|---:|---:|
| Overall | 36/54 | 66.7% |
| Validation | 24/34 | 70.6% |
| Test | 12/20 | 60.0% |
| Left turn | 15/26 | 57.7% |
| Right turn | 21/28 | 75.0% |
| Load mass | 26/40 | 65.0% |
| CoM shift | 10/14 | 71.4% |
| Expected tip | 20/27 | 74.1% |
| Expected recovered | 16/27 | 59.3% |

Naive query-level Wilson 95% interval은 53.4-77.8%다. Query 두 개가 한 pair에 묶여 있으므로 독립 표본으로
취급한 p-value를 주 결과로 쓰지 않는다. Pair-cluster bootstrap interval은 약 57.4-75.9%였다.

### Pair 구조

| Pair result | Pairs |
|---|---:|
| A/B query 모두 correct | 9/27 |
| 둘 중 하나만 correct | 18/27 |
| 둘 다 wrong | 0/27 |

Candidate 선택 pattern은 `A/B` 9개, `A/A` 8개, `B/B` 10개, reversed `B/A` 0개였다. 즉 모델이 두
query를 명확히 구분한 9개는 모두 올바른 방향이었지만, 18개 pair에서는 candidate 하나를 두 query에
공통으로 선호했다. Absolute energy에는 candidate-specific bias가 크다.

Smooth L1도 36/54와 9/27로 같은 aggregate 결과를 냈지만 개별 candidate 선택 일치는 44/54였다.

## 10. Post-hoc joint assignment

Candidate-specific offset을 상쇄하기 위해 다음 2x2 assignment margin을 batch를 본 뒤 추가 계산했다.

```text
M_joint = [E(A,B) + E(B,A)] - [E(A,A) + E(B,B)]
M_joint > 0  => diagonal A->A, B->B assignment가 더 적합
```

- Cosine joint assignment: 26/27 = 96.3%
- Smooth L1 joint assignment: 26/27 = 96.3%
- Test: 10/10
- Right: 14/14
- CoM shift: 7/7
- 유일한 miss: `A_mass_s2_left_vmedium_smedium_comright_m125_to_m150`
- Miss margin: `-0.000336`, 사실상 near-tie

이 수치는 매우 좋아 보이지만 **결과를 본 뒤 정의한 post-hoc metric**이다. Part 01 test의 confirmatory
결과로 보고하면 안 된다. Metric을 지금 freeze한 뒤 untouched shard 02/03에서 재검증해야 한다.

## 11. 단순 pixel baseline이 바꾸는 해석

J0와 동일하게 A/B future candidate를 제공하고, V-JEPA와 동일한 256 center-crop preprocessing에서 다음
한 줄짜리 energy를 계산했다.

```text
E_pixel(q,c) = MSE(query frame 64, candidate frame 65)
```

| Metric | V-JEPA J0 | Boundary RGB MSE |
|---|---:|---:|
| Per-query top-1 | 36/54 (66.7%) | 50/54 (92.6%) |
| Both queries correct | 9/27 (33.3%) | 23/27 (85.2%) |
| Joint assignment | 26/27 (96.3%, post-hoc) | 27/27 (100%) |

Pixel baseline이 모든 J0 retrieval metric을 능가한다. 이는 candidate A/B가 실제 future clip이기 때문에
frame 64에서 frame 65로 이어지는 단순 visual continuity만으로 정답을 찾을 수 있음을 뜻한다. J0의 높은
joint score를 latent physics understanding으로 해석할 수 없는 결정적인 이유다.

이것은 V-JEPA 자체가 물리를 학습하지 못했다는 증거도 아니다. 현재 evaluation task가 world-model
prediction과 단순 clip identity matching을 분리하지 못한다는 뜻이다.

## 12. Temporal latent diagnostic

Future token을 10개 temporal tubelet으로 나눠 본 exploratory 결과는 다음과 같다.

| Future tubelet | Approx. time | Local accuracy | Cumulative accuracy |
|---:|---:|---:|---:|
| 1 | 2.708-2.833 s | 68.5% | 68.5% |
| 4 | 3.417-3.542 s | 63.0% | 75.9% |
| 10 | 4.833-4.958 s | 55.6% | 66.7% |

후반부로 갈수록 local accuracy가 개선되지 않고 첫 tubelet부터 신호가 강하다. Pixel baseline과 함께 보면
short-horizon continuity 의존 가능성이 높다. 단, target encoder는 bidirectional self-attention을 사용하므로
각 tubelet을 독립적인 causal time slice로 해석하면 안 된다.

## 13. 현재 주장 가능한 것

1. Dataset generation, sensor logging, label modernization, split audit pipeline이 end-to-end로 동작한다.
2. Frozen V-JEPA 2 encoder/predictor로 fully masked future latent를 안정적으로 계산할 수 있다.
3. Predictor energy는 query A/B 변화에 반응하며 reversed `B/A` pattern은 관측되지 않았다.
4. 현재 candidate retrieval에는 강한 boundary-continuity shortcut이 존재한다.
5. LTX는 pipeline 수준에서는 실행되지만 neutral rollout에서 tip을 재현한다는 증거가 아직 없다.

## 14. 아직 주장하면 안 되는 것

1. V-JEPA가 CCTV prefix만으로 tip-over를 zero-shot 66.7%로 예측했다.
2. Joint assignment 96.3%가 latent physics understanding을 증명한다.
3. J0를 LTX/Qwen의 standalone safety output과 같은 metric으로 직접 비교할 수 있다.
4. Part 01 결과가 full 1,008-pair grid 또는 다양한 speed/steering regime을 대표한다.
5. ZMP margin의 부호만으로 stable/tip ground truth를 만들 수 있다.

## 15. 다음 실험 우선순위

### Priority 1: J1 linear probe

Candidate future 없이 observation prefix만 입력한다.

```text
frozen V-JEPA observation tokens
-> fixed mean pooling
-> linear tip / margin / pairwise heads
```

이 실험이 standalone safety information이 representation에 선형적으로 존재하는지 답한다.

### Priority 2: J2 temporal Transformer probe

동일 frozen tokens에 2-layer temporal Transformer를 붙인다. J1보다 유의하게 좋아질 때만 temporal
reasoning을 주장한다. Encoder는 계속 freeze한다.

### Priority 3: 반드시 포함할 controls

- Boundary RGB MSE baseline
- Last-frame/optical-flow baseline
- Silver/Bronze same-outcome pair에서 J0 identity matching
- 첫 1-4 future tubelet을 제외한 J0 energy
- Candidate future를 unrelated route에서 가져온 negative control
- Left/right mirror generalization
- Observation horizon ablation

Same-outcome pair에서도 J0 joint assignment가 높다면 J0는 safety가 아니라 generic trajectory identity를
측정한다는 뜻이다.

### Priority 4: untouched data validation

Post-hoc joint assignment를 지금 protocol에 고정하더라도 part 01 test에는 다시 적용해 confirmatory
claim을 만들지 않는다. Shard 02/03을 병합하기 전에 metric과 controls를 고정하고, untouched shard에서
재검증한다.

## 16. 논문에 쓸 수 있는 현재 한 문단

> In an exploratory closed-set future retrieval experiment on shard 01, the frozen V-JEPA 2 predictor achieved
> 66.7% per-query accuracy. A post-hoc joint A/B assignment reached 96.3%; however, a one-frame RGB continuity
> baseline achieved 92.6% per-query and 100% joint assignment. These results show that the current candidate
> retrieval protocol is dominated by boundary continuity and cannot establish zero-shot safety prediction.
> We therefore treat J0 as a protocol diagnostic and evaluate safety accessibility using observation-only frozen
> linear and temporal probes.

## 17. 주요 산출물

- Dataset statistics: `docs/DATASET_STATISTICS_KO.md`
- Dataset card: `docs/DATASET_CARD_KO.md`
- Model plan: `docs/MODEL_EXPERIMENT_PLAN_KO.md`
- LTX audit: `docs/LTX_PROMPT_AUDIT_KO.md`
- J0 config: `configs/experiments/vjepa_native_future.json`
- J0 batch summary: `outputs/vjepa_native_batch/part_01/batch_summary.json`
- J0 pair manifest: `outputs/vjepa_native_batch/part_01/batch_manifest.jsonl`
- J0 raw latents: `outputs/vjepa_native_batch/part_01/{val,test}/{pair_id}/latents.pt`

## 18. Reproducibility 주의

- J0 checkpoint와 external repository revision은 batch snapshot에 저장돼 있다.
- J0 outputs는 `experiment_sha256`와 `registry_sha256`가 일치할 때만 resume된다.
- Dataset statistics 생성 당시 generator worktree는 dirty 상태였다. Final release 전에 clean commit에서 dataset
  provenance를 다시 고정해야 한다.
- Shard 02/03 병합 후 statistics, split balance, label counts를 다시 생성해야 한다.
