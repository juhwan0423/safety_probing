# ExoFork-CF Dataset Card

## 1. 목적

이 데이터셋은 동일한 시각적 장면과 동일한 조종 명령에서 숨겨진 payload mass 또는
payload-local lateral CoM만 개입했을 때 미래의 횡방향 안정성이 어떻게 달라지는지를 연구한다.
주요 연구 질문은 다음과 같다.

1. frozen visual representation에서 미래 전도를 선형 또는 작은 temporal probe로 예측할 수 있는가?
2. 모델이 raw roll 방향이나 카메라 shortcut이 아니라 회전 바깥쪽 support boundary까지의 안정성 margin을 찾는가?
3. 좌회전에서 학습한 표현이 우회전으로, 또는 그 반대로 일반화되는가?
4. 시각적으로 유사하지만 미래 결과가 다른 matched counterfactual pair를 구분할 수 있는가?

이번 버전의 범위는 **turn-induced lateral tip-over**다. Straight braking과 종방향 pitch hazard는
포함하지 않는다.

## 2. 생성 명세

| 항목 | 값 |
|---|---|
| Physics/log frequency | 72 Hz |
| RGB frequency | 24 FPS |
| Episode duration | 5.0 s |
| Physics rows/run | 360 |
| RGB frames/camera/run | 120 |
| Run 구성 | matched `run_A`, `run_B` |
| Camera | `main_fixed_northwest`, `top_fixed_debug` |
| Resolution | 1280 x 720 |
| Tip 이후 저장 | 5초 끝까지 계속 저장 |
| Payload attachment | spawn pose를 보존한 fixed joint to fork |

Action schedule은 모든 route에서 다음과 같다.

| 구간 | Physics step | 시간 | 명령 |
|---|---:|---:|---|
| Approach | 0-143 | 0.0-2.0 s | 직진, 목표 wheel velocity 도달 |
| Turn | 144-215 | 2.0-3.0 s | 좌 또는 우 steering pulse |
| Recovery | 216-359 | 3.0-5.0 s | steering 0으로 복귀, 직진 유지 |

조향 복귀 시점은 A/B state에 따라 동적으로 바꾸지 않는다. 질량과 CoM이 달라도 command timing을
동일하게 유지하기 위한 결정이다.

## 3. Factorial Axes

| 축 | 값 |
|---|---|
| Turn direction | left, right |
| Drive command | slow `-15`, medium `-20`, fast `-25` |
| Steering magnitude | wide `1.0`, medium `1.25`, tight `1.5` rad |
| Payload mass | 240, 360, 480, 600, 720 kg |
| Load-local lateral CoM | -0.25, 0.0, +0.25 m |
| Seeds | 7, 8, 9 |

전체 설계는 18 routes, 15 scenes, 1,008 counterfactual pairs다.

- Adjacent mass interventions: 648 pairs
- Sampled adjacent lateral-CoM interventions: 360 pairs
- Distributed shards: 604 / 302 / 102 pairs

## 4. Counterfactual Contract

한 pair에서 고정되는 항목:

- route와 command timestamp
- 카메라, 조명, 바닥
- load mesh, 색상, visible spawn pose
- physics context
- seed

변경되는 항목은 `load_mass` 또는 `com_shift` 하나뿐이다. 여기서 same-action은 **동일한 command**를
뜻하며 동일한 realized trajectory를 뜻하지 않는다. Payload가 motor response, 실제 속도, contact,
roll dynamics를 바꾸는 것은 관측하려는 causal consequence에 포함된다.

## 5. Safety Labels

### Censor-aware lateral safety outcome

Raw `metadata_run.json`의 detector 결과는 provenance로 보존하고, probing manifest에서는 dense log를
재판정한 `censor-aware-lateral-tip-v3`를 사용한다. v3는 65-frame observation boundary를 사용해
조향 onset 이후의 명확한 초기 회전까지 입력에 포함한다.

- `observed_tip`: `abs(roll) >= 20 deg` 또는 contact count가 6 physics steps 동안 3 미만
- `confirmed_recovered`: 위 event가 없고 종료 0.5초의 absolute roll이 계속 증가하지 않음
- `right_censored`: event는 아직 없지만 종료 시 absolute roll이 0.5도 이상이고 2 deg/s 이상 증가 중

`observed_tip`은 완전히 옆으로 누운 상태뿐 아니라 지속적인 2-point support/wheel-lift instability를
포함한다. 이는 회복 가능한 단일-frame contact drop과 구분하기 위해 6-step persistence를 요구한다.
Right-censored run은 binary AUROC와 pairwise binary target에서 제외하고 survival 분석에는 포함한다.

Raw detector의 pitch threshold는 89도로 유지하여 이번 lateral-only 연구에서 종방향 braking failure가
label에 들어오는 것을 억제한다.

### Outward lateral margin

회전 방향 label이나 raw roll sign을 직접 사용하지 않고 실제 chassis velocity와 yaw rate로 회전
바깥쪽 단위벡터를 계산한다.

- `zmp_outward_lateral_margin_m > 0`: outward support boundary 안쪽
- `zmp_outward_lateral_margin_m < 0`: outward boundary 바깥쪽
- `com_outward_lateral_margin_m`: 동일한 경계에 대한 projected CoM margin

`outward_lateral_margin_valid=true`인 row만 학습 target으로 사용한다. 이 mask는 저속/저 yaw-rate,
조향 transition 직후 6 steps, 계산 불가능 구간, tip 이후를 제외한다.

부호 정의가 이론적으로 위와 같더라도 finite-difference dynamic ZMP는 순간적으로 support boundary를
벗어났다가 차량이 회복할 수 있다. 따라서 `ZMP margin < 0`은 binary tip-over label과 동의어가 아니다.
ZMP는 continuous target으로 사용하고 binary tip은 detector label을 사용한다.

## 6. Pair Quality

| Tier | 의미 | 권장 용도 |
|---|---|---|
| Gold | definitive A/B safety outcome이 다름 | 핵심 pairwise causal 평가 |
| Silver | 결과는 같지만 roll/pitch consequence 차이가 큼 | margin regression 및 control |
| Bronze | 측정 가능한 consequence 차이 | 보조 control |
| Reject | 결과와 consequence 차이가 모두 작음 | raw 보존, primary benchmark 제외 |
| Censored | A/B 중 적어도 하나가 right-censored | binary pair 평가 제외, survival 보조 분석 |

Quality tier는 현재 pair metadata의 roll/pitch consequence 기준이다. Outward ZMP near-boundary subset은
dense log를 이용해 별도로 선별해야 한다.

## 7. Provenance와 버전 고정

논문 수치를 만들 때 다음 네 항목을 함께 기록한다.

1. generator Git commit과 dirty 여부
2. source experiment config SHA-256
3. leakage-safe split manifest SHA-256
4. statistics 생성 timestamp

이 값은 자동 생성되는 `DATASET_STATISTICS_KO.md`와 `dataset_statistics.json`에 기록된다.

## 8. 알려진 제약

- 동일 command라도 payload에 따라 realized speed와 yaw가 달라진다.
- ZMP는 flat-ground assumption과 finite-difference system CoM acceleration으로 계산한 근사값이다.
- Generic `zmp_support_margin_m`은 steering transition에서 큰 spike가 날 수 있다. Primary target은
  transition mask가 적용된 outward margin을 사용한다.
- Support polygon은 contact sensor position을 우선하고, 없으면 contact body position을 fallback으로 쓴다.
- 두 카메라는 고정되어 있어 camera-specific shortcut 여부를 cross-camera 또는 top-view ablation으로
  확인해야 한다.
- RGB 개수 검사는 완전성만 보장한다. Blank frame, duplicate frame, compression artifact는 sampled pixel
  audit가 추가로 필요하다.
- 세 shard를 모두 합치기 전 통계는 preliminary다.
