# Full-Grid Completion Runbook

## 1. 정확한 의미

`part_01/02/03`은 서로 다른 scientific split이 아니라 1,008 counterfactual pairs를 세 컴퓨터에 분배한
생성 작업 shard다.

| Shard | Pairs | Fraction |
|---|---:|---:|
| part 01 | 604 | 60% |
| part 02 | 302 | 30% |
| part 03 | 102 | 10% |

따라서 part 02/03을 합치면 full-grid 통계와 각 component 안의 표본은 늘지만 새로운 독립 test component가
생기지는 않는다. 이를 `confirmatory shard`라고 부르지 않는다. 독립 확인에는 새 route/seed/scene로 구성한
held-out component가 추가로 필요하다.

## 2. 복사 위치

다른 컴퓨터에서 생성한 폴더를 내용 변경 없이 다음 위치에 둔다.

```text
~/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/
  eccv_workshop_grid_part_01/
  eccv_workshop_grid_part_02/
  eccv_workshop_grid_part_03/
```

`pairs/`, `configs/`, `scenarios/`, `summaries/`를 전부 복사한다. 서로 다른 shard의 pair를 한 폴더에 합치지
않는다.

## 3. Preflight

```bash
cd ~/isaac_ws/safety_probing
scripts/run_full_grid_completion.sh preflight
```

검사 항목:

- shard config union이 source 1,008 pair와 정확히 일치
- shard 간 duplicate pair ID 없음
- expected/actual pair directory 일치
- pair/run metadata와 command/dense logs 존재
- 모든 dense log가 360 rows
- 두 camera가 run마다 각각 120 PNG frames

2026-07-13 완료 기준으로 part 01/02/03 모두 ready이며 `1,008/1,008` pairs complete, missing `0`,
incomplete `0`이다. `--strict`는 1,008 pairs가 전부 ready가 아니면 exit code 2로 중단한다.

## 4. Frozen protocol

`artifacts/protocols/physical_safety_query_workshop_v1.json`은 다음을 SHA-256으로 동결한다.

- censor-aware label과 leakage-safe split
- V-JEPA/LTX Safety Query config
- prompt-v1 LTX continuation config
- target/query/cache 구현
- query seeds와 rollout seeds

각 긴 단계는 실행 전에 lock을 검증한다. Full grid를 본 뒤 target, metric, prompt를 바꾸면 같은 protocol
ID로 결과를 덮어쓰지 않는다.

## 5. 실행 순서

```bash
cd ~/isaac_ws/safety_probing

scripts/run_full_grid_completion.sh audit
scripts/run_full_grid_completion.sh vjepa
scripts/run_full_grid_completion.sh vjepa-baselines
scripts/run_full_grid_completion.sh ltx-real
scripts/run_full_grid_completion.sh ltx-rollout
scripts/run_full_grid_completion.sh ltx-query
```

GPU 단계는 자동으로 한꺼번에 이어 실행하지 않는다. 각 단계의 output contract와 completion count를 확인한
뒤 다음 단계로 넘어간다. `ltx-rollout`은 prompt-v1 비교를 동일 cohort로 확장하는 경우에만 실행한다.

## 6. 논문 반영 규칙

1. 세 model view가 같은 full canonical cohort를 모두 커버한 뒤 표를 교체한다.
2. Full-grid 결과는 `full-grid completion`으로 부른다.
3. Shard 01에서 architecture와 metric을 설계했다는 사실은 계속 명시한다.
4. 독립 confirmatory claim은 새로운 held-out component 실험 전까지 사용하지 않는다.
5. 기존 shard-01 표와 PDF는 exploratory snapshot으로 보존한다.

세 model view의 full output이 모두 완료된 뒤에만 논문 통계를 명시적으로 교체한다.

```bash
cd /home/jude/CWM_paper/Safety_Probing_paper
make refresh-stats COHORT=full
make all
```

## 7. 2026-07-13 완료 기록

| Artifact | Result |
|---|---:|
| Generated pairs / run instances | 1,008 / 2,016 |
| Canonical conditions | 810 |
| Train / validation / test | 570 / 120 / 120 |
| Definitive test | 117 = tip 70 + recovered 47 |
| Test Gold pairs | 20, across 3 effective components |
| V-JEPA tip AUROC | 0.9974 |
| V-JEPA mass / CoM / future-ZMP MAE | 19.86 kg / 0.0553 m / 0.0543 m |
| LTX real-future oracle tip AUROC | 1.0000 |
| LTX generated tip AUROC | 0.7612 |
| LTX generated mass / CoM / future-ZMP MAE | 95.50 kg / 0.1828 m / 0.1114 m |

LTX prompt-v1 val/test batch는 240 conditions, four rollout seeds, 총 960 jobs다. 기존 part-01과 동일한
experiment hash 및 split을 가진 800 jobs는 원본을 변경하지 않는 hard link로 재사용했고, 새 40 conditions의
160 jobs만 생성했다. 최종 batch는 `960/960 complete`, failed `0`, missing `0`이다.

주요 산출물:

```text
artifacts/features/vjepa_2_1_full_observation_spatial12.pt
artifacts/features/vjepa_2_1_full_observation.pt
artifacts/features/ltx_2_3_full_real_future.pt
artifacts/features/ltx_2_3_full_rollout_future.pt
outputs/safety_query/full/result.json
outputs/vjepa_linear_probe/full/result.json
outputs/vjepa_transformer_probe/full/result.json
outputs/ltx_safety_query/full_prompt_v1/result.json
```

이 completion은 workload shard union의 결과다. Test의 endpoint/mirror component는 여전히 4개이므로,
새 simulator/route/scene에서 생성한 독립 held-out component 없이 population-level confirmatory claim을 하지
않는다.
