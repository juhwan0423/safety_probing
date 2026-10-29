# Dataset Statistics

> 이 문서는 `safety-probing analyze`가 원본 shard를 읽어 자동 생성한다. 수기로 수정하지 않는다.

## Snapshot

- Dataset: `ExoFork-CF ECCV Workshop Lateral Tip-Over`
- Version: `v0.1-finite-steering-pulse`
- Generated at: `2026-07-13T04:44:55.469798+00:00`
- Generator commit: `2a8f790e8d07795cdd7de9cf00a9918911eaad54`
- Generator worktree dirty at audit: `True`
- Source config SHA-256: `e4d2da4417feafffedd4b4d10b3c5e10043bdecc9dc8f3edf041827b0c538468`
- Split manifest SHA-256: `882d26fa1d8b4eb200a62c753ae080a42ede2a8c65ccfc182ca02bb869be486c`
- Label schema: `censor-aware-lateral-tip-v3`
- Definitive event: `abs(roll) >= 20 deg` or fewer than 3 support contacts for 6 physics steps
- Right censoring: no definitive event, but terminal absolute roll remains outward-growing

## Completion

현재 접근 가능한 데이터는 전체 grid의 **100.00%**다.

| Dataset root | Available | Planned pairs | Completed pairs | Completed runs |
|---|---:|---:|---:|---:|
| `/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/eccv_workshop_grid_part_01` | yes | 604 | 604 | 1,208 |
| `/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/eccv_workshop_grid_part_02` | yes | 302 | 302 | 604 |
| `/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/eccv_workshop_grid_part_03` | yes | 102 | 102 | 204 |

- Completed pairs: **1,008 / 1,008**
- Completed runs: **2,016 / 2,016**
- Incomplete pair directories: **0**
- Duplicate pair IDs across roots: **0**

## Counterfactual Pair Outcomes

| Transition | Pairs | Fraction |
|---|---:|---:|
| `A_recovered_to_B_tip` | 29 | 2.88% |
| `A_tip_to_B_recovered` | 75 | 7.44% |
| `both_recovered` | 366 | 36.31% |
| `both_tip` | 504 | 50.00% |
| `confirmed_recovered_to_right_censored` | 6 | 0.60% |
| `observed_tip_to_right_censored` | 7 | 0.69% |
| `right_censored_to_confirmed_recovered` | 3 | 0.30% |
| `right_censored_to_observed_tip` | 18 | 1.79% |

| Quality tier | Pairs | Fraction |
|---|---:|---:|
| `bronze` | 85 | 8.43% |
| `censored` | 34 | 3.37% |
| `gold` | 104 | 10.32% |
| `reject` | 566 | 56.15% |
| `silver` | 219 | 21.73% |

Accepted (`gold/silver/bronze`): **408 (40.48%)**. Primary pairwise experiments should not train on the raw 1,008-pair set without applying this quality policy.

## Run Outcomes

- Observed tip/instability runs: **1,137**
- Confirmed recovered runs: **845**
- Right-censored runs: **34**
- Tip rate among definitive labels: **57.37%**
- Source tips upgraded by dense-log rules: **192**
- Failure time median: **3.375 s**
- Failure time range: `2.958` to `4.694` s

| Direction | Recovered | Observed tip | Censored | Tip rate |
|---|---:|---:|---:|---:|
| `left` | 415 | 559 | 34 | 57.39% |
| `right` | 430 | 578 | 0 | 57.34% |

| Speed | Recovered | Observed tip | Censored | Tip rate |
|---|---:|---:|---:|---:|
| `fast` | 15 | 653 | 4 | 97.75% |
| `medium` | 158 | 484 | 30 | 75.39% |
| `slow` | 672 | 0 | 0 | 0.00% |

| Steering | Recovered | Observed tip | Censored | Tip rate |
|---|---:|---:|---:|---:|
| `medium` | 285 | 363 | 0 | 56.02% |
| `tight` | 294 | 372 | 18 | 55.86% |
| `wide` | 266 | 402 | 16 | 60.18% |

### Preliminary observations

- Speed separates regimes strongly: slow `0.00%`, medium `75.39%`, fast `97.75%` tip rate.
- Steering label is not monotonic in observed tip rate: wide `60.18%`, medium `56.02%`, tight `55.86%`. The model and analysis must use realized dynamics rather than assuming `tight > wide` risk.
- Direction gap is left `57.39%` versus right `57.34%`; mirror-generalization reporting is therefore necessary.
- These observations cover only currently available shards and must be recomputed on the full grid.

## Intervention Statistics

| Intervention | Pairs | Gold | Silver | Bronze | Reject | Censored |
|---|---:|---:|---:|---:|---:|---:|
| `com_shift` | 360 | 30 | 90 | 42 | 186 | 12 |
| `load_mass` | 648 | 74 | 129 | 43 | 380 | 22 |

## Route Statistics

| Route | Pairs | Runs | Tip | Censored runs | Tip rate | Gold | Silver | Bronze | Reject | Censored pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `left_vfast_smedium` | 54 | 108 | 102 | 0 | 94.44% | 6 | 9 | 6 | 33 | 0 |
| `left_vfast_stight` | 57 | 114 | 111 | 3 | 100.00% | 0 | 18 | 6 | 30 | 3 |
| `left_vfast_swide` | 57 | 114 | 113 | 1 | 100.00% | 0 | 11 | 6 | 39 | 1 |
| `left_vmedium_smedium` | 54 | 108 | 78 | 0 | 72.22% | 18 | 24 | 0 | 12 | 0 |
| `left_vmedium_stight` | 57 | 114 | 72 | 15 | 72.73% | 15 | 24 | 0 | 3 | 15 |
| `left_vmedium_swide` | 57 | 114 | 83 | 15 | 83.84% | 7 | 26 | 6 | 3 | 15 |
| `left_vslow_smedium` | 54 | 108 | 0 | 0 | 0.00% | 0 | 0 | 6 | 48 | 0 |
| `left_vslow_stight` | 57 | 114 | 0 | 0 | 0.00% | 0 | 0 | 0 | 57 | 0 |
| `left_vslow_swide` | 57 | 114 | 0 | 0 | 0.00% | 0 | 0 | 9 | 48 | 0 |
| `right_vfast_smedium` | 54 | 108 | 105 | 0 | 97.22% | 3 | 12 | 0 | 39 | 0 |
| `right_vfast_stight` | 57 | 114 | 108 | 0 | 94.74% | 6 | 9 | 0 | 42 | 0 |
| `right_vfast_swide` | 57 | 114 | 114 | 0 | 100.00% | 0 | 12 | 0 | 45 | 0 |
| `right_vmedium_smedium` | 54 | 108 | 78 | 0 | 72.22% | 18 | 24 | 3 | 9 | 0 |
| `right_vmedium_stight` | 57 | 114 | 81 | 0 | 71.05% | 21 | 18 | 3 | 15 | 0 |
| `right_vmedium_swide` | 57 | 114 | 92 | 0 | 80.70% | 10 | 32 | 3 | 12 | 0 |
| `right_vslow_smedium` | 54 | 108 | 0 | 0 | 0.00% | 0 | 0 | 12 | 42 | 0 |
| `right_vslow_stight` | 57 | 114 | 0 | 0 | 0.00% | 0 | 0 | 10 | 47 | 0 |
| `right_vslow_swide` | 57 | 114 | 0 | 0 | 0.00% | 0 | 0 | 15 | 42 | 0 |

## Dense Log Integrity

- Scanned rows: **725,760**
- Rows/run median: **360**
- Wrong row count runs: **0**
- Missing dense logs: **0**
- Missing required columns: **0 runs**

| Signal/mask | True rows | Total rows | Fraction |
|---|---:|---:|---:|
| `imu_valid` | 725,760 | 725,760 | 100.000% |
| `outward_lateral_margin_valid` | 206,259 | 725,760 | 28.420% |
| `system_com_available` | 725,760 | 725,760 | 100.000% |
| `zmp_valid` | 725,760 | 725,760 | 100.000% |

- Support polygon valid row fraction: **82.127%**
- Active contact samples with complete x/y: **100.000%**
- Per-run minimum outward ZMP margin median: **-0.427 m**

`outward_lateral_margin_valid`은 전체 frame의 완전성 지표가 아니다. 직진 구간, 조향 transition grace, yaw-rate gate, tip 이후를 의도적으로 제외하므로 해당 mask가 `true`인 row만 safety target에 사용한다.

### Target diagnostic

| Target | Stable median | Tip median | Stable negative | Tip negative | Tip AUROC from `-margin` |
|---|---:|---:|---:|---:|---:|
| `future_min_outward_zmp_margin_m` | -0.339 | -0.480 | 100.000% | 100.000% | 0.847 |
| `min_outward_com_margin_m` | 0.112 | 0.017 | 4.615% | 30.431% | 0.953 |

현재 ZMP minimum은 stable과 tip 모두 음수이므로 `margin < 0`을 binary tip criterion으로 쓰면 안 된다. ZMP는 continuous regression/ranking target으로 사용하고, near-boundary threshold가 필요하면 train split에서만 calibration한다. Projected CoM outward margin의 부호가 더 직접적인 support-boundary diagnostic이지만 이것도 binary detector를 대체하지 않는다.

## RGB Integrity

| Camera | Runs | Min frames | Median | Max frames |
|---|---:|---:|---:|---:|
| `main_fixed_northwest` | 2,016 | 120 | 120 | 120 |
| `top_fixed_debug` | 2,016 | 120 | 120 | 120 |

Frame-count mismatches: **0**. 이 검사는 파일 개수만 확인하며, 아래 pixel 검사는 deterministic subset의 시작·중간·끝 frame을 확인한다.

- Pixel-audited images: **384**
- Near-blank sampled images: **0**
- Wrong-resolution sampled images: **0**
- Duplicate sampled timestamps: **0**
- Decode failures: **0**

Pixel audit은 전수 검사가 아니라 deterministic sample audit이다. 최종 dataset release 전에는 sample 수를 늘리고 perceptual duplicate 검사까지 추가한다.

## Split Audit

기존 split은 pair ID만 묶었기 때문에 인접 mass/CoM pair가 공유하는 실제 endpoint가 train/validation/test에 반복될 수 있었다. 새 split은 shared endpoint graph의 connected component를 최소 split 단위로 사용한다.

| Protocol | Local train | Local val | Local test | Reference leaks | New leaks |
|---|---:|---:|---:|---:|---:|
| `main` | 710 | 148 | 150 | 410 | 0 |
| `left_to_right` | 430 | 74 | 504 | 115 | 0 |
| `right_to_left` | 429 | 75 | 504 | 115 | 0 |

### Local main-split label balance

| Split | Pairs | Gold pairs | Recovered | Tip | Censored | Tip rate |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 710 | 60 | 618 | 774 | 28 | 55.60% |
| `val` | 148 | 24 | 116 | 180 | 0 | 60.81% |
| `test` | 150 | 20 | 111 | 183 | 6 | 62.24% |

현재 표는 available shard만 반영한다. 모든 shard를 합친 뒤 component leakage를 유지하면서 label/route balance가 과도하게 치우치지 않았는지 확인하고 최종 split hash를 freeze한다.

실제로 발견된 duplicate physical condition group은 **738개**이며, 새 main split을 넘는 중복은 **0개**다.
동일 condition 반복 실행 사이 tip label 불일치는 **6개 group**이다.

## Probing Readiness

- Pair manifest: `/home/jude/isaac_ws/safety_probing/artifacts/current/manifests/pairs.jsonl`
- Run manifest: `/home/jude/isaac_ws/safety_probing/artifacts/current/manifests/runs.jsonl`
- Canonical deduplicated run manifest: `/home/jude/isaac_ws/safety_probing/artifacts/current/manifests/canonical_runs.jsonl` (810 conditions)
- Primary visual input: `main_fixed_northwest`
- Primary observation window: `0.0-2.7083 s` (65 frames at 24 FPS)
- Primary future target window: `2.7083-5.0 s`
- Primary pairwise subset: `gold`, with `silver/bronze` used for margin regression/control
- `reject` pairs are retained in raw storage but excluded from primary probing

## Current Interpretation

이 snapshot은 접근 가능한 shard만 반영한다. 나머지 shard가 합쳐지면 같은 명령으로 문서를 다시 생성하고, full-grid 통계를 논문 수치로 고정해야 한다. 현재 수치는 pipeline과 분포 이상을 점검하는 preliminary statistics다.
