# Dataset Statistics

> 이 문서는 `safety-probing analyze`가 원본 shard를 읽어 자동 생성한다. 수기로 수정하지 않는다.

## Snapshot

- Dataset: `ExoFork-CF ECCV Workshop Lateral Tip-Over`
- Version: `v0.1-finite-steering-pulse`
- Generated at: `2026-07-11T11:13:29.607387+00:00`
- Generator commit: `2a8f790e8d07795cdd7de9cf00a9918911eaad54`
- Generator worktree dirty at audit: `True`
- Source config SHA-256: `e4d2da4417feafffedd4b4d10b3c5e10043bdecc9dc8f3edf041827b0c538468`
- Split manifest SHA-256: `00c7321b3c056a8ac42542928808c6c1f3f00461620d8b08c13be471439150dc`
- Label schema: `censor-aware-lateral-tip-v2`
- Definitive event: `abs(roll) >= 20 deg` or fewer than 3 support contacts for 6 physics steps
- Right censoring: no definitive event, but terminal absolute roll remains outward-growing

## Completion

현재 접근 가능한 데이터는 전체 grid의 **59.92%**다.

| Dataset root | Available | Planned pairs | Completed pairs | Completed runs |
|---|---:|---:|---:|---:|
| `/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/eccv_workshop_grid_part_01` | yes | 604 | 604 | 1,208 |
| `/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/eccv_workshop_grid_part_02` | no | N/A | 0 | 0 |
| `/home/jude/isaac_ws/ExoFork-CF/project_march/sim_project/dataset_root/eccv_workshop_grid_part_03` | no | N/A | 0 | 0 |

- Completed pairs: **604 / 1,008**
- Completed runs: **1,208 / 2,016**
- Incomplete pair directories: **0**
- Duplicate pair IDs across roots: **0**

## Counterfactual Pair Outcomes

| Transition | Pairs | Fraction |
|---|---:|---:|
| `A_recovered_to_B_tip` | 19 | 3.15% |
| `A_tip_to_B_recovered` | 39 | 6.46% |
| `both_recovered` | 218 | 36.09% |
| `both_tip` | 308 | 50.99% |
| `confirmed_recovered_to_right_censored` | 5 | 0.83% |
| `observed_tip_to_right_censored` | 4 | 0.66% |
| `right_censored_to_confirmed_recovered` | 2 | 0.33% |
| `right_censored_to_observed_tip` | 9 | 1.49% |

| Quality tier | Pairs | Fraction |
|---|---:|---:|
| `bronze` | 53 | 8.77% |
| `censored` | 20 | 3.31% |
| `gold` | 58 | 9.60% |
| `reject` | 336 | 55.63% |
| `silver` | 137 | 22.68% |

Accepted (`gold/silver/bronze`): **248 (41.06%)**. Primary pairwise experiments should not train on the raw 1,008-pair set without applying this quality policy.

## Run Outcomes

- Observed tip/instability runs: **687**
- Confirmed recovered runs: **501**
- Right-censored runs: **20**
- Tip rate among definitive labels: **57.83%**
- Source tips upgraded by dense-log rules: **112**
- Failure time median: **3.347 s**
- Failure time range: `2.958` to `4.694` s

| Direction | Recovered | Observed tip | Censored | Tip rate |
|---|---:|---:|---:|---:|
| `left` | 248 | 336 | 20 | 57.53% |
| `right` | 253 | 351 | 0 | 58.11% |

| Speed | Recovered | Observed tip | Censored | Tip rate |
|---|---:|---:|---:|---:|
| `fast` | 9 | 389 | 2 | 97.74% |
| `medium` | 92 | 298 | 18 | 76.41% |
| `slow` | 400 | 0 | 0 | 0.00% |

| Steering | Recovered | Observed tip | Censored | Tip rate |
|---|---:|---:|---:|---:|
| `medium` | 180 | 220 | 0 | 55.00% |
| `tight` | 154 | 207 | 11 | 57.34% |
| `wide` | 167 | 260 | 9 | 60.89% |

### Preliminary observations

- Speed separates regimes strongly: slow `0.00%`, medium `76.41%`, fast `97.74%` tip rate.
- Steering label is not monotonic in observed tip rate: wide `60.89%`, medium `55.00%`, tight `57.34%`. The model and analysis must use realized dynamics rather than assuming `tight > wide` risk.
- Direction gap is left `57.53%` versus right `58.11%`; mirror-generalization reporting is therefore necessary.
- These observations cover only currently available shards and must be recomputed on the full grid.

## Intervention Statistics

| Intervention | Pairs | Gold | Silver | Bronze | Reject | Censored |
|---|---:|---:|---:|---:|---:|---:|
| `com_shift` | 216 | 15 | 57 | 27 | 111 | 6 |
| `load_mass` | 388 | 43 | 80 | 26 | 225 | 14 |

## Route Statistics

| Route | Pairs | Runs | Tip | Censored runs | Tip rate | Gold | Silver | Bronze | Reject | Censored pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `left_vfast_smedium` | 34 | 68 | 63 | 0 | 92.65% | 5 | 6 | 4 | 19 | 0 |
| `left_vfast_stight` | 31 | 62 | 60 | 2 | 100.00% | 0 | 11 | 2 | 16 | 2 |
| `left_vfast_swide` | 35 | 70 | 70 | 0 | 100.00% | 0 | 7 | 2 | 26 | 0 |
| `left_vmedium_smedium` | 34 | 68 | 46 | 0 | 67.65% | 12 | 15 | 0 | 7 | 0 |
| `left_vmedium_stight` | 31 | 62 | 38 | 9 | 71.70% | 8 | 13 | 0 | 1 | 9 |
| `left_vmedium_swide` | 37 | 74 | 59 | 9 | 90.77% | 4 | 19 | 3 | 2 | 9 |
| `left_vslow_smedium` | 32 | 64 | 0 | 0 | 0.00% | 0 | 0 | 5 | 27 | 0 |
| `left_vslow_stight` | 31 | 62 | 0 | 0 | 0.00% | 0 | 0 | 0 | 31 | 0 |
| `left_vslow_swide` | 37 | 74 | 0 | 0 | 0.00% | 0 | 0 | 7 | 30 | 0 |
| `right_vfast_smedium` | 34 | 68 | 66 | 0 | 97.06% | 2 | 10 | 0 | 22 | 0 |
| `right_vfast_stight` | 31 | 62 | 60 | 0 | 96.77% | 2 | 3 | 0 | 26 | 0 |
| `right_vfast_swide` | 35 | 70 | 70 | 0 | 100.00% | 0 | 6 | 0 | 29 | 0 |
| `right_vmedium_smedium` | 34 | 68 | 45 | 0 | 66.18% | 13 | 12 | 3 | 6 | 0 |
| `right_vmedium_stight` | 31 | 62 | 49 | 0 | 79.03% | 9 | 13 | 2 | 7 | 0 |
| `right_vmedium_swide` | 37 | 74 | 61 | 0 | 82.43% | 3 | 22 | 3 | 9 | 0 |
| `right_vslow_smedium` | 32 | 64 | 0 | 0 | 0.00% | 0 | 0 | 6 | 26 | 0 |
| `right_vslow_stight` | 31 | 62 | 0 | 0 | 0.00% | 0 | 0 | 4 | 27 | 0 |
| `right_vslow_swide` | 37 | 74 | 0 | 0 | 0.00% | 0 | 0 | 12 | 25 | 0 |

## Dense Log Integrity

- Scanned rows: **434,880**
- Rows/run median: **360**
- Wrong row count runs: **0**
- Missing dense logs: **0**
- Missing required columns: **0 runs**

| Signal/mask | True rows | Total rows | Fraction |
|---|---:|---:|---:|
| `imu_valid` | 434,880 | 434,880 | 100.000% |
| `outward_lateral_margin_valid` | 123,413 | 434,880 | 28.379% |
| `system_com_available` | 434,880 | 434,880 | 100.000% |
| `zmp_valid` | 434,880 | 434,880 | 100.000% |

- Support polygon valid row fraction: **81.916%**
- Active contact samples with complete x/y: **100.000%**
- Per-run minimum outward ZMP margin median: **-0.430 m**

`outward_lateral_margin_valid`은 전체 frame의 완전성 지표가 아니다. 직진 구간, 조향 transition grace, yaw-rate gate, tip 이후를 의도적으로 제외하므로 해당 mask가 `true`인 row만 safety target에 사용한다.

### Target diagnostic

| Target | Stable median | Tip median | Stable negative | Tip negative | Tip AUROC from `-margin` |
|---|---:|---:|---:|---:|---:|
| `future_min_outward_zmp_margin_m` | -0.347 | -0.489 | 100.000% | 100.000% | 0.851 |
| `min_outward_com_margin_m` | 0.112 | 0.016 | 3.792% | 29.840% | 0.961 |

현재 ZMP minimum은 stable과 tip 모두 음수이므로 `margin < 0`을 binary tip criterion으로 쓰면 안 된다. ZMP는 continuous regression/ranking target으로 사용하고, near-boundary threshold가 필요하면 train split에서만 calibration한다. Projected CoM outward margin의 부호가 더 직접적인 support-boundary diagnostic이지만 이것도 binary detector를 대체하지 않는다.

## RGB Integrity

| Camera | Runs | Min frames | Median | Max frames |
|---|---:|---:|---:|---:|
| `main_fixed_northwest` | 1,208 | 120 | 120 | 120 |
| `top_fixed_debug` | 1,208 | 120 | 120 | 120 |

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
| `main` | 426 | 84 | 94 | 410 | 0 |
| `left_to_right` | 254 | 48 | 302 | 115 | 0 |
| `right_to_left` | 259 | 43 | 302 | 115 | 0 |

### Local main-split label balance

| Split | Pairs | Gold pairs | Recovered | Tip | Censored | Tip rate |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 426 | 31 | 363 | 473 | 16 | 56.58% |
| `val` | 84 | 17 | 67 | 101 | 0 | 60.12% |
| `test` | 94 | 10 | 71 | 113 | 4 | 61.41% |

현재 표는 available shard만 반영한다. 모든 shard를 합친 뒤 component leakage를 유지하면서 label/route balance가 과도하게 치우치지 않았는지 확인하고 최종 split hash를 freeze한다.

실제로 발견된 duplicate physical condition group은 **428개**이며, 새 main split을 넘는 중복은 **0개**다.
동일 condition 반복 실행 사이 tip label 불일치는 **0개 group**이다.

## Probing Readiness

- Pair manifest: `/home/jude/isaac_ws/safety_probing/artifacts/current/manifests/pairs.jsonl`
- Run manifest: `/home/jude/isaac_ws/safety_probing/artifacts/current/manifests/runs.jsonl`
- Canonical deduplicated run manifest: `/home/jude/isaac_ws/safety_probing/artifacts/current/manifests/canonical_runs.jsonl` (670 conditions)
- Primary visual input: `main_fixed_northwest`
- Primary observation window: `0.0-2.375 s` (57 frames at 24 FPS)
- Primary future target window: `2.375-5.0 s`
- Primary pairwise subset: `gold`, with `silver/bronze` used for margin regression/control
- `reject` pairs are retained in raw storage but excluded from primary probing

## Current Interpretation

이 snapshot은 접근 가능한 shard만 반영한다. 나머지 shard가 합쳐지면 같은 명령으로 문서를 다시 생성하고, full-grid 통계를 논문 수치로 고정해야 한다. 현재 수치는 pipeline과 분포 이상을 점검하는 preliminary statistics다.
