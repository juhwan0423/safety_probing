# Schema Reference

## 1. Directory Layout

```text
eccv_workshop_grid_part_XX/
├── configs/experiment_config.snapshot.json
├── scenarios/<scenario_id>.json
├── pairs/<pair_id>/
│   ├── assumptions.json
│   ├── metadata_pair.json
│   ├── run_A/
│   │   ├── metadata_run.json
│   │   ├── dense_log.csv
│   │   ├── command_log.csv
│   │   └── rgb/<camera_name>/frame_XXXXXX.png
│   └── run_B/...
└── summaries/dataset_summary.json
```

경로는 manifest에서 absolute path로 저장하지만, 학습 artifact를 다른 장비로 옮길 때는
`dataset_root`를 기준으로 다시 resolve해야 한다.

## 2. `metadata_pair.json`

| Field | Type | 의미 |
|---|---|---|
| `pair_id` | string | 전역 pair 식별자 |
| `route_id` | string | direction/speed/steering route |
| `mirror_group_id` | string | 좌우 mirror route 그룹 |
| `turn_direction` | enum | `left`, `right` |
| `speed_label` | enum | `slow`, `medium`, `fast` |
| `steering_label` | enum | `wide`, `medium`, `tight` |
| `seed` | integer | pair 공통 simulator seed |
| `intervention_type` | enum | `load_mass`, `com_shift` |
| `intervention_value_A/B` | number | A/B에 적용된 물리값 |
| `same_action_trajectory` | boolean | command sequence 동일 여부 |
| `summary_outcome_comparison` | object | A/B tip, failure time, roll/pitch delta |
| `quality_gate` | object | tier, accepted, reason |

`quality_gate.accepted`는 raw 파일 삭제 여부가 아니라 primary curated subset 포함 권고다.

## 3. `metadata_run.json`

| Field | 의미 |
|---|---|
| `run_id`, `run_label`, `pair_id` | run identity |
| `scenario_id`, `route_id`, `seed` | simulator condition |
| `intervention_type/value` | 실제 run에 적용된 intervention |
| `load_state` | realized mass, local CoM, size, spawn/attachment parameters |
| `initial_state` | settle 이후 forklift/load initial pose |
| `tip_over`, `tip_over_reason`, `failure_time` | simulator 원본 detector outcome; provenance 전용 |
| `max_roll_deg`, `max_pitch_deg` | 전체 5초 consequence summary |
| `min_wheel_contact_count` | episode 최소 contact count |
| `realized_motion_summary` | speed, yaw-rate, lateral-acceleration proxy summary |
| `config_snapshot` | 해당 run의 scene/route/logging/detector 설정 |
| `asset_identifiers` | runtime prim/sensor paths |

## 4. `dense_log.csv`

### Time, state, and intervention

| Columns | Unit/frame | 비고 |
|---|---|---|
| `episode_step` | integer | 0-359 |
| `sim_time` | s | 첫 logged row는 `1/72 s` |
| `episode_status` | enum | running, tip_over, completed |
| `event_reason` | string | detector event |
| `turn_direction` | enum | route metadata; target으로 직접 사용 시 shortcut 주의 |
| `intervention_type/value` | mixed | hidden variable의 정답; visual-only input에 사용 금지 |

### Forklift and load pose

| Columns | Unit/frame | 비고 |
|---|---|---|
| `forklift_pos_{x,y,z}` | m, world | moving base-link prim pose |
| `forklift_quat_{x,y,z,w}` | world orientation | quaternion order는 XYZW |
| `linear_vel_{x,y,z}` | m/s, world | world position finite difference |
| `angular_vel_{x,y,z}` | rad/s | Euler-angle finite difference |
| `roll_deg`, `pitch_deg`, `yaw_deg` | deg | forklift quaternion에서 계산 |
| `load_pos_{x,y,z}` | m, world | payload pose |
| `load_quat_{x,y,z,w}` | world orientation | XYZW |
| `joint_positions`, `joint_velocities` | JSON-like list | articulation joint order |

### Support polygon and IMU

| Columns | Unit/frame | 비고 |
|---|---|---|
| `wheel_contact_count` | count | active configured contact sensors |
| `wheel_contact_source` | string | runtime source description |
| `support_polygon_vertex_count` | count | convex hull vertex count |
| `support_polygon_xy` | JSON list, world m | active contact points의 XY hull |
| `imu_valid` | boolean | IMU fields 사용 가능 여부 |
| `imu_sensor_time`, `imu_physics_step` | sensor-native | Isaac sensor reading |
| `imu_lin_acc_{x,y,z}` | m/s², IMU sensor frame | world frame으로 별도 변환하지 않음 |
| `imu_ang_vel_{x,y,z}` | rad/s, IMU sensor frame | sensor-native |
| `imu_quat_{x,y,z,w}` | XYZW | sensor-native orientation reading |

### System CoM and ZMP

| Columns | Unit/frame | 비고 |
|---|---|---|
| `system_com_available` | boolean | system CoM fields valid |
| `system_mass_kg` | kg | forklift links + payload mass |
| `system_com_{x,y,z}` | m, world | body-mass weighted CoM |
| `system_com_vel_{x,y,z}` | m/s, world | finite difference |
| `system_com_acc_{x,y,z}` | m/s², world | finite difference |
| `system_com_component_count` | count | mass aggregation components |
| `zmp_valid` | boolean | flat-ground ZMP 계산 가능 여부 |
| `zmp_{x,y}` | m, world | flat-ground ZMP |
| `zmp_support_margin_m` | m | polygon 전체 경계 signed distance |
| `com_support_margin_m` | m | projected CoM의 전체 경계 signed distance |
| `com_outside_support_polygon` | boolean | projected CoM polygon 이탈 |
| `outward_direction_{x,y}` | unit vector, world | velocity와 yaw-rate에서 계산 |
| `zmp_outward_lateral_margin_m` | m | 회전 바깥쪽 boundary margin |
| `com_outward_lateral_margin_m` | m | projected CoM outward margin |
| `outward_lateral_margin_valid` | boolean | primary safety target mask |

Margin은 양수면 경계 안쪽, 음수면 경계 바깥쪽이다. Generic ZMP margin은 command transition spike를
포함할 수 있으므로 primary regression에는 `outward_lateral_margin_valid=true`인 directional margin을 쓴다.

### Per-contact fields

다섯 contact prefix에 동일한 11개 field가 반복된다.

```text
back_wheel
roller_back_left
roller_back_right
roller_front_left
roller_front_right
```

| Suffix | Unit/frame | 의미 |
|---|---|---|
| `_contact` | boolean | contact active |
| `_force_z` | N-like sensor output | vertical force reading |
| `_contact_pos_{x,y,z}` | m, world | raw contact mean 또는 body-position fallback |
| `_contact_normal_{x,y,z}` | world vector | raw contact normal, available할 때만 |
| `_contact_impulse_{x,y,z}` | impulse | raw sensor impulse, available할 때만 |

`_contact=true`인데 x/y가 비어 있으면 support polygon audit failure다.

## 5. `command_log.csv`

| Field | 의미 |
|---|---|
| `episode_step` | command가 적용된 physics step |
| `phase_name` | approach, turn, recover_straight |
| `joint_indices` | controlled articulation indices |
| `joint_positions` | target position list; `nan`은 해당 slot 미지정 |
| `joint_velocities` | target velocity list; `nan`은 해당 slot 미지정 |

## 6. Probing Manifest

`artifacts/current/manifests/runs.jsonl`은 각 run의 RGB/dense/metadata 경로, 파생 outcome label,
`condition_id`, quality tier, 세 split protocol assignment를 한 줄에 저장한다.

| Derived field | Type | 의미 |
|---|---|---|
| `label_schema_version` | string | 현재 `censor-aware-lateral-tip-v3` |
| `safety_outcome` | enum | `observed_tip`, `confirmed_recovered`, `right_censored` |
| `binary_tip_label` | 0/1/null | censored는 `null` |
| `event_observed`, `failure_time_s` | bool, s | event 관측 여부와 최초 확인 시간 |
| `censor_time_s`, `followup_time_s` | s/null | survival target용 시간 |
| `source_tip_over` | bool | raw metadata detector 결과 |
| `upgraded_from_source` | bool | dense support-loss rule로 승격됐는지 여부 |
| `future_prediction_eligible` | bool | 2.7083초(65-frame) 관측 prefix 기반 binary target 사용 가능 여부 |

`pairs.jsonl`의 `quality_tier`와 `tip_state_transition`은 파생 label로 다시 계산된다.
`source_quality_tier`, `source_tip_state_transition`에는 simulator metadata 값을 보존한다.

- Run-level classification/regression은 `canonical_runs.jsonl`에 선택된 동일 `condition_id`의 대표 sample
  하나만 사용한다. 선택은 label이나 quality가 아니라 `run_id` lexical order로 결정한다.
- Pairwise task는 `pairs.jsonl`의 pair context를 유지한다.
- `condition_id`는 route, seed, realized mass, local CoM, load geometry를 hash한 값이다.

## 7. Leakage Columns

Visual-only probe의 입력 feature에 다음 값은 포함하지 않는다.

- `intervention_type`, `intervention_value`, `load_state`
- roll/pitch/yaw, IMU, contact, CoM, ZMP
- `safety_outcome`, `binary_tip_label`, `tip_over`, `failure_time`, `quality_tier`
- split name이나 pair ID에서 파생한 numeric feature

이 값들은 target 생성, stratified reporting, audit에만 사용한다.
