from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .io_utils import write_text_atomic


PROTOCOL_NAMES = ("main", "left_to_right", "right_to_left")


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.2f}%"


def _fraction(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.3f}%"


def _distribution_rows(counts: dict[str, int], total: int) -> list[str]:
    return [
        f"| `{name}` | {count:,} | {_percent(100.0 * count / total if total else None)} |"
        for name, count in counts.items()
    ]


def render_statistics_markdown(stats: dict[str, Any]) -> str:
    dataset = stats["dataset"]
    completion = stats["completion"]
    pairs = stats["pairs"]
    runs = stats["runs"]
    dense = stats["dense_logs"]
    rgb = stats["rgb"]
    splits = stats["splits"]
    provenance = stats["provenance"]

    def outcome_rate(outcomes: dict[str, int]) -> float | None:
        tip = outcomes.get("observed_tip", 0)
        total = tip + outcomes.get("confirmed_recovered", 0)
        return 100.0 * tip / total if total else None

    lines = [
        "# Dataset Statistics",
        "",
        "> 이 문서는 `safety-probing analyze`가 원본 shard를 읽어 자동 생성한다. 수기로 수정하지 않는다.",
        "",
        "## Snapshot",
        "",
        f"- Dataset: `{dataset['name']}`",
        f"- Version: `{dataset['version']}`",
        f"- Generated at: `{stats['generated_at_utc']}`",
        f"- Generator commit: `{provenance['source_git'].get('commit')}`",
        f"- Generator worktree dirty at audit: `{provenance['source_git'].get('dirty')}`",
        f"- Source config SHA-256: `{provenance['source_config_sha256']}`",
        f"- Split manifest SHA-256: `{provenance['split_manifest_sha256']}`",
        f"- Label schema: `{dataset['labeling']['schema_version']}`",
        "- Definitive event: `abs(roll) >= 20 deg` or fewer than 3 support contacts for 6 physics steps",
        "- Right censoring: no definitive event, but terminal absolute roll remains outward-growing",
        "",
        "## Completion",
        "",
        f"현재 접근 가능한 데이터는 전체 grid의 **{_percent(completion['completed_pair_percent_of_full_grid'])}**다.",
        "",
        "| Dataset root | Available | Planned pairs | Completed pairs | Completed runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in completion["dataset_roots"]:
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                row["dataset_root"],
                "yes" if row["available"] else "no",
                _number(row["planned_pair_count"]),
                _number(row["completed_pair_count"]),
                _number(row["completed_run_count"]),
            )
        )

    lines.extend(
        [
            "",
            f"- Completed pairs: **{pairs['count']:,} / {dataset['expected_pair_count']:,}**",
            f"- Completed runs: **{runs['count']:,} / {dataset['expected_run_count']:,}**",
            f"- Incomplete pair directories: **{completion['incomplete_pair_directory_count']:,}**",
            f"- Duplicate pair IDs across roots: **{completion['duplicate_pair_id_count']:,}**",
            "",
            "## Counterfactual Pair Outcomes",
            "",
            "| Transition | Pairs | Fraction |",
            "|---|---:|---:|",
            *_distribution_rows(pairs["tip_state_transition_counts"], pairs["count"]),
            "",
            "| Quality tier | Pairs | Fraction |",
            "|---|---:|---:|",
            *_distribution_rows(pairs["quality_tier_counts"], pairs["count"]),
            "",
            f"Accepted (`gold/silver/bronze`): **{pairs['accepted_count']:,} "
            f"({_percent(pairs['accepted_percent'])})**. Primary pairwise experiments should not train on "
            "the raw 1,008-pair set without applying this quality policy.",
            "",
            "## Run Outcomes",
            "",
            f"- Observed tip/instability runs: **{runs['tip_count']:,}**",
            f"- Confirmed recovered runs: **{runs['stable_count']:,}**",
            f"- Right-censored runs: **{runs['censored_count']:,}**",
            f"- Tip rate among definitive labels: **{_percent(runs['tip_rate_percent'])}**",
            f"- Source tips upgraded by dense-log rules: **{runs['posthoc_upgraded_tip_count']:,}**",
            f"- Failure time median: **{_number(runs['failure_time_s'].get('median'))} s**",
            f"- Failure time range: `{_number(runs['failure_time_s'].get('min'))}` to "
            f"`{_number(runs['failure_time_s'].get('max'))}` s",
            "",
            "| Direction | Recovered | Observed tip | Censored | Tip rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for direction, outcomes in runs["direction_outcomes"].items():
        stable = outcomes.get("confirmed_recovered", 0)
        tip = outcomes.get("observed_tip", 0)
        censored = outcomes.get("right_censored", 0)
        lines.append(
            f"| `{direction}` | {stable:,} | {tip:,} | {censored:,} | "
            f"{_percent(100.0 * tip / (stable + tip) if stable + tip else None)} |"
        )

    lines.extend(
        [
            "",
            "| Speed | Recovered | Observed tip | Censored | Tip rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for speed, outcomes in runs["speed_outcomes"].items():
        stable = outcomes.get("confirmed_recovered", 0)
        tip = outcomes.get("observed_tip", 0)
        censored = outcomes.get("right_censored", 0)
        lines.append(
            f"| `{speed}` | {stable:,} | {tip:,} | {censored:,} | "
            f"{_percent(100.0 * tip / (stable + tip) if stable + tip else None)} |"
        )

    lines.extend(
        [
            "",
            "| Steering | Recovered | Observed tip | Censored | Tip rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for steering, outcomes in runs["steering_outcomes"].items():
        stable = outcomes.get("confirmed_recovered", 0)
        tip = outcomes.get("observed_tip", 0)
        censored = outcomes.get("right_censored", 0)
        lines.append(
            f"| `{steering}` | {stable:,} | {tip:,} | {censored:,} | "
            f"{_percent(100.0 * tip / (stable + tip) if stable + tip else None)} |"
        )

    speed_rates = {
        name: outcome_rate(outcomes) for name, outcomes in runs["speed_outcomes"].items()
    }
    steering_rates = {
        name: outcome_rate(outcomes)
        for name, outcomes in runs["steering_outcomes"].items()
    }
    direction_rates = {
        name: outcome_rate(outcomes)
        for name, outcomes in runs["direction_outcomes"].items()
    }
    lines.extend(
        [
            "",
            "### Preliminary observations",
            "",
            f"- Speed separates regimes strongly: slow `{_percent(speed_rates.get('slow'))}`, "
            f"medium `{_percent(speed_rates.get('medium'))}`, fast `{_percent(speed_rates.get('fast'))}` tip rate.",
            f"- Steering label is not monotonic in observed tip rate: wide `{_percent(steering_rates.get('wide'))}`, "
            f"medium `{_percent(steering_rates.get('medium'))}`, tight `{_percent(steering_rates.get('tight'))}`. "
            "The model and analysis must use realized dynamics rather than assuming `tight > wide` risk.",
            f"- Direction gap is left `{_percent(direction_rates.get('left'))}` versus right "
            f"`{_percent(direction_rates.get('right'))}`; mirror-generalization reporting is therefore necessary.",
            "- These observations cover only currently available shards and must be recomputed on the full grid.",
        ]
    )

    lines.extend(
        [
            "",
            "## Intervention Statistics",
            "",
            "| Intervention | Pairs | Gold | Silver | Bronze | Reject | Censored |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for intervention, row in pairs["intervention_breakdown"].items():
        tiers = row["quality_tier_counts"]
        count = pairs["intervention_type_counts"][intervention]
        lines.append(
            f"| `{intervention}` | {count:,} | {tiers.get('gold', 0):,} | "
            f"{tiers.get('silver', 0):,} | {tiers.get('bronze', 0):,} | "
            f"{tiers.get('reject', 0):,} | {tiers.get('censored', 0):,} |"
        )

    lines.extend(
        [
            "",
            "## Route Statistics",
            "",
            "| Route | Pairs | Runs | Tip | Censored runs | Tip rate | Gold | Silver | Bronze | Reject | Censored pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for route_id, row in pairs["by_route"].items():
        tiers = row["quality_tier_counts"]
        censored = row["safety_outcome_counts"].get("right_censored", 0)
        lines.append(
            f"| `{route_id}` | {row['pair_count']:,} | {row['run_count']:,} | "
            f"{row['tip_run_count']:,} | {censored:,} | {_percent(row['tip_rate_percent'])} | "
            f"{tiers.get('gold', 0):,} | {tiers.get('silver', 0):,} | "
            f"{tiers.get('bronze', 0):,} | {tiers.get('reject', 0):,} | "
            f"{tiers.get('censored', 0):,} |"
        )

    lines.extend(
        [
            "",
            "## Dense Log Integrity",
            "",
            f"- Scanned rows: **{dense['total_row_count']:,}**",
            f"- Rows/run median: **{_number(dense['rows_per_run'].get('median'), 0)}**",
            f"- Wrong row count runs: **{dense['row_count_mismatch_run_count']:,}**",
            f"- Missing dense logs: **{dense['missing_dense_log_run_count']:,}**",
            f"- Missing required columns: **{dense['missing_required_columns_run_count']:,} runs**",
            "",
            "| Signal/mask | True rows | Total rows | Fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in dense["signal_validity"].items():
        lines.append(
            f"| `{name}` | {row['true_rows']:,} | {row['total_rows']:,} | {_fraction(row['fraction'])} |"
        )
    lines.extend(
        [
            "",
            f"- Support polygon valid row fraction: **{_fraction(dense['support_polygon_valid_row_fraction'])}**",
            f"- Active contact samples with complete x/y: **{_fraction(dense['active_contact_xy_complete_fraction'])}**",
            f"- Per-run minimum outward ZMP margin median: "
            f"**{_number(dense['per_run_min_outward_zmp_margin_m'].get('median'))} m**",
            "",
            "`outward_lateral_margin_valid`은 전체 frame의 완전성 지표가 아니다. 직진 구간, 조향 transition grace, "
            "yaw-rate gate, tip 이후를 의도적으로 제외하므로 해당 mask가 `true`인 row만 safety target에 사용한다.",
            "",
            "### Target diagnostic",
            "",
            "| Target | Stable median | Tip median | Stable negative | Tip negative | Tip AUROC from `-margin` |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in (
        "future_min_outward_zmp_margin_m",
        "min_outward_com_margin_m",
    ):
        row = dense["target_diagnostics"][metric]
        lines.append(
            f"| `{metric}` | {_number(row['stable'].get('median'))} | "
            f"{_number(row['tip'].get('median'))} | "
            f"{_fraction(row['stable'].get('negative_fraction'))} | "
            f"{_fraction(row['tip'].get('negative_fraction'))} | "
            f"{_number(row.get('tip_auc_using_negative_margin_as_risk'))} |"
        )
    lines.extend(
        [
            "",
            "현재 ZMP minimum은 stable과 tip 모두 음수이므로 `margin < 0`을 binary tip criterion으로 쓰면 "
            "안 된다. ZMP는 continuous regression/ranking target으로 사용하고, near-boundary threshold가 "
            "필요하면 train split에서만 calibration한다. Projected CoM outward margin의 부호가 더 직접적인 "
            "support-boundary diagnostic이지만 이것도 binary detector를 대체하지 않는다.",
            "",
            "## RGB Integrity",
            "",
            "| Camera | Runs | Min frames | Median | Max frames |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for camera_name, row in rgb["camera_frame_counts"].items():
        lines.append(
            f"| `{camera_name}` | {row.get('count', 0):,} | {_number(row.get('min'), 0)} | "
            f"{_number(row.get('median'), 0)} | {_number(row.get('max'), 0)} |"
        )
    lines.extend(
        [
            "",
            f"Frame-count mismatches: **{rgb['frame_count_mismatch_count']:,}**. 이 검사는 파일 개수만 확인하며, "
            "아래 pixel 검사는 deterministic subset의 시작·중간·끝 frame을 확인한다.",
            "",
            f"- Pixel-audited images: **{rgb['pixel_audit'].get('audited_image_count', 0):,}**",
            f"- Near-blank sampled images: **{rgb['pixel_audit'].get('near_blank_image_count', 0):,}**",
            f"- Wrong-resolution sampled images: **{rgb['pixel_audit'].get('wrong_resolution_image_count', 0):,}**",
            f"- Duplicate sampled timestamps: **{rgb['pixel_audit'].get('duplicate_time_sample_count', 0):,}**",
            f"- Decode failures: **{rgb['pixel_audit'].get('decode_failure_count', 0):,}**",
            "",
            "Pixel audit은 전수 검사가 아니라 deterministic sample audit이다. 최종 dataset release 전에는 "
            "sample 수를 늘리고 perceptual duplicate 검사까지 추가한다.",
            "",
            "## Split Audit",
            "",
            "기존 split은 pair ID만 묶었기 때문에 인접 mass/CoM pair가 공유하는 실제 endpoint가 "
            "train/validation/test에 반복될 수 있었다. 새 split은 shared endpoint graph의 connected component를 "
            "최소 split 단위로 사용한다.",
            "",
            "| Protocol | Local train | Local val | Local test | Reference leaks | New leaks |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol_name in PROTOCOL_NAMES:
        local = splits["local_pair_counts"][protocol_name]
        reference = splits["reference_audit"].get(protocol_name, {})
        generated = splits["generated_audit"].get(protocol_name, {})
        lines.append(
            f"| `{protocol_name}` | {local.get('train', 0):,} | {local.get('val', 0):,} | "
            f"{local.get('test', 0):,} | "
            f"{reference.get('cross_split_physical_condition_count', 'N/A')} | "
            f"{generated.get('cross_split_physical_condition_count', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "### Local main-split label balance",
            "",
            "| Split | Pairs | Gold pairs | Recovered | Tip | Censored | Tip rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in ("train", "val", "test"):
        pair_count = splits["local_pair_counts"]["main"].get(split_name, 0)
        gold = splits["local_quality_tier_counts"]["main"].get(split_name, {}).get("gold", 0)
        outcomes = splits["local_run_outcomes"]["main"].get(split_name, {})
        stable = outcomes.get("confirmed_recovered", 0)
        tip = outcomes.get("observed_tip", 0)
        censored = outcomes.get("right_censored", 0)
        lines.append(
            f"| `{split_name}` | {pair_count:,} | {gold:,} | {stable:,} | {tip:,} | {censored:,} | "
            f"{_percent(100.0 * tip / (stable + tip) if stable + tip else None)} |"
        )
    lines.extend(
        [
            "",
            "현재 표는 available shard만 반영한다. 모든 shard를 합친 뒤 component leakage를 유지하면서 "
            "label/route balance가 과도하게 치우치지 않았는지 확인하고 최종 split hash를 freeze한다.",
        ]
    )

    lines.extend(
        [
            "",
            f"실제로 발견된 duplicate physical condition group은 **{runs['duplicate_physical_condition_group_count']:,}개**이며, "
            f"새 main split을 넘는 중복은 **{runs['main_split_duplicate_condition_leak_count']:,}개**다.",
            f"동일 condition 반복 실행 사이 tip label 불일치는 "
            f"**{runs['duplicate_condition_tip_disagreement_count']:,}개 group**이다.",
            "",
            "## Probing Readiness",
            "",
            f"- Pair manifest: `{stats['artifacts']['pair_manifest']}`",
            f"- Run manifest: `{stats['artifacts']['run_manifest']}`",
            f"- Canonical deduplicated run manifest: `{stats['artifacts']['canonical_run_manifest']}` "
            f"({runs['unique_physical_condition_count']:,} conditions)",
            "- Primary visual input: `main_fixed_northwest`",
            "- Primary observation window: `0.0-2.375 s` (57 frames at 24 FPS)",
            "- Primary future target window: `2.375-5.0 s`",
            "- Primary pairwise subset: `gold`, with `silver/bronze` used for margin regression/control",
            "- `reject` pairs are retained in raw storage but excluded from primary probing",
            "",
            "## Current Interpretation",
            "",
            "이 snapshot은 접근 가능한 shard만 반영한다. 나머지 shard가 합쳐지면 같은 명령으로 문서를 다시 생성하고, "
            "full-grid 통계를 논문 수치로 고정해야 한다. 현재 수치는 pipeline과 분포 이상을 점검하는 preliminary statistics다.",
            "",
        ]
    )
    return "\n".join(lines)


def write_statistics_document(config: DatasetConfig, stats: dict[str, Any]) -> Path:
    write_text_atomic(config.statistics_document, render_statistics_markdown(stats))
    return config.statistics_document
