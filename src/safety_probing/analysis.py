from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

from .config import DatasetConfig, LabelingConfig
from .io_utils import load_json, write_json, write_jsonl
from .labeling import (
    CONFIRMED_RECOVERED,
    OBSERVED_TIP,
    RIGHT_CENSORED,
    derive_safety_label,
    modernized_quality,
    safety_transition,
)


PROTOCOL_NAMES = ("main", "left_to_right", "right_to_left")
RUN_LABELS = ("run_A", "run_B")


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _percent(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 4)


def _quantile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def describe(values: Iterable[float | int | None]) -> dict[str, Any]:
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return {"count": 0}
    return {
        "count": len(finite),
        "min": finite[0],
        "q25": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.50),
        "q75": _quantile(finite, 0.75),
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
    }


def _binary_auc(stable_values: list[float], tip_values: list[float]) -> float | None:
    if not stable_values or not tip_values:
        return None
    samples = [(-value, 0) for value in stable_values] + [
        (-value, 1) for value in tip_values
    ]
    samples.sort(key=lambda item: item[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(samples):
        end = index + 1
        while end < len(samples) and samples[end][0] == samples[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum_positive += average_rank * sum(label for _, label in samples[index:end])
        index = end
    positive_count = len(tip_values)
    negative_count = len(stable_values)
    return (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _target_diagnostic(stable_values: list[float], tip_values: list[float]) -> dict[str, Any]:
    return {
        "stable": {
            **describe(stable_values),
            "negative_fraction": (
                sum(value < 0.0 for value in stable_values) / len(stable_values)
                if stable_values
                else None
            ),
        },
        "tip": {
            **describe(tip_values),
            "negative_fraction": (
                sum(value < 0.0 for value in tip_values) / len(tip_values)
                if tip_values
                else None
            ),
        },
        "tip_auc_using_negative_margin_as_risk": _binary_auc(stable_values, tip_values),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(source_config: Path) -> dict[str, Any]:
    repository = source_config.parent.parent

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repository), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "repository": str(repository),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _split_maps(split_manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for protocol_name in PROTOCOL_NAMES:
        protocol = split_manifest[protocol_name]
        mapping: dict[str, str] = {}
        for split_name, key in (
            ("train", "train_pair_ids"),
            ("val", "val_pair_ids"),
            ("test", "test_pair_ids"),
        ):
            for pair_id in protocol[key]:
                if pair_id in mapping:
                    raise ValueError(
                        f"Pair {pair_id} is assigned twice in protocol {protocol_name}"
                    )
                mapping[pair_id] = split_name
        result[protocol_name] = mapping
    return result


def _condition_id(run_metadata: dict[str, Any]) -> str:
    load_state = run_metadata.get("load_state", {})
    payload = {
        "route_id": run_metadata.get("route_id"),
        "seed": run_metadata.get("seed"),
        "mass_kg": load_state.get("load_mass_kg"),
        "com_xyz_m": load_state.get("load_com_offset_local_xyz_m"),
        "load_spawn_lift_m": load_state.get("load_spawn_lift_m"),
        "load_size_m": load_state.get("load_size_m"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _rgb_frame_counts(rgb_dir: Path) -> dict[str, int]:
    if not rgb_dir.is_dir():
        return {}
    camera_dirs = sorted(path for path in rgb_dir.iterdir() if path.is_dir())
    if camera_dirs:
        return {
            camera_dir.name: sum(1 for path in camera_dir.iterdir() if path.suffix == ".png")
            for camera_dir in camera_dirs
        }
    return {"default": sum(1 for path in rgb_dir.iterdir() if path.suffix == ".png")}


def _audit_rgb_pixels(
    run_rows: list[dict[str, Any]],
    *,
    camera_names: tuple[str, ...],
    expected_resolution: tuple[int, int],
    sample_run_count: int,
) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return {
            "available": False,
            "reason": "Pillow is not installed; install safety-probing[vision-audit].",
        }

    ordered = sorted(
        run_rows,
        key=lambda row: hashlib.sha256(str(row["run_id"]).encode("utf-8")).hexdigest(),
    )
    selected = ordered[: min(sample_run_count, len(ordered))]
    audited_images = 0
    near_blank_images: list[str] = []
    wrong_resolution_images: list[str] = []
    duplicate_time_samples: list[str] = []
    decode_failures: list[str] = []
    grayscale_stddevs: list[float] = []

    for run in selected:
        for camera_name in camera_names:
            camera_dir_value = run.get("rgb_dirs", {}).get(camera_name)
            if not camera_dir_value:
                continue
            frame_paths = sorted(Path(camera_dir_value).glob("*.png"))
            if not frame_paths:
                continue
            indices = sorted({0, len(frame_paths) // 2, len(frame_paths) - 1})
            content_hashes: list[str] = []
            for index in indices:
                frame_path = frame_paths[index]
                try:
                    with Image.open(frame_path) as image:
                        image.load()
                        audited_images += 1
                        if image.size != expected_resolution:
                            wrong_resolution_images.append(str(frame_path))
                        grayscale = image.convert("L")
                        statistics = ImageStat.Stat(grayscale)
                        stddev = float(statistics.stddev[0])
                        grayscale_stddevs.append(stddev)
                        minimum, maximum = grayscale.getextrema()
                        if stddev < 1.0 or maximum - minimum < 3:
                            near_blank_images.append(str(frame_path))
                    content_hashes.append(_sha256_file(frame_path))
                except Exception as exc:  # Pillow raises several decoder-specific exceptions.
                    decode_failures.append(f"{frame_path}: {type(exc).__name__}: {exc}")
            if len(content_hashes) > 1 and len(set(content_hashes)) != len(content_hashes):
                duplicate_time_samples.append(f"{run['run_id']}:{camera_name}")

    return {
        "available": True,
        "sampled_run_count": len(selected),
        "audited_image_count": audited_images,
        "sampling": "first/middle/last frame for deterministic hash-selected runs",
        "expected_resolution": list(expected_resolution),
        "near_blank_image_count": len(near_blank_images),
        "near_blank_examples": near_blank_images[:20],
        "wrong_resolution_image_count": len(wrong_resolution_images),
        "wrong_resolution_examples": wrong_resolution_images[:20],
        "duplicate_time_sample_count": len(duplicate_time_samples),
        "duplicate_time_sample_examples": duplicate_time_samples[:20],
        "decode_failure_count": len(decode_failures),
        "decode_failure_examples": decode_failures[:20],
        "grayscale_stddev": describe(grayscale_stddevs),
    }


def _linear_slope(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    mean_time = sum(time for time, _ in samples) / len(samples)
    mean_value = sum(value for _, value in samples) / len(samples)
    denominator = sum((time - mean_time) ** 2 for time, _ in samples)
    if denominator <= 0.0:
        return None
    return sum(
        (time - mean_time) * (value - mean_value) for time, value in samples
    ) / denominator


def analyze_dense_log(
    path: Path, labeling: LabelingConfig | None = None
) -> dict[str, Any]:
    labeling = labeling or LabelingConfig.from_dict({})
    required_columns = {
        "episode_step",
        "sim_time",
        "roll_deg",
        "pitch_deg",
        "wheel_contact_count",
        "imu_valid",
        "system_com_available",
        "zmp_valid",
        "zmp_outward_lateral_margin_m",
        "com_outward_lateral_margin_m",
        "outward_lateral_margin_valid",
    }
    row_count = 0
    first_time: float | None = None
    last_time: float | None = None
    true_counts: Counter[str] = Counter()
    parsed_counts: Counter[str] = Counter()
    min_outward_zmp: float | None = None
    min_outward_com: float | None = None
    min_support_margin: float | None = None
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    active_contact_rows = 0
    active_contact_rows_missing_xy = 0
    support_polygon_valid_rows = 0
    future_outward_zmp: list[float] = []
    roll_samples: list[tuple[float, float]] = []
    posthoc_roll_tip_time: float | None = None
    posthoc_support_loss_time: float | None = None
    low_support_streak = 0
    longest_support_loss_streak = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        contact_prefixes = sorted(
            name[: -len("_contact")]
            for name in fieldnames
            if name.endswith("_contact")
            and f"{name[: -len('_contact')]}_contact_pos_x" in fieldnames
        )

        for row in reader:
            row_count += 1
            sim_time = _float(row.get("sim_time"))
            if sim_time is not None:
                first_time = sim_time if first_time is None else first_time
                last_time = sim_time

            for column in (
                "imu_valid",
                "system_com_available",
                "zmp_valid",
                "outward_lateral_margin_valid",
            ):
                value = _bool(row.get(column))
                if value is not None:
                    parsed_counts[column] += 1
                if value is True:
                    true_counts[column] += 1

            vertex_count = _float(row.get("support_polygon_vertex_count"))
            if vertex_count is not None and vertex_count >= 3:
                support_polygon_valid_rows += 1

            roll = _float(row.get("roll_deg"))
            pitch = _float(row.get("pitch_deg"))
            if roll is not None:
                max_abs_roll = max(max_abs_roll, abs(roll))
                if sim_time is not None:
                    roll_samples.append((sim_time, abs(roll)))
                    if (
                        posthoc_roll_tip_time is None
                        and abs(roll) >= labeling.roll_tip_threshold_deg
                    ):
                        posthoc_roll_tip_time = sim_time
            if pitch is not None:
                max_abs_pitch = max(max_abs_pitch, abs(pitch))

            contact_count = _float(row.get("wheel_contact_count"))
            if (
                contact_count is not None
                and contact_count < labeling.support_loss_contact_count_lt
            ):
                low_support_streak += 1
                longest_support_loss_streak = max(
                    longest_support_loss_streak, low_support_streak
                )
                if (
                    posthoc_support_loss_time is None
                    and low_support_streak >= labeling.support_loss_grace_steps
                ):
                    posthoc_support_loss_time = sim_time
            else:
                low_support_streak = 0

            support_margin = _float(row.get("zmp_support_margin_m"))
            if support_margin is not None:
                min_support_margin = (
                    support_margin
                    if min_support_margin is None
                    else min(min_support_margin, support_margin)
                )

            if _bool(row.get("outward_lateral_margin_valid")) is True:
                zmp_margin = _float(row.get("zmp_outward_lateral_margin_m"))
                com_margin = _float(row.get("com_outward_lateral_margin_m"))
                if zmp_margin is not None:
                    min_outward_zmp = (
                        zmp_margin
                        if min_outward_zmp is None
                        else min(min_outward_zmp, zmp_margin)
                    )
                    if (
                        sim_time is not None
                        and sim_time >= labeling.observation_end_s
                    ):
                        future_outward_zmp.append(zmp_margin)
                if com_margin is not None:
                    min_outward_com = (
                        com_margin
                        if min_outward_com is None
                        else min(min_outward_com, com_margin)
                    )

            for prefix in contact_prefixes:
                if _bool(row.get(f"{prefix}_contact")) is not True:
                    continue
                active_contact_rows += 1
                x = _float(row.get(f"{prefix}_contact_pos_x"))
                y = _float(row.get(f"{prefix}_contact_pos_y"))
                if x is None or y is None:
                    active_contact_rows_missing_xy += 1

    terminal_roll_samples: list[tuple[float, float]] = []
    if last_time is not None:
        window_start = last_time - labeling.terminal_window_s
        terminal_roll_samples = [
            sample for sample in roll_samples if sample[0] >= window_start
        ]
    terminal_abs_roll = roll_samples[-1][1] if roll_samples else None
    terminal_abs_roll_rate = _linear_slope(terminal_roll_samples)

    validity = {
        column: {
            "true_rows": true_counts[column],
            "parsed_rows": parsed_counts[column],
            "fraction_of_all_rows": (
                true_counts[column] / row_count if row_count else None
            ),
        }
        for column in (
            "imu_valid",
            "system_com_available",
            "zmp_valid",
            "outward_lateral_margin_valid",
        )
    }
    return {
        "path": str(path),
        "row_count": row_count,
        "column_count": len(fieldnames),
        "missing_required_columns": missing_columns,
        "first_sim_time_s": first_time,
        "last_sim_time_s": last_time,
        "validity": validity,
        "support_polygon_valid_rows": support_polygon_valid_rows,
        "active_contact_rows": active_contact_rows,
        "active_contact_rows_missing_xy": active_contact_rows_missing_xy,
        "max_abs_roll_deg": max_abs_roll,
        "max_abs_pitch_deg": max_abs_pitch,
        "posthoc_roll_tip_time_s": posthoc_roll_tip_time,
        "posthoc_support_loss_time_s": posthoc_support_loss_time,
        "longest_support_loss_streak_steps": longest_support_loss_streak,
        "terminal_support_loss_streak_steps": low_support_streak,
        "terminal_abs_roll_deg": terminal_abs_roll,
        "terminal_abs_roll_rate_deg_s": terminal_abs_roll_rate,
        "terminal_window_s": labeling.terminal_window_s,
        "min_zmp_support_margin_m": min_support_margin,
        "min_outward_zmp_margin_m": min_outward_zmp,
        "min_outward_com_margin_m": min_outward_com,
        "future_target_start_s": labeling.observation_end_s,
        "future_min_outward_zmp_margin_m": (
            min(future_outward_zmp) if future_outward_zmp else None
        ),
    }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def analyze_dataset(config: DatasetConfig, *, scan_dense: bool = True) -> dict[str, Any]:
    if not config.generated_split_manifest.is_file():
        raise FileNotFoundError(
            f"Generate leakage-safe splits first: {config.generated_split_manifest}"
        )
    split_manifest = load_json(config.generated_split_manifest)
    split_maps = _split_maps(split_manifest)

    expected_rows = int(config.expected["physics_rows_per_run"])
    expected_rgb_frames = int(config.expected["rgb_frames_per_camera_per_run"])
    expected_cameras = tuple(str(name) for name in config.expected["camera_names"])

    root_rows: list[dict[str, Any]] = []
    pair_manifest_rows: list[dict[str, Any]] = []
    run_manifest_rows: list[dict[str, Any]] = []
    seen_pair_ids: Counter[str] = Counter()
    incomplete_pair_ids: list[str] = []
    dense_row_mismatch_run_ids: list[str] = []
    rgb_frame_mismatch_run_ids: list[str] = []
    missing_dense_run_ids: list[str] = []
    missing_required_columns_by_run: dict[str, list[str]] = {}

    source_transition_counts: Counter[str] = Counter()
    source_quality_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    intervention_counts: Counter[str] = Counter()
    direction_pair_counts: Counter[str] = Counter()
    speed_pair_counts: Counter[str] = Counter()
    steering_pair_counts: Counter[str] = Counter()
    seed_pair_counts: Counter[str] = Counter()
    source_tip_reason_counts: Counter[str] = Counter()
    event_reason_counts: Counter[str] = Counter()
    source_to_derived_counts: Counter[str] = Counter()
    direction_run_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    speed_run_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    steering_run_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    intervention_pair_breakdown: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {"transitions": Counter(), "quality_tiers": Counter()}
    )
    route_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pair_count": 0,
            "run_count": 0,
            "tip_run_count": 0,
            "outcomes": Counter(),
            "transitions": Counter(),
            "quality_tiers": Counter(),
        }
    )
    split_local_counts: dict[str, Counter[str]] = {
        name: Counter() for name in PROTOCOL_NAMES
    }
    split_quality_counts: dict[str, dict[str, Counter[str]]] = {
        name: defaultdict(Counter) for name in PROTOCOL_NAMES
    }
    split_run_outcomes: dict[str, dict[str, Counter[str]]] = {
        name: defaultdict(Counter) for name in PROTOCOL_NAMES
    }

    max_roll_values: list[float] = []
    max_pitch_values: list[float] = []
    failure_times: list[float] = []
    roll_delta_values: list[float] = []
    pitch_delta_values: list[float] = []
    dense_row_counts: list[int] = []
    per_run_min_outward_zmp: list[float] = []
    per_run_min_outward_com: list[float] = []
    per_run_min_support_margin: list[float] = []
    camera_frame_counts: dict[str, list[int]] = defaultdict(list)
    signal_true_rows: Counter[str] = Counter()
    signal_total_rows: Counter[str] = Counter()
    total_support_polygon_valid_rows = 0
    total_active_contact_rows = 0
    total_active_contact_rows_missing_xy = 0
    total_dense_rows = 0
    condition_to_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    target_values_by_outcome: dict[str, dict[str, list[float]]] = {
        metric: {"stable": [], "tip": []}
        for metric in (
            "min_outward_zmp_margin_m",
            "future_min_outward_zmp_margin_m",
            "min_outward_com_margin_m",
        )
    }

    for dataset_root in config.dataset_roots:
        pairs_dir = dataset_root / "pairs"
        if not pairs_dir.is_dir():
            root_rows.append(
                {
                    "dataset_root": str(dataset_root),
                    "available": False,
                    "planned_pair_count": None,
                    "pair_directory_count": 0,
                    "completed_pair_count": 0,
                    "completed_run_count": 0,
                }
            )
            continue

        snapshot_path = dataset_root / "configs" / "experiment_config.snapshot.json"
        snapshot = load_json(snapshot_path) if snapshot_path.is_file() else {}
        planned_pair_count = len(snapshot.get("pair_requests", [])) or None
        pair_dirs = sorted(path for path in pairs_dir.iterdir() if path.is_dir())
        root_completed_pairs = 0
        root_completed_runs = 0

        for pair_dir in pair_dirs:
            pair_metadata_path = pair_dir / "metadata_pair.json"
            if not pair_metadata_path.is_file():
                incomplete_pair_ids.append(pair_dir.name)
                continue
            pair = load_json(pair_metadata_path)
            pair_id = str(pair["pair_id"])
            seen_pair_ids[pair_id] += 1
            root_completed_pairs += 1
            transition = str(
                pair.get("summary_outcome_comparison", {}).get(
                    "tip_state_transition", "unknown"
                )
            )
            quality_tier = str(pair.get("quality_gate", {}).get("tier", "unknown"))
            intervention = str(pair.get("intervention_type", "unknown"))
            direction = str(pair.get("turn_direction", "unknown"))
            speed = str(pair.get("speed_label", "unknown"))
            steering = str(pair.get("steering_label", "unknown"))
            seed = str(pair.get("seed", "unknown"))
            route_id = str(pair.get("route_id", "unknown"))

            source_transition_counts[transition] += 1
            source_quality_counts[quality_tier] += 1
            intervention_counts[intervention] += 1
            intervention_pair_breakdown[intervention]["transitions"][transition] += 1
            intervention_pair_breakdown[intervention]["quality_tiers"][quality_tier] += 1
            direction_pair_counts[direction] += 1
            speed_pair_counts[speed] += 1
            steering_pair_counts[steering] += 1
            seed_pair_counts[seed] += 1
            route_rows[route_id]["pair_count"] += 1
            route_rows[route_id]["transitions"][transition] += 1
            route_rows[route_id]["quality_tiers"][quality_tier] += 1

            comparison = pair.get("summary_outcome_comparison", {})
            roll_delta = _float(comparison.get("max_roll_deg_delta"))
            pitch_delta = _float(comparison.get("max_pitch_deg_delta"))
            if roll_delta is not None:
                roll_delta_values.append(abs(roll_delta))
            if pitch_delta is not None:
                pitch_delta_values.append(abs(pitch_delta))

            split_assignments = {
                name: split_maps[name].get(pair_id) for name in PROTOCOL_NAMES
            }
            for protocol_name, split_name in split_assignments.items():
                split_local_counts[protocol_name][split_name or "unassigned"] += 1

            pair_manifest_rows.append(
                {
                    "pair_id": pair_id,
                    "dataset_root": str(dataset_root),
                    "pair_dir": str(pair_dir),
                    "metadata_path": str(pair_metadata_path),
                    "route_id": route_id,
                    "turn_direction": direction,
                    "speed_label": speed,
                    "steering_label": steering,
                    "seed": pair.get("seed"),
                    "intervention_type": intervention,
                    "intervention_value_A": pair.get("intervention_value_A"),
                    "intervention_value_B": pair.get("intervention_value_B"),
                    "same_action_trajectory": bool(
                        pair.get("same_action_trajectory", True)
                    ),
                    "source_tip_state_transition": transition,
                    "source_quality_tier": quality_tier,
                    "source_quality_accepted": pair.get("quality_gate", {}).get(
                        "accepted"
                    ),
                    "max_roll_deg_delta": roll_delta,
                    "max_pitch_deg_delta": pitch_delta,
                    "splits": split_assignments,
                }
            )

            for run_dir_name in RUN_LABELS:
                run_dir = pair_dir / run_dir_name
                run_metadata_path = run_dir / "metadata_run.json"
                if not run_metadata_path.is_file():
                    continue
                run = load_json(run_metadata_path)
                root_completed_runs += 1
                run_id = str(run["run_id"])
                source_tip_over = bool(run.get("tip_over"))
                source_tip_reason_counts[
                    str(run.get("tip_over_reason") or "none")
                ] += 1

                max_roll = _float(run.get("max_roll_deg"))
                max_pitch = _float(run.get("max_pitch_deg"))
                if max_roll is not None:
                    max_roll_values.append(max_roll)
                if max_pitch is not None:
                    max_pitch_values.append(max_pitch)

                dense_log_relpath = Path(str(run.get("dense_log_path", f"{run_dir_name}/dense_log.csv")))
                dense_log_path = (
                    dense_log_relpath
                    if dense_log_relpath.is_absolute()
                    else pair_dir / dense_log_relpath
                )
                dense_summary: dict[str, Any] | None = None
                if not dense_log_path.is_file():
                    missing_dense_run_ids.append(run_id)
                elif scan_dense:
                    dense_summary = analyze_dense_log(dense_log_path, config.labeling)
                    row_count = int(dense_summary["row_count"])
                    dense_row_counts.append(row_count)
                    total_dense_rows += row_count
                    if row_count != expected_rows:
                        dense_row_mismatch_run_ids.append(run_id)
                    missing_columns = dense_summary["missing_required_columns"]
                    if missing_columns:
                        missing_required_columns_by_run[run_id] = missing_columns
                    for column, values in dense_summary["validity"].items():
                        signal_true_rows[column] += int(values["true_rows"])
                        signal_total_rows[column] += row_count
                    total_support_polygon_valid_rows += int(
                        dense_summary["support_polygon_valid_rows"]
                    )
                    total_active_contact_rows += int(dense_summary["active_contact_rows"])
                    total_active_contact_rows_missing_xy += int(
                        dense_summary["active_contact_rows_missing_xy"]
                    )
                    if dense_summary["min_outward_zmp_margin_m"] is not None:
                        per_run_min_outward_zmp.append(
                            float(dense_summary["min_outward_zmp_margin_m"])
                        )
                    if dense_summary["min_outward_com_margin_m"] is not None:
                        per_run_min_outward_com.append(
                            float(dense_summary["min_outward_com_margin_m"])
                        )
                    if dense_summary["min_zmp_support_margin_m"] is not None:
                        per_run_min_support_margin.append(
                            float(dense_summary["min_zmp_support_margin_m"])
                        )
                if dense_summary is not None:
                    safety_label = derive_safety_label(
                        run, dense_summary, config.labeling
                    )
                else:
                    safety_label = {
                        "label_schema_version": config.labeling.schema_version,
                        "safety_outcome": (
                            OBSERVED_TIP if source_tip_over else RIGHT_CENSORED
                        ),
                        "event_observed": source_tip_over,
                        "binary_tip_label": 1 if source_tip_over else None,
                        "event_time_s": _float(run.get("failure_time")),
                        "event_reason": (
                            str(run.get("tip_over_reason") or "source_tip_over")
                            if source_tip_over
                            else "dense_label_evidence_unavailable"
                        ),
                        "event_source": "source_metadata",
                        "censor_time_s": None,
                        "followup_time_s": _float(run.get("failure_time")),
                        "source_tip_over": source_tip_over,
                        "source_failure_time_s": _float(run.get("failure_time")),
                        "source_tip_over_reason": run.get("tip_over_reason"),
                        "upgraded_from_source": False,
                        "future_prediction_eligible": False,
                        "future_tip_label": None,
                        "observation_end_s": config.labeling.observation_end_s,
                        "labeling_complete": False,
                    }

                outcome = str(safety_label["safety_outcome"])
                event_time = _float(safety_label.get("event_time_s"))
                if event_time is not None:
                    failure_times.append(event_time)
                event_reason_counts[str(safety_label["event_reason"])] += 1
                source_to_derived_counts[
                    f"source_{'tip' if source_tip_over else 'stable'}_to_{outcome}"
                ] += 1
                endpoint = "A" if run_dir_name == "run_A" else "B"
                labels_by_pair[pair_id][endpoint] = safety_label

                for protocol_name, split_name in split_assignments.items():
                    split_run_outcomes[protocol_name][split_name or "unassigned"][
                        outcome
                    ] += 1
                direction_run_outcomes[direction][outcome] += 1
                speed_run_outcomes[speed][outcome] += 1
                steering_run_outcomes[steering][outcome] += 1
                route_rows[route_id]["run_count"] += 1
                route_rows[route_id]["outcomes"][outcome] += 1
                if outcome == OBSERVED_TIP:
                    route_rows[route_id]["tip_run_count"] += 1

                if dense_summary is not None and outcome != RIGHT_CENSORED:
                    outcome_key = "tip" if outcome == OBSERVED_TIP else "stable"
                    for metric in target_values_by_outcome:
                        value = dense_summary.get(metric)
                        if value is not None:
                            target_values_by_outcome[metric][outcome_key].append(
                                float(value)
                            )

                rgb_relpath = Path(str(run.get("rgb_dir") or f"{run_dir_name}/rgb"))
                rgb_dir = rgb_relpath if rgb_relpath.is_absolute() else pair_dir / rgb_relpath
                rgb_counts = _rgb_frame_counts(rgb_dir)
                for camera_name in expected_cameras:
                    count = rgb_counts.get(camera_name, 0)
                    camera_frame_counts[camera_name].append(count)
                    if count != expected_rgb_frames:
                        rgb_frame_mismatch_run_ids.append(f"{run_id}:{camera_name}")

                condition_id = _condition_id(run)
                condition_to_runs[condition_id].append(
                    {
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "main_split": split_assignments["main"] or "",
                        "safety_outcome": outcome,
                        "binary_tip_label": safety_label["binary_tip_label"],
                    }
                )
                run_manifest_rows.append(
                    {
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "run_label": run.get("run_label"),
                        "condition_id": condition_id,
                        "dataset_root": str(dataset_root),
                        "run_dir": str(run_dir),
                        "metadata_path": str(run_metadata_path),
                        "dense_log_path": str(dense_log_path),
                        "rgb_dirs": {
                            camera_name: str(rgb_dir / camera_name)
                            for camera_name in rgb_counts
                        },
                        "route_id": route_id,
                        "turn_direction": direction,
                        "speed_label": speed,
                        "steering_label": steering,
                        "seed": run.get("seed"),
                        "intervention_type": intervention,
                        "intervention_value": run.get("intervention_value"),
                        "load_mass_kg": run.get("load_state", {}).get("load_mass_kg"),
                        "load_com_offset_local_xyz_m": run.get("load_state", {}).get(
                            "load_com_offset_local_xyz_m"
                        ),
                        "label_schema_version": safety_label[
                            "label_schema_version"
                        ],
                        "safety_outcome": outcome,
                        "event_observed": safety_label["event_observed"],
                        "binary_tip_label": safety_label["binary_tip_label"],
                        "tip_over": safety_label["binary_tip_label"],
                        "failure_time_s": safety_label["event_time_s"],
                        "event_reason": safety_label["event_reason"],
                        "event_source": safety_label["event_source"],
                        "censor_time_s": safety_label["censor_time_s"],
                        "followup_time_s": safety_label["followup_time_s"],
                        "future_prediction_eligible": safety_label[
                            "future_prediction_eligible"
                        ],
                        "future_tip_label": safety_label["future_tip_label"],
                        "source_tip_over": source_tip_over,
                        "source_failure_time_s": safety_label[
                            "source_failure_time_s"
                        ],
                        "source_tip_over_reason": safety_label[
                            "source_tip_over_reason"
                        ],
                        "upgraded_from_source": safety_label[
                            "upgraded_from_source"
                        ],
                        "label_evidence": safety_label,
                        "max_roll_deg": run.get("max_roll_deg"),
                        "max_pitch_deg": run.get("max_pitch_deg"),
                        "source_quality_tier": quality_tier,
                        "source_quality_accepted": pair.get("quality_gate", {}).get(
                            "accepted"
                        ),
                        "splits": split_assignments,
                        "dense_summary": dense_summary,
                    }
                )

        root_rows.append(
            {
                "dataset_root": str(dataset_root),
                "available": True,
                "planned_pair_count": planned_pair_count,
                "pair_directory_count": len(pair_dirs),
                "completed_pair_count": root_completed_pairs,
                "completed_run_count": root_completed_runs,
                "config_snapshot": str(snapshot_path) if snapshot_path.is_file() else None,
            }
        )

    source_intervention_breakdown = {
        intervention: {
            "tip_state_transition_counts": _counter_dict(values["transitions"]),
            "quality_tier_counts": _counter_dict(values["quality_tiers"]),
        }
        for intervention, values in sorted(intervention_pair_breakdown.items())
    }
    intervention_pair_breakdown = defaultdict(
        lambda: {"transitions": Counter(), "quality_tiers": Counter()}
    )
    split_quality_counts = {
        name: defaultdict(Counter) for name in PROTOCOL_NAMES
    }
    for values in route_rows.values():
        values["transitions"] = Counter()
        values["quality_tiers"] = Counter()

    pair_quality_by_id: dict[str, dict[str, Any]] = {}
    for pair_row in pair_manifest_rows:
        pair_id = str(pair_row["pair_id"])
        endpoint_labels = labels_by_pair.get(pair_id, {})
        label_a = endpoint_labels.get("A")
        label_b = endpoint_labels.get("B")
        if label_a is None or label_b is None:
            outcome_a = RIGHT_CENSORED
            outcome_b = RIGHT_CENSORED
        else:
            outcome_a = str(label_a["safety_outcome"])
            outcome_b = str(label_b["safety_outcome"])

        transition = safety_transition(outcome_a, outcome_b)
        quality = modernized_quality(
            outcome_a=outcome_a,
            outcome_b=outcome_b,
            same_action=bool(pair_row["same_action_trajectory"]),
            max_roll_deg_delta=_float(pair_row.get("max_roll_deg_delta")),
            max_pitch_deg_delta=_float(pair_row.get("max_pitch_deg_delta")),
        )
        pair_quality_by_id[pair_id] = quality
        pair_row.update(
            {
                "label_schema_version": config.labeling.schema_version,
                "run_A_safety_outcome": outcome_a,
                "run_B_safety_outcome": outcome_b,
                "run_A_binary_tip_label": (
                    label_a.get("binary_tip_label") if label_a else None
                ),
                "run_B_binary_tip_label": (
                    label_b.get("binary_tip_label") if label_b else None
                ),
                "tip_state_transition": transition,
                "quality_tier": quality["tier"],
                "quality_accepted": quality["accepted"],
                "quality_reason": quality["reason"],
                "binary_pair_eligible": RIGHT_CENSORED
                not in {outcome_a, outcome_b},
            }
        )

        transition_counts[transition] += 1
        quality_counts[str(quality["tier"])] += 1
        intervention = str(pair_row["intervention_type"])
        intervention_pair_breakdown[intervention]["transitions"][transition] += 1
        intervention_pair_breakdown[intervention]["quality_tiers"][
            str(quality["tier"])
        ] += 1
        route_id = str(pair_row["route_id"])
        route_rows[route_id]["transitions"][transition] += 1
        route_rows[route_id]["quality_tiers"][str(quality["tier"])] += 1
        for protocol_name, split_name in pair_row["splits"].items():
            split_quality_counts[protocol_name][split_name or "unassigned"][
                str(quality["tier"])
            ] += 1

    for run_row in run_manifest_rows:
        quality = pair_quality_by_id.get(str(run_row["pair_id"]), {})
        run_row["quality_tier"] = quality.get("tier", "censored")
        run_row["quality_accepted"] = quality.get("accepted", False)

    completed_pairs = len(pair_manifest_rows)
    completed_runs = len(run_manifest_rows)
    accepted_pairs = sum(1 for row in pair_manifest_rows if row["quality_accepted"] is True)
    tip_runs = sum(
        1 for row in run_manifest_rows if row["safety_outcome"] == OBSERVED_TIP
    )
    recovered_runs = sum(
        1
        for row in run_manifest_rows
        if row["safety_outcome"] == CONFIRMED_RECOVERED
    )
    censored_runs = sum(
        1 for row in run_manifest_rows if row["safety_outcome"] == RIGHT_CENSORED
    )
    duplicate_conditions = {
        condition_id: rows for condition_id, rows in condition_to_runs.items() if len(rows) > 1
    }
    actual_condition_split_leaks = {
        condition_id: rows
        for condition_id, rows in duplicate_conditions.items()
        if len({row["main_split"] for row in rows}) > 1
    }
    duplicate_condition_outcome_disagreements = {
        condition_id: rows
        for condition_id, rows in duplicate_conditions.items()
        if len({str(row["safety_outcome"]) for row in rows}) > 1
    }
    expected_resolution_values = config.expected.get("camera_resolution", [1280, 720])
    pixel_audit = _audit_rgb_pixels(
        run_manifest_rows,
        camera_names=expected_cameras,
        expected_resolution=(
            int(expected_resolution_values[0]),
            int(expected_resolution_values[1]),
        ),
        sample_run_count=int(config.expected.get("pixel_audit_run_sample_count", 64)),
    )

    route_summary = {}
    for route_id, values in sorted(route_rows.items()):
        run_count = int(values["run_count"])
        definitive_run_count = int(values["outcomes"][OBSERVED_TIP]) + int(
            values["outcomes"][CONFIRMED_RECOVERED]
        )
        route_summary[route_id] = {
            "pair_count": int(values["pair_count"]),
            "run_count": run_count,
            "tip_run_count": int(values["tip_run_count"]),
            "tip_rate_percent": _percent(
                values["tip_run_count"], definitive_run_count
            ),
            "safety_outcome_counts": _counter_dict(values["outcomes"]),
            "tip_state_transition_counts": _counter_dict(values["transitions"]),
            "quality_tier_counts": _counter_dict(values["quality_tiers"]),
        }

    signal_validity = {
        column: {
            "true_rows": signal_true_rows[column],
            "total_rows": signal_total_rows[column],
            "fraction": (
                signal_true_rows[column] / signal_total_rows[column]
                if signal_total_rows[column]
                else None
            ),
        }
        for column in sorted(signal_total_rows)
    }

    pair_manifest_path = config.output_dir / "manifests" / "pairs.jsonl"
    run_manifest_path = config.output_dir / "manifests" / "runs.jsonl"
    canonical_run_manifest_path = config.output_dir / "manifests" / "canonical_runs.jsonl"
    canonical_by_condition: dict[str, dict[str, Any]] = {}
    for row in run_manifest_rows:
        condition_id = str(row["condition_id"])
        previous = canonical_by_condition.get(condition_id)
        if previous is None or str(row["run_id"]) < str(previous["run_id"]):
            canonical_by_condition[condition_id] = row
    canonical_run_rows = [
        canonical_by_condition[condition_id] for condition_id in sorted(canonical_by_condition)
    ]
    canonical_outcome_counts = Counter(
        str(row["safety_outcome"]) for row in canonical_run_rows
    )
    source_tip_runs = sum(1 for row in run_manifest_rows if row["source_tip_over"])
    upgraded_runs = sum(1 for row in run_manifest_rows if row["upgraded_from_source"])
    binary_pair_count = sum(
        1 for row in pair_manifest_rows if row["binary_pair_eligible"]
    )
    write_jsonl(pair_manifest_path, pair_manifest_rows)
    write_jsonl(run_manifest_path, run_manifest_rows)
    write_jsonl(canonical_run_manifest_path, canonical_run_rows)

    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": config.dataset_name,
            "version": config.dataset_version,
            "expected_pair_count": int(config.expected["pair_count"]),
            "expected_run_count": int(config.expected["run_count"]),
            "expected_physics_rows_per_run": expected_rows,
            "expected_rgb_frames_per_camera_per_run": expected_rgb_frames,
            "expected_camera_names": list(expected_cameras),
            "labeling": {
                "schema_version": config.labeling.schema_version,
                "observation_end_s": config.labeling.observation_end_s,
                "roll_tip_threshold_deg": config.labeling.roll_tip_threshold_deg,
                "support_loss_contact_count_lt": config.labeling.support_loss_contact_count_lt,
                "support_loss_grace_steps": config.labeling.support_loss_grace_steps,
                "terminal_window_s": config.labeling.terminal_window_s,
                "right_censor_min_abs_roll_deg": config.labeling.right_censor_min_abs_roll_deg,
                "right_censor_min_abs_roll_rate_deg_s": config.labeling.right_censor_min_abs_roll_rate_deg_s,
            },
        },
        "provenance": {
            "analysis_config": str(config.config_path),
            "source_config": str(config.source_config),
            "source_config_sha256": _sha256_file(config.source_config),
            "split_manifest": str(config.generated_split_manifest),
            "split_manifest_sha256": _sha256_file(config.generated_split_manifest),
            "source_git": _git_state(config.source_config),
        },
        "completion": {
            "dataset_roots": root_rows,
            "available_root_count": sum(1 for row in root_rows if row["available"]),
            "configured_root_count": len(root_rows),
            "completed_pair_count": completed_pairs,
            "completed_pair_percent_of_full_grid": _percent(
                completed_pairs, int(config.expected["pair_count"])
            ),
            "completed_run_count": completed_runs,
            "completed_run_percent_of_full_grid": _percent(
                completed_runs, int(config.expected["run_count"])
            ),
            "incomplete_pair_directory_count": len(incomplete_pair_ids),
            "incomplete_pair_ids": sorted(incomplete_pair_ids)[:100],
            "duplicate_pair_id_count": sum(1 for count in seen_pair_ids.values() if count > 1),
        },
        "pairs": {
            "count": completed_pairs,
            "accepted_count": accepted_pairs,
            "accepted_percent": _percent(accepted_pairs, completed_pairs),
            "rejected_count": completed_pairs - accepted_pairs,
            "binary_pair_eligible_count": binary_pair_count,
            "right_censored_pair_count": completed_pairs - binary_pair_count,
            "tip_state_transition_counts": _counter_dict(transition_counts),
            "quality_tier_counts": _counter_dict(quality_counts),
            "source_tip_state_transition_counts": _counter_dict(
                source_transition_counts
            ),
            "source_quality_tier_counts": _counter_dict(source_quality_counts),
            "intervention_type_counts": _counter_dict(intervention_counts),
            "intervention_breakdown": {
                intervention: {
                    "tip_state_transition_counts": _counter_dict(values["transitions"]),
                    "quality_tier_counts": _counter_dict(values["quality_tiers"]),
                }
                for intervention, values in sorted(intervention_pair_breakdown.items())
            },
            "source_intervention_breakdown": source_intervention_breakdown,
            "turn_direction_counts": _counter_dict(direction_pair_counts),
            "speed_label_counts": _counter_dict(speed_pair_counts),
            "steering_label_counts": _counter_dict(steering_pair_counts),
            "seed_counts": _counter_dict(seed_pair_counts),
            "absolute_max_roll_delta_deg": describe(roll_delta_values),
            "absolute_max_pitch_delta_deg": describe(pitch_delta_values),
            "by_route": route_summary,
        },
        "runs": {
            "count": completed_runs,
            "tip_count": tip_runs,
            "stable_count": recovered_runs,
            "censored_count": censored_runs,
            "definitive_count": tip_runs + recovered_runs,
            "tip_rate_percent": _percent(tip_runs, tip_runs + recovered_runs),
            "safety_outcome_counts": {
                OBSERVED_TIP: tip_runs,
                CONFIRMED_RECOVERED: recovered_runs,
                RIGHT_CENSORED: censored_runs,
            },
            "canonical_safety_outcome_counts": _counter_dict(
                canonical_outcome_counts
            ),
            "source_tip_count": source_tip_runs,
            "posthoc_upgraded_tip_count": upgraded_runs,
            "event_reason_counts": _counter_dict(event_reason_counts),
            "source_tip_reason_counts": _counter_dict(source_tip_reason_counts),
            "source_to_derived_counts": _counter_dict(source_to_derived_counts),
            "direction_outcomes": {
                direction: _counter_dict(counter)
                for direction, counter in sorted(direction_run_outcomes.items())
            },
            "speed_outcomes": {
                speed: _counter_dict(counter)
                for speed, counter in sorted(speed_run_outcomes.items())
            },
            "steering_outcomes": {
                steering: _counter_dict(counter)
                for steering, counter in sorted(steering_run_outcomes.items())
            },
            "failure_time_s": describe(failure_times),
            "max_roll_deg": describe(max_roll_values),
            "max_pitch_deg": describe(max_pitch_values),
            "duplicate_physical_condition_group_count": len(duplicate_conditions),
            "unique_physical_condition_count": len(canonical_run_rows),
            "duplicate_condition_tip_disagreement_count": len(
                duplicate_condition_outcome_disagreements
            ),
            "duplicate_condition_tip_disagreement_examples": dict(
                list(sorted(duplicate_condition_outcome_disagreements.items()))[:20]
            ),
            "main_split_duplicate_condition_leak_count": len(actual_condition_split_leaks),
        },
        "dense_logs": {
            "scanned": scan_dense,
            "total_row_count": total_dense_rows,
            "rows_per_run": describe(dense_row_counts),
            "row_count_mismatch_run_count": len(dense_row_mismatch_run_ids),
            "row_count_mismatch_run_ids": dense_row_mismatch_run_ids[:100],
            "missing_dense_log_run_count": len(missing_dense_run_ids),
            "missing_dense_log_run_ids": missing_dense_run_ids[:100],
            "missing_required_columns_run_count": len(missing_required_columns_by_run),
            "missing_required_columns_examples": dict(
                list(sorted(missing_required_columns_by_run.items()))[:20]
            ),
            "signal_validity": signal_validity,
            "support_polygon_valid_row_fraction": (
                total_support_polygon_valid_rows / total_dense_rows
                if total_dense_rows
                else None
            ),
            "active_contact_row_count": total_active_contact_rows,
            "active_contact_missing_xy_count": total_active_contact_rows_missing_xy,
            "active_contact_xy_complete_fraction": (
                1.0 - total_active_contact_rows_missing_xy / total_active_contact_rows
                if total_active_contact_rows
                else None
            ),
            "per_run_min_outward_zmp_margin_m": describe(per_run_min_outward_zmp),
            "per_run_min_outward_com_margin_m": describe(per_run_min_outward_com),
            "per_run_min_zmp_support_margin_m": describe(per_run_min_support_margin),
            "target_diagnostics": {
                metric: _target_diagnostic(values["stable"], values["tip"])
                for metric, values in target_values_by_outcome.items()
            },
        },
        "rgb": {
            "camera_frame_counts": {
                camera_name: describe(values)
                for camera_name, values in sorted(camera_frame_counts.items())
            },
            "frame_count_mismatch_count": len(rgb_frame_mismatch_run_ids),
            "frame_count_mismatch_examples": rgb_frame_mismatch_run_ids[:100],
            "pixel_audit": pixel_audit,
        },
        "splits": {
            "local_pair_counts": {
                protocol_name: _counter_dict(counter)
                for protocol_name, counter in split_local_counts.items()
            },
            "local_quality_tier_counts": {
                protocol_name: {
                    split_name: _counter_dict(counter)
                    for split_name, counter in sorted(split_counts.items())
                }
                for protocol_name, split_counts in split_quality_counts.items()
            },
            "local_run_outcomes": {
                protocol_name: {
                    split_name: _counter_dict(counter)
                    for split_name, counter in sorted(split_counts.items())
                }
                for protocol_name, split_counts in split_run_outcomes.items()
            },
            "generated_audit": split_manifest.get("audit", {}).get(
                "generated_manifest", {}
            ),
            "reference_audit": split_manifest.get("audit", {}).get(
                "reference_manifest", {}
            ),
        },
        "artifacts": {
            "pair_manifest": str(pair_manifest_path),
            "run_manifest": str(run_manifest_path),
            "canonical_run_manifest": str(canonical_run_manifest_path),
        },
    }
    write_json(config.output_dir / "dataset_statistics.json", payload)
    return payload
