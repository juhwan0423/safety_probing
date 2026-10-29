from __future__ import annotations

import argparse
import bisect
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .adapters.vjepa import uniform_frame_indices
from .io_utils import load_json, write_json, write_torch_atomic
from .vjepa_features import select_canonical_rows


STATE_NAMES = (
    "roll_deg",
    "pitch_deg",
    "imu_ang_vel_x",
    "imu_ang_vel_y",
    "imu_ang_vel_z",
    "linear_vel_body_x",
    "linear_vel_body_y",
    "linear_vel_body_z",
    "imu_lin_acc_x",
    "imu_lin_acc_y",
    "imu_lin_acc_z",
    "system_com_body_x",
    "system_com_body_y",
    "system_com_body_z",
    "zmp_body_x",
    "zmp_body_y",
    "zmp_outward_margin_m",
    "com_outward_margin_m",
)

CONTACT_NAMES = (
    "back_wheel",
    "roller_back_left",
    "roller_back_right",
    "roller_front_left",
    "roller_front_right",
)

STATIC_NAMES = ("load_mass_kg", "load_com_lateral_m")

SUMMARY_NAMES = (
    "min_zmp_outward_margin_m",
    "min_com_outward_margin_m",
    "max_abs_roll_deg",
    "min_contact_count",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def world_vector_to_yaw_local(
    x: float, y: float, yaw_deg: float
) -> tuple[float, float]:
    yaw = math.radians(yaw_deg)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return cosine * x + sine * y, -sine * x + cosine * y


def world_point_to_yaw_local(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    yaw_deg: float,
) -> tuple[float, float]:
    return world_vector_to_yaw_local(
        x - origin_x, y - origin_y, yaw_deg
    )


def tubelet_center_times(
    frame_indices: list[int], source_fps: float
) -> list[float]:
    if len(frame_indices) % 2:
        raise ValueError("Frame indices must contain complete two-frame tubelets")
    frame_times = [(index + 1) / source_fps for index in frame_indices]
    return [
        (frame_times[index] + frame_times[index + 1]) / 2
        for index in range(0, len(frame_times), 2)
    ]


def _read_dense_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row["sim_time"]))
    return rows


def _nearest_rows(
    rows: list[dict[str, str]], target_times: list[float]
) -> tuple[list[dict[str, str]], float]:
    times = [float(row["sim_time"]) for row in rows]
    selected = []
    max_error = 0.0
    for target in target_times:
        insertion = bisect.bisect_left(times, target)
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < len(rows)
        ]
        index = min(candidates, key=lambda item: abs(times[item] - target))
        max_error = max(max_error, abs(times[index] - target))
        selected.append(rows[index])
    return selected, max_error


def _state_from_row(row: dict[str, str]) -> tuple[list[float], list[bool]]:
    forklift_x = _float(row.get("forklift_pos_x"))
    forklift_y = _float(row.get("forklift_pos_y"))
    forklift_z = _float(row.get("forklift_pos_z"))
    yaw_deg = _float(row.get("yaw_deg"))
    if None in (forklift_x, forklift_y, forklift_z, yaw_deg):
        raise RuntimeError("Dense row is missing the forklift reference pose")
    assert forklift_x is not None
    assert forklift_y is not None
    assert forklift_z is not None
    assert yaw_deg is not None

    values = [0.0] * len(STATE_NAMES)
    mask = [False] * len(STATE_NAMES)

    def put(name: str, value: float | None, valid: bool = True) -> None:
        index = STATE_NAMES.index(name)
        if value is not None and valid:
            values[index] = value
            mask[index] = True

    put("roll_deg", _float(row.get("roll_deg")))
    put("pitch_deg", _float(row.get("pitch_deg")))

    imu_valid = _bool(row.get("imu_valid"))
    for axis in "xyz":
        put(
            f"imu_ang_vel_{axis}",
            _float(row.get(f"imu_ang_vel_{axis}")),
            imu_valid,
        )
        put(
            f"imu_lin_acc_{axis}",
            _float(row.get(f"imu_lin_acc_{axis}")),
            imu_valid,
        )

    velocity_x = _float(row.get("linear_vel_x"))
    velocity_y = _float(row.get("linear_vel_y"))
    velocity_z = _float(row.get("linear_vel_z"))
    if velocity_x is not None and velocity_y is not None:
        local_x, local_y = world_vector_to_yaw_local(
            velocity_x, velocity_y, yaw_deg
        )
        put("linear_vel_body_x", local_x)
        put("linear_vel_body_y", local_y)
    put("linear_vel_body_z", velocity_z)

    com_valid = _bool(row.get("system_com_available"))
    com_x = _float(row.get("system_com_x"))
    com_y = _float(row.get("system_com_y"))
    com_z = _float(row.get("system_com_z"))
    if com_valid and com_x is not None and com_y is not None:
        local_x, local_y = world_point_to_yaw_local(
            com_x, com_y, forklift_x, forklift_y, yaw_deg
        )
        put("system_com_body_x", local_x)
        put("system_com_body_y", local_y)
    if com_valid and com_z is not None:
        put("system_com_body_z", com_z - forklift_z)

    zmp_valid = _bool(row.get("zmp_valid"))
    outward_valid = _bool(row.get("outward_lateral_margin_valid"))
    zmp_x = _float(row.get("zmp_x"))
    zmp_y = _float(row.get("zmp_y"))
    if zmp_valid and outward_valid and zmp_x is not None and zmp_y is not None:
        local_x, local_y = world_point_to_yaw_local(
            zmp_x, zmp_y, forklift_x, forklift_y, yaw_deg
        )
        put("zmp_body_x", local_x)
        put("zmp_body_y", local_y)

    put(
        "zmp_outward_margin_m",
        _float(row.get("zmp_outward_lateral_margin_m")),
        outward_valid,
    )
    put(
        "com_outward_margin_m",
        _float(row.get("com_outward_lateral_margin_m")),
        outward_valid,
    )
    return values, mask


def _contacts_from_row(
    row: dict[str, str]
) -> tuple[list[bool], list[list[float]], list[bool]]:
    forklift_x = _float(row.get("forklift_pos_x"))
    forklift_y = _float(row.get("forklift_pos_y"))
    yaw_deg = _float(row.get("yaw_deg"))
    if None in (forklift_x, forklift_y, yaw_deg):
        raise RuntimeError("Dense row is missing the forklift reference pose")
    assert forklift_x is not None
    assert forklift_y is not None
    assert yaw_deg is not None
    active = []
    positions = []
    position_mask = []
    for contact in CONTACT_NAMES:
        is_active = _bool(row.get(f"{contact}_contact"))
        x = _float(row.get(f"{contact}_contact_pos_x"))
        y = _float(row.get(f"{contact}_contact_pos_y"))
        valid = is_active and x is not None and y is not None
        if valid:
            assert x is not None and y is not None
            local_x, local_y = world_point_to_yaw_local(
                x, y, forklift_x, forklift_y, yaw_deg
            )
            positions.append([local_x, local_y])
        else:
            positions.append([0.0, 0.0])
        active.append(is_active)
        position_mask.append(valid)
    return active, positions, position_mask


def _future_summary(
    rows: list[dict[str, str]], observation_end_s: float
) -> tuple[list[float], list[bool]]:
    future = [
        row for row in rows if float(row["sim_time"]) > observation_end_s
    ]
    if not future:
        return [0.0] * len(SUMMARY_NAMES), [False] * len(SUMMARY_NAMES)

    zmp_margins = [
        value
        for row in future
        if _bool(row.get("outward_lateral_margin_valid"))
        for value in [_float(row.get("zmp_outward_lateral_margin_m"))]
        if value is not None
    ]
    com_margins = [
        value
        for row in future
        if _bool(row.get("outward_lateral_margin_valid"))
        for value in [_float(row.get("com_outward_lateral_margin_m"))]
        if value is not None
    ]
    rolls = [
        abs(value)
        for row in future
        for value in [_float(row.get("roll_deg"))]
        if value is not None
    ]
    contacts = [
        value
        for row in future
        for value in [_float(row.get("wheel_contact_count"))]
        if value is not None
    ]
    candidates = [zmp_margins, com_margins, rolls, contacts]
    reducers = [min, min, max, min]
    values = []
    masks = []
    for items, reducer in zip(candidates, reducers, strict=True):
        if items:
            values.append(float(reducer(items)))
            masks.append(True)
        else:
            values.append(0.0)
            masks.append(False)
    return values, masks


def build_run_targets(
    record: dict[str, Any],
    *,
    observed_times: list[float],
    future_times: list[float],
    observation_end_s: float,
) -> dict[str, Any]:
    rows = _read_dense_rows(Path(record["dense_log_path"]))
    observed_rows, observed_error = _nearest_rows(rows, observed_times)
    future_rows, future_error = _nearest_rows(rows, future_times)

    observed_state = []
    observed_state_mask = []
    observed_contact = []
    observed_contact_xy = []
    observed_contact_xy_mask = []
    for row in observed_rows:
        state, state_mask = _state_from_row(row)
        contact, contact_xy, contact_xy_mask = _contacts_from_row(row)
        observed_state.append(state)
        observed_state_mask.append(state_mask)
        observed_contact.append(contact)
        observed_contact_xy.append(contact_xy)
        observed_contact_xy_mask.append(contact_xy_mask)

    future_state = []
    future_state_mask = []
    future_contact = []
    future_contact_xy = []
    future_contact_xy_mask = []
    for row in future_rows:
        state, state_mask = _state_from_row(row)
        contact, contact_xy, contact_xy_mask = _contacts_from_row(row)
        future_state.append(state)
        future_state_mask.append(state_mask)
        future_contact.append(contact)
        future_contact_xy.append(contact_xy)
        future_contact_xy_mask.append(contact_xy_mask)

    com_offset = record["load_com_offset_local_xyz_m"]
    summary, summary_mask = _future_summary(rows, observation_end_s)
    binary_label = record.get("binary_tip_label")
    return {
        "static": [float(record["load_mass_kg"]), float(com_offset[1])],
        "static_mask": [True, True],
        "observed_state": observed_state,
        "observed_state_mask": observed_state_mask,
        "future_state": future_state,
        "future_state_mask": future_state_mask,
        "observed_contact": observed_contact,
        "observed_contact_xy": observed_contact_xy,
        "observed_contact_xy_mask": observed_contact_xy_mask,
        "future_contact": future_contact,
        "future_contact_xy": future_contact_xy,
        "future_contact_xy_mask": future_contact_xy_mask,
        "future_summary": summary,
        "future_summary_mask": summary_mask,
        "future_tip": 0.0 if binary_label is None else float(binary_label),
        "future_tip_mask": binary_label is not None,
        "max_alignment_error_s": max(observed_error, future_error),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build chassis-local physical targets for Safety Query probes."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_safety_query.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/current/manifests/canonical_runs.jsonl"),
    )
    parser.add_argument("--shard", default="part_01")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")
    experiment_path = args.experiment.resolve()
    manifest_path = args.manifest.resolve()
    project_root = experiment_path.parent.parent.parent
    experiment = load_json(experiment_path)
    rows = select_canonical_rows(
        _read_jsonl(manifest_path), shard=args.shard
    )
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    if not rows:
        raise RuntimeError("No canonical rows selected for Safety Query targets")

    configured = experiment["targets"]
    if tuple(configured["static"]) != STATIC_NAMES:
        raise RuntimeError("Static target config and implementation differ")
    if tuple(configured["state"]) != STATE_NAMES:
        raise RuntimeError("State target config and implementation differ")
    if tuple(configured["contacts"]) != CONTACT_NAMES:
        raise RuntimeError("Contact target config and implementation differ")
    if tuple(configured["future_summary"]) != SUMMARY_NAMES:
        raise RuntimeError("Summary target config and implementation differ")

    input_config = experiment["input"]
    fps = float(input_config["source_fps"])
    observed_indices = uniform_frame_indices(
        0,
        int(input_config["source_frame_count"]) - 1,
        int(input_config["sampled_frame_count"]),
    )
    future_indices = uniform_frame_indices(
        int(configured["future_source_start_index"]),
        int(configured["future_source_end_index"]),
        int(configured["future_sampled_frame_count"]),
    )
    observed_times = tubelet_center_times(observed_indices, fps)
    future_times = tubelet_center_times(future_indices, fps)
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "artifacts"
            / "targets"
            / f"safety_query_{args.shard}.pt"
        ).resolve()
    )
    contract = {
        "schema_version": 1,
        "experiment_sha256": _sha256(experiment_path),
        "manifest_sha256": _sha256(manifest_path),
        "shard": args.shard,
        "run_count": len(rows),
        "condition_ids": [str(row["condition_id"]) for row in rows],
        "observed_frame_indices": observed_indices,
        "future_frame_indices": future_indices,
        "observed_times_s": observed_times,
        "future_times_s": future_times,
        "coordinate_frame": configured["coordinate_frame"],
    }
    display = {
        **contract,
        "condition_ids": {
            "count": len(rows),
            "first": rows[0]["condition_id"],
            "last": rows[-1]["condition_id"],
        },
        "output": str(output_path),
    }
    print(json.dumps(display, indent=2), flush=True)
    if args.dry_run:
        return

    if output_path.is_file() and not args.overwrite:
        import torch

        existing = torch.load(
            output_path, map_location="cpu", mmap=True, weights_only=True
        )
        if existing.get("contract") != contract:
            raise RuntimeError(
                "Existing target cache contract differs; use --overwrite"
            )
        print(f"[Safety Query targets] Reusing {output_path}")
        return

    import torch

    built = []
    for index, record in enumerate(rows, start=1):
        built.append(
            build_run_targets(
                record,
                observed_times=observed_times,
                future_times=future_times,
                observation_end_s=float(input_config["end_time_s"]),
            )
        )
        if index % 100 == 0 or index == len(rows):
            print(
                f"[Safety Query targets] {index}/{len(rows)}", flush=True
            )

    tensor_keys = (
        "static",
        "static_mask",
        "observed_state",
        "observed_state_mask",
        "future_state",
        "future_state_mask",
        "observed_contact",
        "observed_contact_xy",
        "observed_contact_xy_mask",
        "future_contact",
        "future_contact_xy",
        "future_contact_xy_mask",
        "future_summary",
        "future_summary_mask",
        "future_tip",
        "future_tip_mask",
    )
    boolean_keys = {
        key for key in tensor_keys if key.endswith("mask") or key.endswith("contact")
    }
    tensors = {
        key: torch.tensor(
            [item[key] for item in built],
            dtype=torch.bool if key in boolean_keys else torch.float32,
        )
        for key in tensor_keys
    }
    records = [
        {
            "condition_id": row["condition_id"],
            "run_id": row["run_id"],
            "pair_id": row["pair_id"],
            "run_label": row["run_label"],
            "split": row["splits"]["main"],
            "safety_outcome": row["safety_outcome"],
            "turn_direction": row["turn_direction"],
            "intervention_type": row["intervention_type"],
            "quality_tier": row["quality_tier"],
        }
        for row in rows
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "channels": {
            "static": list(STATIC_NAMES),
            "state": list(STATE_NAMES),
            "contacts": list(CONTACT_NAMES),
            "contact_xy": ["body_x", "body_y"],
            "future_summary": list(SUMMARY_NAMES),
        },
        "records": records,
        "max_alignment_error_s": max(
            item["max_alignment_error_s"] for item in built
        ),
        **tensors,
    }
    write_torch_atomic(output_path, payload)
    sidecar = {
        key: value
        for key, value in payload.items()
        if key not in tensors
    }
    sidecar["tensor_shapes"] = {
        key: list(value.shape) for key, value in tensors.items()
    }
    sidecar["valid_counts"] = {
        key: int(value.sum())
        for key, value in tensors.items()
        if value.dtype == torch.bool
    }
    write_json(output_path.with_suffix(".json"), sidecar)
    print(f"[Safety Query targets] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
