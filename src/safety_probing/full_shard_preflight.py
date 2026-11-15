from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .io_utils import load_json, write_json


REQUIRED_PAIR_FILES = ("metadata_pair.json", "assumptions.json")
REQUIRED_RUN_FILES = ("metadata_run.json", "dense_log.csv", "command_log.csv")


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def _safe_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return load_json(path), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"{path.name}: {type(error).__name__}: {error}"


def audit_pair_directory(
    pair_dir: Path,
    *,
    pair_id: str,
    expected_dense_rows: int,
    expected_rgb_frames: int,
    camera_names: tuple[str, ...],
    deep: bool,
) -> list[str]:
    issues: list[str] = []
    for relative in REQUIRED_PAIR_FILES:
        if not (pair_dir / relative).is_file():
            issues.append(f"missing {relative}")

    metadata_path = pair_dir / "metadata_pair.json"
    if metadata_path.is_file():
        metadata, error = _safe_json(metadata_path)
        if error is not None:
            issues.append(error)
        elif str(metadata.get("pair_id")) != pair_id:
            issues.append(
                f"metadata_pair pair_id={metadata.get('pair_id')!r}, expected {pair_id!r}"
            )

    for run_label in ("A", "B"):
        run_dir = pair_dir / f"run_{run_label}"
        if not run_dir.is_dir():
            issues.append(f"missing run_{run_label}/")
            continue
        for relative in REQUIRED_RUN_FILES:
            if not (run_dir / relative).is_file():
                issues.append(f"missing run_{run_label}/{relative}")
        if not deep:
            continue
        dense_path = run_dir / "dense_log.csv"
        if dense_path.is_file():
            data_rows = max(0, _line_count(dense_path) - 1)
            if data_rows != expected_dense_rows:
                issues.append(
                    f"run_{run_label}/dense_log.csv rows={data_rows}, "
                    f"expected {expected_dense_rows}"
                )
        for camera in camera_names:
            rgb_dir = run_dir / "rgb" / camera
            if not rgb_dir.is_dir():
                issues.append(f"missing run_{run_label}/rgb/{camera}/")
                continue
            frame_count = sum(1 for _ in rgb_dir.glob("frame_*.png"))
            if frame_count != expected_rgb_frames:
                issues.append(
                    f"run_{run_label}/rgb/{camera} frames={frame_count}, "
                    f"expected {expected_rgb_frames}"
                )
    return issues


def audit_shard(
    *,
    sim_root: Path,
    shard_spec: dict[str, Any],
    expected_dense_rows: int,
    expected_rgb_frames: int,
    camera_names: tuple[str, ...],
    deep: bool,
    max_examples: int,
) -> dict[str, Any]:
    config_path = (sim_root / str(shard_spec["config"])).resolve()
    root = (sim_root / str(shard_spec["dataset_root"])).resolve()
    expected_count = int(shard_spec["pair_count"])
    report: dict[str, Any] = {
        "shard_index": int(shard_spec["shard_index"]),
        "config": str(config_path),
        "dataset_root": str(root),
        "expected_pair_count": expected_count,
        "available": root.is_dir(),
    }
    if not config_path.is_file():
        return report | {
            "ready": False,
            "config_error": "missing shard config",
            "actual_pair_count": 0,
            "complete_pair_count": 0,
        }
    shard_config = load_json(config_path)
    expected_ids = [str(item["pair_id"]) for item in shard_config["pair_requests"]]
    report["configured_pair_count"] = len(expected_ids)
    report["config_pair_ids_unique"] = len(expected_ids) == len(set(expected_ids))
    report["config_count_matches_manifest"] = len(expected_ids) == expected_count
    if not root.is_dir():
        return report | {
            "ready": False,
            "actual_pair_count": 0,
            "complete_pair_count": 0,
            "missing_pair_count": expected_count,
            "missing_pair_examples": expected_ids[:max_examples],
            "unknown_pair_count": 0,
            "incomplete_pair_count": 0,
            "incomplete_pair_examples": [],
        }

    pairs_root = root / "pairs"
    actual_ids = sorted(
        path.name for path in pairs_root.iterdir() if path.is_dir()
    ) if pairs_root.is_dir() else []
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    incomplete: list[dict[str, Any]] = []
    complete_count = 0
    for pair_id in sorted(expected_set & actual_set):
        issues = audit_pair_directory(
            pairs_root / pair_id,
            pair_id=pair_id,
            expected_dense_rows=expected_dense_rows,
            expected_rgb_frames=expected_rgb_frames,
            camera_names=camera_names,
            deep=deep,
        )
        if issues:
            if len(incomplete) < max_examples:
                incomplete.append({"pair_id": pair_id, "issues": issues[:20]})
        else:
            complete_count += 1

    snapshot = root / "configs" / "experiment_config.snapshot.json"
    summary = root / "summaries" / "dataset_summary.json"
    report.update(
        {
            "pairs_directory_available": pairs_root.is_dir(),
            "config_snapshot_available": snapshot.is_file(),
            "dataset_summary_available": summary.is_file(),
            "actual_pair_count": len(actual_ids),
            "complete_pair_count": complete_count,
            "missing_pair_count": len(missing),
            "missing_pair_examples": missing[:max_examples],
            "unknown_pair_count": len(unknown),
            "unknown_pair_examples": unknown[:max_examples],
            "incomplete_pair_count": len(expected_set & actual_set) - complete_count,
            "incomplete_pair_examples": incomplete,
        }
    )
    report["ready"] = all(
        (
            report["config_pair_ids_unique"],
            report["config_count_matches_manifest"],
            report["pairs_directory_available"],
            report["config_snapshot_available"],
            report["dataset_summary_available"],
            not missing,
            not unknown,
            report["incomplete_pair_count"] == 0,
            complete_count == expected_count,
        )
    )
    return report


def build_preflight_report(
    config: DatasetConfig,
    shard_manifest_path: Path,
    *,
    deep: bool,
    max_examples: int = 20,
) -> dict[str, Any]:
    manifest = load_json(shard_manifest_path)
    sim_root = config.source_config.parent.parent
    shard_reports = [
        audit_shard(
            sim_root=sim_root,
            shard_spec=spec,
            expected_dense_rows=int(config.expected["physics_rows_per_run"]),
            expected_rgb_frames=int(config.expected["rgb_frames_per_camera_per_run"]),
            camera_names=tuple(str(item) for item in config.expected["camera_names"]),
            deep=deep,
            max_examples=max_examples,
        )
        for spec in manifest["shards"]
    ]
    configured_ids: list[str] = []
    for spec in manifest["shards"]:
        config_path = (sim_root / str(spec["config"])).resolve()
        if config_path.is_file():
            configured_ids.extend(
                str(item["pair_id"])
                for item in load_json(config_path)["pair_requests"]
            )
    source_ids = {
        str(item["pair_id"])
        for item in load_json(config.source_config)["pair_requests"]
    }
    configured_set = set(configured_ids)
    config_partition_ready = (
        len(configured_ids) == len(configured_set)
        and configured_set == source_ids
        and len(source_ids) == int(manifest["source_pair_count"])
        and len(source_ids) == int(config.expected["pair_count"])
    )
    ready = config_partition_ready and all(item.get("ready", False) for item in shard_reports)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "deep" if deep else "required-files",
        "ready": ready,
        "interpretation": (
            "These are generation-work shards of one shared pair grid, not independent "
            "confirmatory test shards."
        ),
        "config": str(config.config_path),
        "shard_manifest": str(shard_manifest_path),
        "expected_pair_count": int(config.expected["pair_count"]),
        "expected_run_count": int(config.expected["run_count"]),
        "config_partition": {
            "ready": config_partition_ready,
            "source_pair_count": len(source_ids),
            "configured_pair_occurrence_count": len(configured_ids),
            "configured_unique_pair_count": len(configured_set),
            "missing_from_shards": sorted(source_ids - configured_set)[:max_examples],
            "unknown_in_shards": sorted(configured_set - source_ids)[:max_examples],
        },
        "summary": {
            "available_shards": sum(bool(item["available"]) for item in shard_reports),
            "ready_shards": sum(bool(item.get("ready")) for item in shard_reports),
            "completed_pairs": sum(int(item["complete_pair_count"]) for item in shard_reports),
            "missing_pairs": sum(int(item.get("missing_pair_count", 0)) for item in shard_reports),
            "incomplete_pairs": sum(int(item.get("incomplete_pair_count", 0)) for item in shard_reports),
        },
        "shards": shard_reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify all ECCV generation shards before rebuilding full-grid manifests."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/local_dataset.json")
    )
    parser.add_argument("--shard-manifest", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/preflight/full_shard_preflight.json")
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also verify dense row counts and every expected RGB frame count.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")
    config = DatasetConfig.load(args.config)
    shard_manifest = (
        args.shard_manifest.expanduser().resolve()
        if args.shard_manifest is not None
        else config.source_config.parent / "eccv_workshop_shards" / "manifest.json"
    )
    if not shard_manifest.is_file():
        raise FileNotFoundError(shard_manifest)
    report = build_preflight_report(
        config, shard_manifest, deep=args.deep, max_examples=args.max_examples
    )
    output = args.output.expanduser().resolve()
    write_json(output, report)
    print(
        json.dumps(
            {
                "ready": report["ready"],
                **report["summary"],
                "output": str(output),
            },
            indent=2,
        )
    )
    if args.strict and not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

