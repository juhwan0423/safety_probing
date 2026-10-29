from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

from .adapters.ltx_continuation import LTXContinuationRunner
from .io_utils import load_json, write_json, write_jsonl
from .smoke_inputs import encode_prefix_video
from .vjepa_features import select_canonical_rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_batch_rows(
    rows: list[dict[str, Any]],
    *,
    shard: str,
    splits: set[str],
    split_protocol: str = "main",
    max_conditions: int | None = None,
    max_per_outcome: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in select_canonical_rows(rows, shard=shard)
        if str(row["splits"][split_protocol]) in splits
    ]
    selected.sort(
        key=lambda row: (
            str(row["splits"][split_protocol]),
            str(row["condition_id"]),
        )
    )
    condition_ids = [str(row["condition_id"]) for row in selected]
    if len(condition_ids) != len(set(condition_ids)):
        raise RuntimeError("Canonical manifest contains duplicate condition_id values")
    if max_per_outcome is not None:
        if max_per_outcome <= 0:
            raise ValueError("max_per_outcome must be positive")
        counts: Counter[str] = Counter()
        balanced: list[dict[str, Any]] = []
        for row in selected:
            outcome = str(row["safety_outcome"])
            if counts[outcome] < max_per_outcome:
                balanced.append(row)
                counts[outcome] += 1
        selected = balanced
    if max_conditions is not None:
        if max_conditions <= 0:
            raise ValueError("max_conditions must be positive")
        selected = selected[:max_conditions]
    return selected


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _record_complete(
    sidecar: Path, *, output_root: Path, experiment_sha256: str
) -> bool:
    if not sidecar.is_file():
        return False
    record = load_json(sidecar)
    if record.get("status") != "complete":
        return False
    if record.get("experiment_sha256") != experiment_sha256:
        return False
    video = output_root / str(record["video_path"])
    latent = output_root / str(record["latent_path"])
    return video.is_file() and latent.is_file()


def _collect_records(output_root: Path) -> list[dict[str, Any]]:
    records = [load_json(path) for path in output_root.rglob("seed_*.json")]
    records.sort(
        key=lambda row: (
            str(row.get("split")),
            str(row.get("condition_id")),
            int(row.get("seed", -1)),
        )
    )
    return records


def _write_batch_state(
    output_root: Path,
    *,
    requested_jobs: int,
    requested_conditions: int,
    interrupted: bool = False,
) -> dict[str, Any]:
    records = _collect_records(output_root)
    counts = Counter(str(row.get("status", "unknown")) for row in records)
    write_jsonl(output_root / "batch_manifest.jsonl", records)
    summary = {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_condition_count": requested_conditions,
        "requested_job_count": requested_jobs,
        "record_count": len(records),
        "status_counts": dict(sorted(counts.items())),
        "interrupted": interrupted,
    }
    write_json(output_root / "batch_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run resumable LTX continuation rollouts with one persistent model load."
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/ltx_2_3_continuation.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/current/manifests/canonical_runs.jsonl"),
    )
    parser.add_argument("--shard", default="part_01")
    parser.add_argument(
        "--split", action="append", choices=("train", "val", "test")
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--max-per-outcome", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_conditions is not None and args.max_per_outcome is not None:
        raise ValueError("Use only one of --max-conditions and --max-per-outcome")
    registry_path = args.registry.resolve()
    experiment_path = args.experiment.resolve()
    manifest_path = args.manifest.resolve()
    project_root = registry_path.parent.parent
    registry = load_json(registry_path)
    experiment = load_json(experiment_path)
    model = registry["models"]["ltx_2_3"]
    runtime = model["runtime"]
    input_spec = experiment["input"]
    generation = experiment["generation"]
    splits = set(args.split or ["val", "test"])
    seeds = list(args.seed or generation["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed values are not allowed")

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            project_root / "outputs" / str(experiment["experiment"]) / args.shard
        ).resolve()
    )
    rows = select_batch_rows(
        _read_jsonl(manifest_path),
        shard=args.shard,
        splits=splits,
        max_conditions=args.max_conditions,
        max_per_outcome=args.max_per_outcome,
    )
    if not rows:
        raise RuntimeError(
            f"No canonical rows found for shard={args.shard}, splits={sorted(splits)}"
        )

    experiment_sha256 = _sha256(experiment_path)
    registry_sha256 = _sha256(registry_path)
    jobs = [(row, seed) for row in rows for seed in seeds]
    condition_counts = Counter(str(row["splits"]["main"]) for row in rows)
    outcome_counts = Counter(str(row["safety_outcome"]) for row in rows)
    plan = {
        "schema_version": 1,
        "shard": args.shard,
        "splits": sorted(splits),
        "seeds": seeds,
        "condition_count": len(rows),
        "job_count": len(jobs),
        "condition_counts_by_split": dict(sorted(condition_counts.items())),
        "condition_counts_by_outcome": dict(sorted(outcome_counts.items())),
        "experiment": str(experiment_path),
        "experiment_sha256": experiment_sha256,
        "registry_sha256": registry_sha256,
        "output_root": str(output_root),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "batch_contract.json"
    if contract_path.is_file() and not args.overwrite:
        previous = load_json(contract_path)
        for key in ("shard", "experiment_sha256", "registry_sha256"):
            if previous.get(key) != plan.get(key):
                raise RuntimeError(
                    f"Existing batch contract differs at {key}; use a new output root"
                )
    write_json(contract_path, plan)
    write_json(output_root / "experiment.snapshot.json", experiment)
    write_json(output_root / "model_registry.snapshot.json", registry)

    prefix_frames = int(input_spec["frame_count"])
    fps = int(input_spec["fps"])
    camera = str(input_spec["camera"])
    prefix_root = output_root / f"_prefixes_{prefix_frames}f_{fps}fps"
    print(f"[LTX batch] Preparing {len(rows)} prefix videos", flush=True)
    for index, row in enumerate(rows, start=1):
        prefix_path = prefix_root / f"{row['condition_id']}.mp4"
        if args.overwrite or not prefix_path.is_file():
            encode_prefix_video(
                Path(row["rgb_dirs"][camera]),
                prefix_path,
                frame_count=prefix_frames,
                fps=fps,
            )
        if index % 25 == 0 or index == len(rows):
            print(f"[LTX batch] Prefixes {index}/{len(rows)}", flush=True)
    if args.prefix_only:
        return

    pending: list[tuple[dict[str, Any], int, Path, Path, Path]] = []
    for row, seed in jobs:
        split = str(row["splits"]["main"])
        condition_root = output_root / split / str(row["condition_id"])
        video_path = condition_root / f"seed_{seed:04d}.mp4"
        latent_path = condition_root / f"seed_{seed:04d}.pt"
        sidecar_path = condition_root / f"seed_{seed:04d}.json"
        if not args.overwrite and _record_complete(
            sidecar_path,
            output_root=output_root,
            experiment_sha256=experiment_sha256,
        ):
            continue
        pending.append((row, seed, video_path, latent_path, sidecar_path))

    print(
        f"[LTX batch] Complete={len(jobs) - len(pending)}, pending={len(pending)}",
        flush=True,
    )
    if not pending:
        summary = _write_batch_state(
            output_root,
            requested_jobs=len(jobs),
            requested_conditions=len(rows),
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    width, height = (int(value) for value in input_spec["resize"])
    source_total_frames = prefix_frames + int(generation["evaluation_frame_count"])
    runner = LTXContinuationRunner(
        code_root=project_root / registry["code_root"] / model["code"]["directory"],
        checkpoint=project_root / runtime["checkpoint"],
        text_encoder=project_root / runtime["text_encoder"],
        spatial_upscaler=project_root / runtime["spatial_upscaler"],
        quantization=runtime["quantization"],
        offload=runtime["offload"],
        progress=lambda message: print(f"[LTX] {message}", flush=True),
    )

    interrupted = False
    try:
        for job_index, (row, seed, video_path, latent_path, sidecar_path) in enumerate(
            pending, start=1
        ):
            split = str(row["splits"]["main"])
            condition_id = str(row["condition_id"])
            prefix_path = prefix_root / f"{condition_id}.mp4"
            print(
                f"[LTX batch] Job {job_index}/{len(pending)} "
                f"split={split} condition={condition_id} seed={seed}",
                flush=True,
            )
            started = time.monotonic()
            base_record = {
                "schema_version": 1,
                "condition_id": condition_id,
                "source_run_id": row["run_id"],
                "source_shard": args.shard,
                "split": split,
                "safety_outcome": row["safety_outcome"],
                "future_tip_label": row.get("future_tip_label"),
                "seed": seed,
                "experiment_sha256": experiment_sha256,
                "registry_sha256": registry_sha256,
                "prefix_frame_count": prefix_frames,
                "prefix_fps": fps,
                "prefix_path": _relative(prefix_path, output_root),
                "video_path": _relative(video_path, output_root),
                "latent_path": _relative(latent_path, output_root),
            }
            try:
                result = runner.run(
                    prefix_video=prefix_path,
                    output_path=video_path,
                    latent_output_path=latent_path,
                    prompt=generation["prompt"],
                    seed=seed,
                    width=width,
                    height=height,
                    fps=fps,
                    prefix_frames=prefix_frames,
                    source_total_frames=source_total_frames,
                    internal_total_frames=int(generation["internal_total_frame_count"]),
                )
                record = {
                    **base_record,
                    "status": "complete",
                    "duration_s": round(time.monotonic() - started, 3),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "latent_shape": result["latent_shape"],
                    "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                }
            except Exception as exc:
                record = {
                    **base_record,
                    "status": "failed",
                    "duration_s": round(time.monotonic() - started, 3),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(sidecar_path, record)
                if args.fail_fast:
                    raise
            else:
                write_json(sidecar_path, record)
    except KeyboardInterrupt:
        interrupted = True
        print("[LTX batch] Interrupted; completed sidecars are resumable", flush=True)
        raise
    finally:
        summary = _write_batch_state(
            output_root,
            requested_jobs=len(jobs),
            requested_conditions=len(rows),
            interrupted=interrupted,
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
