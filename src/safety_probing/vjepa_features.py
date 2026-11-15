from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .adapters.vjepa import load_video_tensor, load_vjepa, uniform_frame_indices
from .io_utils import load_json, write_json, write_torch_atomic


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "condition_id": row["condition_id"],
            "run_id": row["run_id"],
            "pair_id": row["pair_id"],
            "run_label": row["run_label"],
            "split": row["splits"]["main"],
            "safety_outcome": row["safety_outcome"],
            "binary_tip_label": row.get("binary_tip_label"),
            "future_tip_label": row.get("future_tip_label"),
            "future_prediction_eligible": row["future_prediction_eligible"],
            "future_min_outward_zmp_margin_m": row["dense_summary"].get(
                "future_min_outward_zmp_margin_m"
            ),
            "turn_direction": row["turn_direction"],
            "speed_label": row["speed_label"],
            "steering_label": row["steering_label"],
            "intervention_type": row["intervention_type"],
            "dataset_root": row["dataset_root"],
        }
        for row in rows
    ]


def select_canonical_rows(
    rows: list[dict[str, Any]], *, shard: str | None = None
) -> list[dict[str, Any]]:
    dataset_name = None
    if shard is not None and shard.lower() not in {"all", "full"}:
        dataset_name = (
            f"eccv_workshop_grid_{shard}" if shard.startswith("part_") else shard
        )
    selected = [
        row
        for row in rows
        if dataset_name is None
        or Path(str(row["dataset_root"])).name == dataset_name
    ]
    selected.sort(key=lambda row: str(row["condition_id"]))
    condition_ids = [str(row["condition_id"]) for row in selected]
    if len(condition_ids) != len(set(condition_ids)):
        raise RuntimeError("Canonical manifest contains duplicate condition IDs")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache frozen V-JEPA temporal features for observation clips."
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_linear_probe.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/current/manifests/canonical_runs.jsonl"),
    )
    parser.add_argument("--model", choices=("vjepa_2", "vjepa_2_1"))
    parser.add_argument("--shard")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    registry_path = args.registry.resolve()
    experiment_path = args.experiment.resolve()
    manifest_path = args.manifest.resolve()
    project_root = registry_path.parent.parent
    registry = load_json(registry_path)
    experiment = load_json(experiment_path)
    model_key = args.model or str(experiment["primary_model_registry_key"])
    model_spec = registry["models"][model_key]
    runtime = model_spec["runtime"]
    version = "2.1" if model_key == "vjepa_2_1" else "2"
    resolution = int(runtime["resolution"])
    input_spec = experiment["input"]
    frame_count = int(input_spec["sampled_frame_count"])
    source_frame_count = int(input_spec["source_frame_count"])
    frame_indices = uniform_frame_indices(
        0, source_frame_count - 1, frame_count
    )

    rows = select_canonical_rows(_read_jsonl(manifest_path), shard=args.shard)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    if not rows:
        raise RuntimeError("No canonical rows selected for feature extraction")

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "artifacts"
            / "features"
            / f"{model_key}_{args.shard or 'all_shards'}_observation.pt"
        ).resolve()
    )
    preprocessing = {
        "camera": input_spec["camera"],
        "resolution": resolution,
        "source_frame_count": source_frame_count,
        "frame_indices": frame_indices,
        "resize": "short_side_resolution_times_256_over_224",
        "crop": "center_square",
        "normalization": "imagenet_mean_std",
        "tubelet_size": 2,
        "spatial_pooling": "mean_per_temporal_tubelet",
    }
    preprocessing_sha256 = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "schema_version": 1,
        "model_registry_key": model_key,
        "model_code_revision": model_spec["code"]["revision"],
        "experiment_sha256": _sha256(experiment_path),
        "registry_sha256": _sha256(registry_path),
        "manifest_sha256": _sha256(manifest_path),
        "preprocessing_sha256": preprocessing_sha256,
        "shard": args.shard,
        "run_count": len(rows),
        "condition_ids": [str(row["condition_id"]) for row in rows],
    }
    plan = {
        **contract,
        "output": str(output_path),
        "batch_size": args.batch_size,
        "device": args.device,
        "preprocessing": preprocessing,
    }
    display_plan = {
        **plan,
        "condition_ids": {
            "count": len(contract["condition_ids"]),
            "first": contract["condition_ids"][0],
            "last": contract["condition_ids"][-1],
        },
    }
    print(json.dumps(display_plan, indent=2), flush=True)
    if args.dry_run:
        return

    if output_path.is_file() and not args.overwrite:
        import torch

        existing = torch.load(
            output_path, map_location="cpu", mmap=True, weights_only=True
        )
        if existing.get("contract") != contract:
            raise RuntimeError(
                "Existing feature cache contract differs; use --overwrite or a new path"
            )
        print(f"[V-JEPA features] Reusing complete cache: {output_path}")
        return

    import torch

    print(f"[V-JEPA features] Loading frozen {model_key} encoder", flush=True)
    encoder, predictor = load_vjepa(
        code_root=(
            project_root
            / registry["code_root"]
            / model_spec["code"]["directory"]
        ),
        checkpoint=project_root / runtime["checkpoint"],
        version=version,
        num_frames=frame_count,
        device=args.device,
    )
    del predictor
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    temporal_token_count = frame_count // 2
    spatial_token_count = (resolution // 16) ** 2
    feature_batches = []
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        videos = []
        for row in batch_rows:
            video, loaded_indices = load_video_tensor(
                row["rgb_dirs"][input_spec["camera"]],
                frame_count=frame_count,
                source_frame_count=source_frame_count,
                resolution=resolution,
            )
            if loaded_indices != frame_indices:
                raise RuntimeError("Feature frame sampling contract changed")
            videos.append(video)
        video_batch = torch.cat(videos, dim=0).to(args.device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=args.device.startswith("cuda"),
        ):
            tokens = encoder(video_batch)
            if isinstance(tokens, list):
                tokens = tokens[-1]
            expected_tokens = temporal_token_count * spatial_token_count
            if tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    f"Expected {expected_tokens} tokens, got {tokens.shape[1]}"
                )
            temporal_features = tokens.reshape(
                len(batch_rows),
                temporal_token_count,
                spatial_token_count,
                tokens.shape[-1],
            ).mean(dim=2)
        if not bool(torch.isfinite(temporal_features).all()):
            raise RuntimeError("V-JEPA produced non-finite cached features")
        feature_batches.append(temporal_features.to(dtype=torch.float16).cpu())
        completed = min(start + len(batch_rows), len(rows))
        if completed % 25 < args.batch_size or completed == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"[V-JEPA features] {completed}/{len(rows)} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    features = torch.cat(feature_batches, dim=0)
    payload = {
        "schema_version": 1,
        "contract": contract,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "registry_key": model_key,
            "display_name": model_spec["display_name"],
            "checkpoint": str((project_root / runtime["checkpoint"]).resolve()),
            "foundation_parameters_frozen": True,
        },
        "preprocessing": preprocessing,
        "records": _cache_records(rows),
        "features": features,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else 0
        ),
        "duration_s": round(time.monotonic() - started, 3),
    }
    write_torch_atomic(output_path, payload)
    write_json(
        output_path.with_suffix(".json"),
        {key: value for key, value in payload.items() if key != "features"},
    )
    print(
        f"[V-JEPA features] Wrote {features.shape} to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
