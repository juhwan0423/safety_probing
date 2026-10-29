from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .adapters.vjepa import load_video_tensor, load_vjepa, uniform_frame_indices
from .io_utils import load_json, write_json, write_torch_atomic
from .vjepa_features import _read_jsonl, _sha256, select_canonical_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache spatially resolved frozen V-JEPA observation tokens."
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
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
    model_key = str(experiment["primary_model_registry_key"])
    model_spec = registry["models"][model_key]
    runtime = model_spec["runtime"]
    input_config = experiment["input"]
    cache_config = experiment["feature_cache"]
    resolution = int(runtime["resolution"])
    frame_count = int(input_config["sampled_frame_count"])
    source_frame_count = int(input_config["source_frame_count"])
    frame_indices = uniform_frame_indices(
        0, source_frame_count - 1, frame_count
    )
    source_grid = int(cache_config["source_spatial_grid"])
    cached_grid = int(cache_config["cached_spatial_grid"])
    if source_grid != resolution // 16:
        raise RuntimeError("Configured source grid does not match ViT patch grid")
    if source_grid % cached_grid:
        raise RuntimeError("Cached spatial grid must divide the source grid")

    rows = select_canonical_rows(
        _read_jsonl(manifest_path), shard=args.shard
    )
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    if not rows:
        raise RuntimeError("No canonical rows selected for spatial features")
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "artifacts"
            / "features"
            / f"{model_key}_{args.shard}_observation_spatial{cached_grid}.pt"
        ).resolve()
    )
    preprocessing = {
        "camera": input_config["camera"],
        "resolution": resolution,
        "source_frame_count": source_frame_count,
        "frame_indices": frame_indices,
        "resize": "short_side_resolution_times_256_over_224",
        "crop": "center_square",
        "normalization": "imagenet_mean_std",
        "tubelet_size": 2,
        "source_spatial_grid": source_grid,
        "cached_spatial_grid": cached_grid,
        "spatial_reduction": cache_config["spatial_reduction"],
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
    display = {
        **contract,
        "condition_ids": {
            "count": len(rows),
            "first": rows[0]["condition_id"],
            "last": rows[-1]["condition_id"],
        },
        "output": str(output_path),
        "batch_size": args.batch_size,
        "device": args.device,
        "preprocessing": preprocessing,
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
                "Existing spatial cache contract differs; use --overwrite"
            )
        print(f"[V-JEPA spatial] Reusing complete cache: {output_path}")
        return

    import torch
    import torch.nn.functional as functional

    print(f"[V-JEPA spatial] Loading frozen {model_key} encoder", flush=True)
    version = "2.1" if model_key == "vjepa_2_1" else "2"
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

    temporal_tokens = frame_count // 2
    feature_batches = []
    started = time.monotonic()
    reduction = source_grid // cached_grid
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        videos = []
        for row in batch_rows:
            video, loaded_indices = load_video_tensor(
                row["rgb_dirs"][input_config["camera"]],
                frame_count=frame_count,
                source_frame_count=source_frame_count,
                resolution=resolution,
            )
            if loaded_indices != frame_indices:
                raise RuntimeError("Spatial feature frame contract changed")
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
            expected = temporal_tokens * source_grid * source_grid
            if tokens.shape[1] != expected:
                raise RuntimeError(f"Expected {expected} tokens, got {tokens.shape[1]}")
            spatial = tokens.reshape(
                len(batch_rows),
                temporal_tokens,
                source_grid,
                source_grid,
                tokens.shape[-1],
            )
            if reduction > 1:
                pooled = spatial.permute(0, 1, 4, 2, 3).reshape(
                    len(batch_rows) * temporal_tokens,
                    tokens.shape[-1],
                    source_grid,
                    source_grid,
                )
                pooled = functional.avg_pool2d(
                    pooled, kernel_size=reduction, stride=reduction
                )
                spatial = pooled.reshape(
                    len(batch_rows),
                    temporal_tokens,
                    tokens.shape[-1],
                    cached_grid,
                    cached_grid,
                ).permute(0, 1, 3, 4, 2)
        if not bool(torch.isfinite(spatial).all()):
            raise RuntimeError("V-JEPA produced non-finite spatial features")
        feature_batches.append(spatial.to(dtype=torch.float16).cpu())
        completed = min(start + len(batch_rows), len(rows))
        if completed % 25 < args.batch_size or completed == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"[V-JEPA spatial] {completed}/{len(rows)} ({elapsed:.1f}s)",
                flush=True,
            )

    features = torch.cat(feature_batches, dim=0)
    records = [
        {
            "condition_id": row["condition_id"],
            "run_id": row["run_id"],
            "split": row["splits"]["main"],
            "safety_outcome": row["safety_outcome"],
        }
        for row in rows
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "model": {
            "registry_key": model_key,
            "display_name": model_spec["display_name"],
            "checkpoint": str((project_root / runtime["checkpoint"]).resolve()),
            "foundation_parameters_frozen": True,
        },
        "preprocessing": preprocessing,
        "records": records,
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
        f"[V-JEPA spatial] Wrote {features.shape} to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
