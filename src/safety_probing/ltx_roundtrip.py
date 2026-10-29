from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .adapters.ltx_continuation import activate_ltx_code
from .io_utils import load_json, write_json, write_torch_atomic
from .ltx_real_latents import decode_frame_directory, raw_rgb_video_tensor
from .vjepa_features import _read_jsonl, _sha256


EXPECTED_FEATURE_SHAPE = (7, 10, 18, 128)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _path_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _frame_directory_signature(
    directory: Path, *, source_frames: int
) -> dict[str, Any]:
    entries = []
    for frame_index in range(source_frames):
        path = directory / f"frame_{frame_index:06d}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        entries.append((path.name, stat.st_size, stat.st_mtime_ns))
    return {
        "directory": str(directory.resolve()),
        "frames": source_frames,
        "listing_sha256": _json_sha256(entries),
    }


def _job_fingerprint(
    contract_fingerprint: str,
    *,
    condition_id: str,
    decoder_seed: int,
    source_signature: dict[str, Any],
) -> str:
    return _json_sha256(
        {
            "contract_fingerprint": contract_fingerprint,
            "condition_id": condition_id,
            "decoder_seed": decoder_seed,
            "source_signature": source_signature,
        }
    )


def _part_complete(
    path: Path, *, fingerprint: str, expected_shape=EXPECTED_FEATURE_SHAPE
) -> bool:
    if not path.is_file():
        return False
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return False
    feature = payload.get("feature")
    return (
        payload.get("fingerprint") == fingerprint
        and hasattr(feature, "shape")
        and tuple(feature.shape) == tuple(expected_shape)
        and bool(torch.isfinite(feature).all())
    )


def select_condition_rows(
    rows: list[dict[str, Any]],
    *,
    splits: set[str],
    condition_ids: list[str] | None = None,
    max_conditions: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows if str(row["splits"]["main"]) in splits
    ]
    by_condition = {str(row["condition_id"]): row for row in selected}
    if len(by_condition) != len(selected):
        raise RuntimeError("Canonical manifest contains duplicate condition IDs")
    if condition_ids:
        missing = sorted(set(condition_ids) - set(by_condition))
        if missing:
            raise KeyError(f"Unknown selected condition: {missing[0]}")
        selected = [by_condition[condition_id] for condition_id in condition_ids]
    else:
        selected = sorted(selected, key=lambda row: str(row["condition_id"]))
    if max_conditions is not None:
        selected = selected[:max_conditions]
    if not selected:
        raise RuntimeError("No validation/test conditions selected")
    return selected


def decoded_chunks_to_encoder_input(chunks, *, expected_frames: int):
    import torch

    chunks = list(chunks)
    if not chunks:
        raise RuntimeError("LTX decoder returned no pixel chunks")
    pixels = torch.cat(chunks, dim=0)
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise RuntimeError(
            f"Expected decoded chunks [F,H,W,3], got {tuple(pixels.shape)}"
        )
    if len(pixels) != expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} decoded frames, got {len(pixels)}"
        )
    if not bool(torch.isfinite(pixels).all()):
        raise RuntimeError("LTX decoder returned non-finite pixels")
    pixel_min = float(pixels.min())
    pixel_max = float(pixels.max())
    if pixel_min < 0.0 or pixel_max > 1.0:
        raise RuntimeError(
            f"Decoded pixels outside [0,1]: min={pixel_min}, max={pixel_max}"
        )
    video = pixels.permute(3, 0, 1, 2).unsqueeze(0)
    video = video.mul(2.0).sub(1.0)
    return video, {
        "decoded_frame_count": len(pixels),
        "decoded_pixel_min": pixel_min,
        "decoded_pixel_max": pixel_max,
        "decoded_pixel_mean": float(pixels.mean()),
        "decoded_pixel_std": float(pixels.float().std()),
    }


def _future_feature(latent, *, prefix_latent_frames: int):
    import torch

    feature = latent[0, :, prefix_latent_frames:].permute(1, 2, 3, 0)
    feature = feature.to(device="cpu", dtype=torch.float16)
    if tuple(feature.shape) != EXPECTED_FEATURE_SHAPE:
        raise RuntimeError(
            f"Expected future feature {EXPECTED_FEATURE_SHAPE}, got {tuple(feature.shape)}"
        )
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("LTX VAE returned non-finite round-trip features")
    return feature


class LTXVAERoundTripper:
    def __init__(self, *, checkpoint: Path, device: str) -> None:
        import torch
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.loader.single_gpu_model_builder import (
            SingleGPUModelBuilder as Builder,
        )
        from ltx_core.model.video_vae import (
            MEMORY_EFFICIENT_DECODE,
            VAE_DECODER_COMFY_KEYS_FILTER,
            VAE_ENCODER_COMFY_KEYS_FILTER,
            TilingConfig,
            VideoDecoderConfigurator,
            VideoEncoderConfigurator,
        )

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.tiling = TilingConfig.default()
        registry = DummyRegistry()
        encoder_builder = Builder(
            model_path=str(checkpoint),
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry,
        )
        decoder_builder = Builder(
            model_path=str(checkpoint),
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            module_ops=(MEMORY_EFFICIENT_DECODE,),
            registry=registry,
        )
        self.decoder = decoder_builder.build(
            device=self.device, dtype=self.dtype
        ).eval()
        self.encoder = encoder_builder.build(
            device=self.device, dtype=self.dtype
        ).eval()

    def encode(self, video):
        with self.torch.inference_mode():
            return self.encoder.tiled_encode(video, self.tiling)

    def roundtrip(self, latent, *, decoder_seed: int, expected_frames: int):
        latent = latent.to(device=self.device, dtype=self.dtype)
        generator = self.torch.Generator(device=self.device).manual_seed(
            decoder_seed
        )
        with self.torch.inference_mode():
            chunks = self.decoder.decode_video(
                latent,
                self.tiling,
                generator=generator,
            )
            video, pixel_audit = decoded_chunks_to_encoder_input(
                chunks, expected_frames=expected_frames
            )
            reencoded = self.encoder.tiled_encode(video, self.tiling)
        return reencoded, pixel_audit

    def close(self) -> None:
        self.encoder.to("meta")
        self.decoder.to("meta")
        del self.encoder, self.decoder
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode and re-encode GT or generated LTX full-sequence latents."
    )
    parser.add_argument("--source", choices=("gt", "generated"), required=True)
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path("configs/experiments/ltx_roundtrip_control.json"),
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--query-experiment",
        type=Path,
        default=Path("configs/experiments/ltx_safety_query.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/current/manifests/canonical_runs.jsonl"),
    )
    parser.add_argument(
        "--real-cache",
        type=Path,
        default=Path("artifacts/features/ltx_2_3_full_real_future.pt"),
    )
    parser.add_argument(
        "--rollout-cache",
        type=Path,
        default=Path("artifacts/features/ltx_2_3_full_rollout_future.pt"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--condition-id", action="append")
    parser.add_argument("--decoder-seed", action="append", type=int)
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:  # noqa: PLR0915
    args = _parser().parse_args()
    if args.max_conditions is not None and args.max_conditions <= 0:
        raise ValueError("--max-conditions must be positive")
    control_path = args.control_config.resolve()
    registry_path = args.registry.resolve()
    query_path = args.query_experiment.resolve()
    manifest_path = args.manifest.resolve()
    real_path = args.real_cache.resolve()
    rollout_path = args.rollout_cache.resolve()
    project_root = registry_path.parent.parent
    control = load_json(control_path)
    registry = load_json(registry_path)
    query = load_json(query_path)
    model_key = str(control["primary_model_registry_key"])
    model_spec = registry["models"][model_key]
    latent_config = query["real_latent"]
    seeds = list(args.decoder_seed or control["decoder_seeds"])
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Decoder seeds must be unique non-negative integers")

    import torch

    real_cache = torch.load(real_path, map_location="cpu", mmap=True, weights_only=True)
    if real_cache["contract"]["experiment_sha256"] != _sha256(query_path):
        raise RuntimeError("Real cache and query experiment hashes differ")
    all_rows = _read_jsonl(manifest_path)
    rows = select_condition_rows(
        all_rows,
        splits=set(control["splits"]),
        condition_ids=args.condition_id,
        max_conditions=args.max_conditions,
    )
    real_index = {
        str(record["condition_id"]): index
        for index, record in enumerate(real_cache["records"])
    }
    missing_real = [
        str(row["condition_id"])
        for row in rows
        if str(row["condition_id"]) not in real_index
    ]
    if missing_real:
        raise RuntimeError(f"Real cache is missing condition {missing_real[0]}")

    rollout_cache = None
    rollout_index: dict[tuple[str, int], int] = {}
    if args.source == "generated":
        rollout_cache = torch.load(
            rollout_path, map_location="cpu", mmap=True, weights_only=True
        )
        if rollout_cache["contract"]["experiment_sha256"] != _sha256(query_path):
            raise RuntimeError("Rollout cache and query experiment hashes differ")
        rollout_index = {
            (str(record["condition_id"]), int(record["rollout_seed"])): index
            for index, record in enumerate(rollout_cache["records"])
        }
        missing_rollout = [
            (str(row["condition_id"]), seed)
            for row in rows
            for seed in seeds
            if (str(row["condition_id"]), seed) not in rollout_index
        ]
        if missing_rollout:
            raise RuntimeError(f"Rollout cache is missing job {missing_rollout[0]}")

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "artifacts"
            / "features"
            / f"ltx_2_3_full_{args.source}_roundtrip_future.pt"
        ).resolve()
    )
    work_root = (
        args.work_root.resolve()
        if args.work_root is not None
        else output_path.with_name(f"{output_path.stem}_parts")
    )
    checkpoint = (project_root / model_spec["runtime"]["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    condition_ids = [str(row["condition_id"]) for row in rows]
    contract = {
        "schema_version": 1,
        "protocol": control["protocol_version"],
        "source": args.source,
        "model_registry_key": model_key,
        "model_code_revision": model_spec["code"]["revision"],
        "model_hub_revision": model_spec["hub_artifacts"][0]["revision"],
        "checkpoint": _path_signature(checkpoint),
        "control_config_sha256": _sha256(control_path),
        "query_experiment_sha256": _sha256(query_path),
        "registry_sha256": _sha256(registry_path),
        "manifest_sha256": _sha256(manifest_path),
        "real_cache_contract": real_cache["contract"],
        "rollout_cache_contract": (
            rollout_cache["contract"] if rollout_cache is not None else None
        ),
        "condition_ids": condition_ids,
        "decoder_seeds": seeds,
        "job_count": len(rows) * len(seeds),
        "intermediate_codec": "none",
        "decode_scope": "full_16_latent_tokens",
        "reencode_scope": "full_121_pixel_frames",
        "future_slice": [
            int(latent_config["prefix_latent_frames"]),
            int(latent_config["latent_total_frames"]),
        ],
    }
    contract_fingerprint = _json_sha256(contract)
    preprocessing = real_cache["preprocessing"]

    jobs = []
    for row in rows:
        condition_id = str(row["condition_id"])
        if args.source == "gt":
            source_signature = _frame_directory_signature(
                Path(row["rgb_dirs"][query["input"]["camera"]]),
                source_frames=int(latent_config["source_total_frames"]),
            )
        else:
            source_signature = None
        for seed in seeds:
            rollout_record = None
            if args.source == "generated":
                rollout_record = rollout_cache["records"][
                    rollout_index[(condition_id, seed)]
                ]
                source_signature = _path_signature(
                    Path(rollout_record["latent_path"])
                )
            fingerprint = _job_fingerprint(
                contract_fingerprint,
                condition_id=condition_id,
                decoder_seed=seed,
                source_signature=source_signature,
            )
            part_path = work_root / f"{condition_id}_seed_{seed:04d}.pt"
            jobs.append(
                {
                    "row": row,
                    "decoder_seed": seed,
                    "rollout_record": rollout_record,
                    "source_signature": source_signature,
                    "fingerprint": fingerprint,
                    "part_path": part_path,
                }
            )

    pending = [
        job
        for job in jobs
        if args.overwrite
        or not _part_complete(
            job["part_path"], fingerprint=job["fingerprint"]
        )
    ]
    summary = {
        "source": args.source,
        "conditions": len(rows),
        "decoder_seeds": seeds,
        "jobs": len(jobs),
        "pending": len(pending),
        "reused": len(jobs) - len(pending),
        "output": str(output_path),
        "work_root": str(work_root),
        "contract_fingerprint": contract_fingerprint,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    work_root.mkdir(parents=True, exist_ok=True)
    runtime = {
        "processed_job_count": 0,
        "reused_job_count": len(jobs) - len(pending),
        "duration_s": 0.0,
        "peak_gpu_memory_bytes": 0,
        "determinism_max_abs_diff": None,
    }
    if pending:
        code_root = project_root / registry["code_root"] / model_spec["code"]["directory"]
        activate_ltx_code(code_root)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        print("[LTX round-trip] Loading frozen VAE encoder and decoder", flush=True)
        round_tripper = LTXVAERoundTripper(checkpoint=checkpoint, device=args.device)
        determinism_checked = False
        try:
            pending_keys = {
                (str(job["row"]["condition_id"]), int(job["decoder_seed"]))
                for job in pending
            }
            processed = 0
            for row in rows:
                condition_id = str(row["condition_id"])
                row_jobs = [
                    job
                    for job in jobs
                    if str(job["row"]["condition_id"]) == condition_id
                    and (condition_id, int(job["decoder_seed"])) in pending_keys
                ]
                if not row_jobs:
                    continue
                if args.source == "gt":
                    raw = decode_frame_directory(
                        Path(row["rgb_dirs"][query["input"]["camera"]]),
                        source_frames=int(latent_config["source_total_frames"]),
                        fps=int(query["input"]["source_fps"]),
                    )
                    video = raw_rgb_video_tensor(
                        raw,
                        source_frames=int(latent_config["source_total_frames"]),
                        height=int(preprocessing["source_height"]),
                        width=int(preprocessing["source_width"]),
                        repeat_tail_frames=(
                            int(latent_config["internal_total_frames"])
                            - int(latent_config["source_total_frames"])
                        ),
                        output_height=int(latent_config["height"]),
                        output_width=int(latent_config["width"]),
                        preprocess_device=args.device,
                    )
                    del raw
                    source_latent = round_tripper.encode(video)
                    del video
                    cached_native = real_cache["features"][real_index[condition_id]]
                    source_feature = _future_feature(
                        source_latent,
                        prefix_latent_frames=int(
                            latent_config["prefix_latent_frames"]
                        ),
                    )
                    source_diff = float(
                        (source_feature.float() - cached_native.float()).abs().max()
                    )
                    del source_feature, cached_native
                    if source_diff > 0.002:
                        raise RuntimeError(
                            f"GT source re-encode drift for {condition_id}: {source_diff}"
                        )
                for job in row_jobs:
                    if args.source == "generated":
                        latent_path = Path(job["rollout_record"]["latent_path"])
                        latent_payload = torch.load(
                            latent_path, map_location="cpu", weights_only=True
                        )
                        source_latent = latent_payload["latent"]
                        cached_native = rollout_cache["features"][
                            rollout_index[
                                (condition_id, int(job["decoder_seed"]))
                            ]
                        ]
                        source_feature = _future_feature(
                            source_latent,
                            prefix_latent_frames=int(
                                latent_config["prefix_latent_frames"]
                            ),
                        )
                        source_diff = float(
                            (source_feature.float() - cached_native.float())
                            .abs()
                            .max()
                        )
                        del source_feature, cached_native, latent_payload
                        if source_diff > 0.002:
                            raise RuntimeError(
                                f"Generated source cache drift for {condition_id}, "
                                f"seed {job['decoder_seed']}: {source_diff}"
                            )
                    reencoded, pixel_audit = round_tripper.roundtrip(
                        source_latent,
                        decoder_seed=int(job["decoder_seed"]),
                        expected_frames=int(latent_config["internal_total_frames"]),
                    )
                    feature = _future_feature(
                        reencoded,
                        prefix_latent_frames=int(
                            latent_config["prefix_latent_frames"]
                        ),
                    )
                    determinism_diff = None
                    if args.determinism_check and not determinism_checked:
                        repeated, _ = round_tripper.roundtrip(
                            source_latent,
                            decoder_seed=int(job["decoder_seed"]),
                            expected_frames=int(
                                latent_config["internal_total_frames"]
                            ),
                        )
                        repeated_feature = _future_feature(
                            repeated,
                            prefix_latent_frames=int(
                                latent_config["prefix_latent_frames"]
                            ),
                        )
                        determinism_diff = float(
                            (feature.float() - repeated_feature.float()).abs().max()
                        )
                        if determinism_diff > 0.002:
                            raise RuntimeError(
                                "Round-trip is not deterministic within tolerance: "
                                f"{determinism_diff}"
                            )
                        runtime["determinism_max_abs_diff"] = determinism_diff
                        determinism_checked = True
                        del repeated, repeated_feature
                    record = {
                        "condition_id": condition_id,
                        "run_id": str(row["run_id"]),
                        "split": str(row["splits"]["main"]),
                        "safety_outcome": str(row["safety_outcome"]),
                        "source": args.source,
                        "decoder_seed": int(job["decoder_seed"]),
                        "latent_path": (
                            str(job["rollout_record"]["latent_path"])
                            if job["rollout_record"] is not None
                            else None
                        ),
                    }
                    write_torch_atomic(
                        job["part_path"],
                        {
                            "fingerprint": job["fingerprint"],
                            "record": record,
                            "feature": feature,
                            "audit": {
                                **pixel_audit,
                                "source_native_cache_max_abs_diff": source_diff,
                                "determinism_max_abs_diff": determinism_diff,
                                "roundtrip_feature_mean": float(feature.float().mean()),
                                "roundtrip_feature_std": float(feature.float().std()),
                            },
                        },
                    )
                    processed += 1
                    del reencoded, feature
                    if args.source == "generated":
                        del source_latent
                    if processed % 10 == 0 or processed == len(pending):
                        print(
                            f"[LTX round-trip] {processed}/{len(pending)} pending, "
                            f"{time.monotonic() - started:.1f}s",
                            flush=True,
                        )
                if args.source == "gt":
                    del source_latent
            runtime["processed_job_count"] = processed
        finally:
            round_tripper.close()
        runtime["duration_s"] = time.monotonic() - started
        runtime["peak_gpu_memory_bytes"] = (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else 0
        )

    features = []
    records = []
    audits = []
    for job in jobs:
        payload = torch.load(
            job["part_path"], map_location="cpu", weights_only=True
        )
        if payload.get("fingerprint") != job["fingerprint"]:
            raise RuntimeError(f"Part contract changed: {job['part_path']}")
        features.append(payload["feature"])
        records.append(payload["record"])
        audits.append(payload["audit"])
    feature_tensor = torch.stack(features)
    if tuple(feature_tensor.shape[1:]) != EXPECTED_FEATURE_SHAPE:
        raise RuntimeError("Consolidated round-trip feature shape differs")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract | {"fingerprint": contract_fingerprint},
        "records": records,
        "features": feature_tensor,
        "audits": audits,
        "runtime": runtime,
    }
    write_torch_atomic(output_path, payload)
    write_json(
        output_path.with_suffix(".json"),
        {key: value for key, value in payload.items() if key != "features"}
        | {"feature_shape": list(feature_tensor.shape)},
    )
    print(f"[LTX round-trip] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
