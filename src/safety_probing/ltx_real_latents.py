from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from .adapters.ltx_continuation import activate_ltx_code
from .io_utils import load_json, write_json, write_torch_atomic
from .vjepa_features import _read_jsonl, _sha256, select_canonical_rows


def raw_rgb_video_tensor(
    raw: bytes,
    *,
    source_frames: int,
    height: int,
    width: int,
    repeat_tail_frames: int,
    output_height: int | None = None,
    output_width: int | None = None,
    preprocess_device: str = "cpu",
    chunk_size: int = 16,
):
    import numpy as np
    import torch

    if source_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("RGB video dimensions must be positive")
    if repeat_tail_frames < 0:
        raise ValueError("repeat_tail_frames cannot be negative")
    expected = source_frames * height * width * 3
    if len(raw) != expected:
        raise RuntimeError(
            f"ffmpeg returned {len(raw)} RGB bytes; expected {expected}"
        )
    array = np.frombuffer(raw, dtype=np.uint8).reshape(
        source_frames, height, width, 3
    )
    tensor = torch.from_numpy(array.copy())
    if repeat_tail_frames:
        tensor = torch.cat(
            [tensor, tensor[-1:].expand(repeat_tail_frames, -1, -1, -1)],
            dim=0,
        )
    output_height = int(output_height or height)
    output_width = int(output_width or width)
    scale = max(output_height / height, output_width / width)
    resized_height = math.ceil(height * scale)
    resized_width = math.ceil(width * scale)
    crop_top = (resized_height - output_height) // 2
    crop_left = (resized_width - output_width) // 2
    pieces = []
    for start in range(0, len(tensor), chunk_size):
        piece = tensor[start : start + chunk_size].permute(0, 3, 1, 2)
        piece = piece.to(device=preprocess_device, dtype=torch.float32)
        if (resized_height, resized_width) != (height, width):
            piece = torch.nn.functional.interpolate(
                piece,
                size=(resized_height, resized_width),
                mode="bilinear",
                align_corners=False,
            )
        piece = piece[
            :,
            :,
            crop_top : crop_top + output_height,
            crop_left : crop_left + output_width,
        ]
        piece = piece.div_(127.5).sub_(1.0).to(dtype=torch.bfloat16, device="cpu")
        pieces.append(piece.permute(1, 0, 2, 3).unsqueeze(0))
    return torch.cat(pieces, dim=2)


def decode_frame_directory(
    frame_directory: Path,
    *,
    source_frames: int,
    fps: int,
) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "4",
        "-framerate",
        str(fps),
        "-start_number",
        "0",
        "-i",
        str(frame_directory / "frame_%06d.png"),
        "-frames:v",
        str(source_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed for {frame_directory}: {message}")
    return completed.stdout


def load_frame_directory(
    frame_directory: Path,
    *,
    source_frames: int,
    fps: int,
    height: int,
    width: int,
    repeat_tail_frames: int,
    source_height: int,
    source_width: int,
    preprocess_device: str,
):
    raw = decode_frame_directory(
        frame_directory,
        source_frames=source_frames,
        fps=fps,
    )
    return raw_rgb_video_tensor(
        raw,
        source_frames=source_frames,
        height=source_height,
        width=source_width,
        repeat_tail_frames=repeat_tail_frames,
        output_height=height,
        output_width=width,
        preprocess_device=preprocess_device,
    )


def _part_complete(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload.get("fingerprint") == fingerprint and "feature" in payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode real simulator videos with the frozen LTX video VAE."
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/ltx_safety_query.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/current/manifests/canonical_runs.jsonl"),
    )
    parser.add_argument("--shard", default="part_01")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
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
    latent_config = experiment["real_latent"]
    input_config = experiment["input"]
    rows = select_canonical_rows(_read_jsonl(manifest_path), shard=args.shard)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    if not rows:
        raise RuntimeError("No canonical rows selected for LTX real latents")

    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "artifacts"
            / "features"
            / f"ltx_2_3_{args.shard}_real_future.pt"
        ).resolve()
    )
    work_root = (
        args.work_root.resolve()
        if args.work_root is not None
        else output_path.with_name(f"{output_path.stem}_parts")
    )
    repeat_tail_frames = int(latent_config["internal_total_frames"]) - int(
        latent_config["source_total_frames"]
    )
    from PIL import Image

    first_frame = (
        Path(rows[0]["rgb_dirs"][input_config["camera"]]) / "frame_000000.png"
    )
    with Image.open(first_frame) as image:
        source_width, source_height = image.size
    preprocessing = {
        "camera": input_config["camera"],
        "source_frames": latent_config["source_total_frames"],
        "internal_frames": latent_config["internal_total_frames"],
        "tail_padding": latent_config["tail_padding"],
        "fps": input_config["source_fps"],
        "width": latent_config["width"],
        "height": latent_config["height"],
        "source_width": source_width,
        "source_height": source_height,
        "decode": "ffmpeg_rgb24_at_source_resolution",
        "resize": "official_torch_bilinear_aspect_fill_center_crop",
        "pixel_range": "rgb_uint8_to_minus1_plus1",
        "vae_tiling": "official_default",
        "future_slice": [
            latent_config["prefix_latent_frames"],
            latent_config["latent_total_frames"],
        ],
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
    fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    print(
        json.dumps(
            {
                **contract,
                "condition_ids": {
                    "count": len(rows),
                    "first": rows[0]["condition_id"],
                    "last": rows[-1]["condition_id"],
                },
                "output": str(output_path),
                "work_root": str(work_root),
                "preprocessing": preprocessing,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    work_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for row in rows:
        part_path = work_root / f"{row['condition_id']}.pt"
        if args.overwrite or not _part_complete(part_path, fingerprint):
            pending.append((row, part_path))

    runtime: dict[str, Any] = {
        "encoded_run_count": 0,
        "reused_run_count": len(rows) - len(pending),
        "duration_s": 0.0,
        "peak_gpu_memory_bytes": 0,
    }
    if pending:
        import torch

        code_root = project_root / registry["code_root"] / model_spec["code"]["directory"]
        activate_ltx_code(code_root)
        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils.blocks import ImageConditioner

        checkpoint = project_root / model_spec["runtime"]["checkpoint"]
        conditioner = ImageConditioner(
            str(checkpoint.resolve()),
            torch.bfloat16,
            torch.device(args.device),
        )
        started = time.monotonic()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        def encode_pending(encoder):
            def submit_decode(pool, item):
                row, _ = item
                return pool.submit(
                    decode_frame_directory,
                    Path(row["rgb_dirs"][input_config["camera"]]),
                    source_frames=int(latent_config["source_total_frames"]),
                    fps=int(input_config["source_fps"]),
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                decode_future = submit_decode(pool, pending[0])
                for index, (row, part_path) in enumerate(pending, start=1):
                    raw = decode_future.result()
                    if index < len(pending):
                        decode_future = submit_decode(pool, pending[index])
                    video = raw_rgb_video_tensor(
                        raw,
                        source_frames=int(latent_config["source_total_frames"]),
                        height=source_height,
                        width=source_width,
                        repeat_tail_frames=repeat_tail_frames,
                        output_height=int(latent_config["height"]),
                        output_width=int(latent_config["width"]),
                        preprocess_device=args.device,
                    )
                    del raw
                    with torch.inference_mode():
                        latent = encoder.tiled_encode(video, TilingConfig.default())
                    expected = (
                        1,
                        int(latent_config["latent_channels"]),
                        int(latent_config["latent_total_frames"]),
                        int(latent_config["latent_height"]),
                        int(latent_config["latent_width"]),
                    )
                    if tuple(latent.shape) != expected:
                        raise RuntimeError(
                            f"Expected LTX latent {expected}, got {tuple(latent.shape)}"
                        )
                    start = int(latent_config["prefix_latent_frames"])
                    feature = latent[0, :, start:].permute(1, 2, 3, 0)
                    feature = feature.to(device="cpu", dtype=torch.float16)
                    if not bool(torch.isfinite(feature).all()):
                        raise RuntimeError("LTX VAE produced non-finite real features")
                    write_torch_atomic(
                        part_path,
                        {
                            "fingerprint": fingerprint,
                            "condition_id": str(row["condition_id"]),
                            "feature": feature,
                        },
                    )
                    del video, latent, feature
                    if index % 10 == 0 or index == len(pending):
                        print(
                            f"[LTX real latent] {index}/{len(pending)} pending, "
                            f"{time.monotonic() - started:.1f}s",
                            flush=True,
                        )

        conditioner(encode_pending)
        runtime.update(
            {
                "encoded_run_count": len(pending),
                "duration_s": time.monotonic() - started,
                "peak_gpu_memory_bytes": (
                    torch.cuda.max_memory_allocated()
                    if args.device.startswith("cuda")
                    else 0
                ),
            }
        )

    import torch

    features = []
    for row in rows:
        payload = torch.load(
            work_root / f"{row['condition_id']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("fingerprint") != fingerprint:
            raise RuntimeError("LTX real latent part contract changed")
        features.append(payload["feature"])
    feature_tensor = torch.stack(features)
    expected_tail = (
        int(latent_config["future_latent_frames"]),
        int(latent_config["latent_height"]),
        int(latent_config["latent_width"]),
        int(latent_config["latent_channels"]),
    )
    if tuple(feature_tensor.shape[1:]) != expected_tail:
        raise RuntimeError("Consolidated LTX real feature shape differs")
    records = [
        {
            "condition_id": str(row["condition_id"]),
            "run_id": str(row["run_id"]),
            "split": str(row["splits"]["main"]),
            "safety_outcome": str(row["safety_outcome"]),
        }
        for row in rows
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "preprocessing": preprocessing,
        "records": records,
        "features": feature_tensor,
        "runtime": runtime,
    }
    write_torch_atomic(output_path, payload)
    write_json(
        output_path.with_suffix(".json"),
        {
            key: value for key, value in payload.items() if key != "features"
        }
        | {"feature_shape": list(feature_tensor.shape)},
    )
    print(f"[LTX real latent] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
