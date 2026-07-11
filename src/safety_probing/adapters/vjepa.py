from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "").replace("backbone.", "")] = value
    return cleaned


def load_vjepa(
    *,
    code_root: str | Path,
    checkpoint: str | Path,
    version: str,
    num_frames: int = 20,
    device: str = "cpu",
):
    """Load the pinned encoder and pretrained JEPA predictor from a local checkpoint."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("V-JEPA inference requires PyTorch") from exc

    root = Path(code_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.hub.backbones import (  # type: ignore[import-not-found]
        vjepa2_1_vit_large_384,
        vjepa2_vit_large,
    )

    if version == "2.1":
        encoder, predictor = vjepa2_1_vit_large_384(
            pretrained=False, num_frames=num_frames
        )
        encoder_key = "ema_encoder"
    elif version == "2":
        encoder, predictor = vjepa2_vit_large(
            pretrained=False, num_frames=num_frames
        )
        encoder_key = "target_encoder"
    else:
        raise ValueError(f"Unsupported V-JEPA version: {version}")

    state = torch.load(
        Path(checkpoint), map_location="cpu", mmap=True, weights_only=True
    )
    encoder.load_state_dict(
        _clean_state_dict(state[encoder_key]), strict=(version == "2.1")
    )
    predictor.load_state_dict(_clean_state_dict(state["predictor"]), strict=False)
    encoder.requires_grad_(False).eval().to(device)
    predictor.requires_grad_(False).eval().to(device)
    return encoder, predictor


def load_video_tensor(
    frame_directory: str | Path,
    *,
    frame_count: int = 20,
    source_frame_count: int = 65,
    resolution: int = 384,
):
    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as functional
    except ImportError as exc:
        raise RuntimeError("V-JEPA video preprocessing dependencies are missing") from exc

    directory = Path(frame_directory)
    if frame_count <= 1:
        indices = [0]
    else:
        indices = [
            round(index * (source_frame_count - 1) / (frame_count - 1))
            for index in range(frame_count)
        ]
    frames = []
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    resize_side = round(resolution * 256 / 224)
    for index in indices:
        with Image.open(directory / f"frame_{index:06d}.png") as image:
            frame = functional.to_tensor(image.convert("RGB"))
        frame = functional.resize(
            frame, resize_side, antialias=True
        )
        frame = functional.center_crop(frame, [resolution, resolution])
        frames.append((frame - mean) / std)
    return torch.stack(frames, dim=1).unsqueeze(0), indices


def encoder_smoke_test(
    *,
    code_root: str | Path,
    checkpoint: str | Path,
    version: str,
    frame_directory: str | Path,
    source_frame_count: int = 65,
    device: str = "cuda",
) -> dict[str, Any]:
    import torch

    resolution = 384 if version == "2.1" else 256
    video, indices = load_video_tensor(
        frame_directory,
        frame_count=20,
        source_frame_count=source_frame_count,
        resolution=resolution,
    )
    encoder, _ = load_vjepa(
        code_root=code_root,
        checkpoint=checkpoint,
        version=version,
        num_frames=20,
        device=device,
    )
    video = video.to(device)
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
    ):
        features = encoder(video)
    if isinstance(features, list):
        shapes = [list(value.shape) for value in features]
        finite = all(bool(torch.isfinite(value).all()) for value in features)
    else:
        shapes = [list(features.shape)]
        finite = bool(torch.isfinite(features).all())
    return {
        "version": version,
        "device": device,
        "input_shape": list(video.shape),
        "source_frame_indices": indices,
        "feature_shapes": shapes,
        "features_finite": finite,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        ),
    }
