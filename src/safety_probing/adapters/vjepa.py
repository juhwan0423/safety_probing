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
    encoder.load_state_dict(_clean_state_dict(state[encoder_key]), strict=True)
    predictor.load_state_dict(_clean_state_dict(state["predictor"]), strict=True)
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
    indices = uniform_frame_indices(0, source_frame_count - 1, frame_count)
    return load_indexed_video_tensor(
        frame_directory,
        frame_indices=indices,
        resolution=resolution,
    )


def uniform_frame_indices(
    start_index: int, end_index: int, frame_count: int
) -> list[int]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if end_index < start_index:
        raise ValueError("end_index must not precede start_index")
    if frame_count == 1:
        return [start_index]

    indices = [
        round(start_index + index * (end_index - start_index) / (frame_count - 1))
        for index in range(frame_count)
    ]
    if len(set(indices)) != len(indices):
        raise ValueError("frame_count exceeds the number of distinct source frames")
    return indices


def load_indexed_video_tensor(
    frame_directory: str | Path,
    *,
    frame_indices: list[int],
    resolution: int,
):
    try:
        import torch
        from PIL import Image
        from torchvision.transforms import functional as functional
    except ImportError as exc:
        raise RuntimeError("V-JEPA video preprocessing dependencies are missing") from exc

    directory = Path(frame_directory)
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    frames = []
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    resize_side = round(resolution * 256 / 224)
    for index in frame_indices:
        frame_path = directory / f"frame_{index:06d}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(f"Missing V-JEPA source frame: {frame_path}")
        with Image.open(frame_path) as image:
            frame = functional.to_tensor(image.convert("RGB"))
        frame = functional.resize(
            frame, resize_side, antialias=True
        )
        frame = functional.center_crop(frame, [resolution, resolution])
        frames.append((frame - mean) / std)
    return torch.stack(frames, dim=1).unsqueeze(0), list(frame_indices)


def future_patch_index_ranges(
    *,
    context_frame_count: int,
    future_frame_count: int,
    resolution: int,
    patch_size: int = 16,
    tubelet_size: int = 2,
) -> tuple[range, range]:
    if context_frame_count <= 0 or future_frame_count <= 0:
        raise ValueError("context and future frame counts must be positive")
    if context_frame_count % tubelet_size or future_frame_count % tubelet_size:
        raise ValueError("context and future frame counts must align to tubelets")
    if resolution % patch_size:
        raise ValueError("resolution must be divisible by patch_size")

    spatial_token_count = (resolution // patch_size) ** 2
    context_token_count = (
        context_frame_count // tubelet_size
    ) * spatial_token_count
    future_token_count = (
        future_frame_count // tubelet_size
    ) * spatial_token_count
    return (
        range(0, context_token_count),
        range(context_token_count, context_token_count + future_token_count),
    )


class FrozenVJEPANativePredictor:
    """Run the released V-JEPA 2 masked-latent predictor without adaptation."""

    def __init__(
        self,
        *,
        code_root: str | Path,
        checkpoint: str | Path,
        version: str = "2",
        context_frame_count: int = 20,
        future_frame_count: int = 20,
        resolution: int = 256,
        device: str = "cuda",
    ) -> None:
        if version != "2":
            raise ValueError(
                "Native target-energy evaluation currently requires V-JEPA 2. "
                "The released distilled V-JEPA 2.1 ViT-L predictor emits the "
                "1664-dimensional teacher space, while its encoder emits 1024 dimensions."
            )
        self.device = device
        self.version = version
        self.context_frame_count = context_frame_count
        self.future_frame_count = future_frame_count
        self.resolution = resolution
        self.context_indices, self.future_indices = future_patch_index_ranges(
            context_frame_count=context_frame_count,
            future_frame_count=future_frame_count,
            resolution=resolution,
        )
        self.encoder, self.predictor = load_vjepa(
            code_root=code_root,
            checkpoint=checkpoint,
            version=version,
            num_frames=context_frame_count + future_frame_count,
            device=device,
        )
        if any(
            parameter.requires_grad
            for model in (self.encoder, self.predictor)
            for parameter in model.parameters()
        ):
            raise RuntimeError("Native V-JEPA foundation parameters must remain frozen")

    def predict_and_encode_targets(self, video_batch, *, target_video_batch=None):
        """Return predicted tokens and target tokens from optional candidate clips."""

        import torch
        import torch.nn.functional as functional

        if video_batch.ndim != 5:
            raise ValueError("video_batch must have shape [B,C,T,H,W]")
        expected_frames = self.context_frame_count + self.future_frame_count
        if video_batch.shape[2] != expected_frames:
            raise ValueError(
                f"Expected {expected_frames} model frames, got {video_batch.shape[2]}"
            )
        if tuple(video_batch.shape[-2:]) != (self.resolution, self.resolution):
            raise ValueError(
                f"Expected {self.resolution}x{self.resolution} input, "
                f"got {tuple(video_batch.shape[-2:])}"
            )
        if target_video_batch is None:
            target_video_batch = video_batch
        if target_video_batch.ndim != 5:
            raise ValueError("target_video_batch must have shape [B,C,T,H,W]")
        if target_video_batch.shape[1:] != video_batch.shape[1:]:
            raise ValueError("Query and target video shapes must match after batch axis")

        video_batch = video_batch.to(self.device)
        target_video_batch = target_video_batch.to(self.device)
        batch_size = int(video_batch.shape[0])
        target_batch_size = int(target_video_batch.shape[0])
        context_mask = torch.tensor(
            list(self.context_indices), dtype=torch.long, device=self.device
        ).repeat(batch_size, 1)
        future_mask = torch.tensor(
            list(self.future_indices), dtype=torch.long, device=self.device
        ).repeat(batch_size, 1)
        target_future_mask = torch.tensor(
            list(self.future_indices), dtype=torch.long, device=self.device
        ).repeat(target_batch_size, 1)

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.startswith("cuda"),
        ):
            context_latents = self.encoder(video_batch, masks=context_mask)
            predicted_future = self.predictor(
                context_latents,
                context_mask,
                future_mask,
                mask_index=0,
            )
            full_target_latents = self.encoder(target_video_batch)
            full_target_latents = functional.layer_norm(
                full_target_latents, (full_target_latents.size(-1),)
            )
            target_future = torch.gather(
                full_target_latents,
                dim=1,
                index=target_future_mask.unsqueeze(-1).expand(
                    -1, -1, full_target_latents.size(-1)
                ),
            )

        if predicted_future.shape[1:] != target_future.shape[1:]:
            raise RuntimeError(
                "Predicted and target future token shapes differ: "
                f"{tuple(predicted_future.shape)} vs {tuple(target_future.shape)}"
            )
        if not bool(torch.isfinite(predicted_future).all()):
            raise RuntimeError("V-JEPA predicted non-finite future latents")
        if not bool(torch.isfinite(target_future).all()):
            raise RuntimeError("V-JEPA encoded non-finite target latents")

        return {
            "predicted_future": predicted_future.float().cpu(),
            "target_future": target_future.float().cpu(),
            "context_mask": context_mask.cpu(),
            "future_mask": future_mask.cpu(),
            "target_future_mask": target_future_mask.cpu(),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated()
                if self.device.startswith("cuda")
                else 0
            ),
        }


def latent_candidate_energy(
    predicted_future, target_future
) -> dict[str, list[list[float]]]:
    """Score every predicted query against every actual candidate future."""

    import torch.nn.functional as functional

    if predicted_future.ndim != 3 or target_future.ndim not in (3, 4):
        raise ValueError("Targets must have shape [C,N,D] or [Q,C,N,D]")
    target_token_shape = target_future.shape[-2:]
    if predicted_future.shape[1:] != target_token_shape:
        raise ValueError("Predicted and target token shapes must match")
    if target_future.ndim == 4 and target_future.shape[0] != predicted_future.shape[0]:
        raise ValueError("Query-conditioned targets must match the query batch size")

    cosine_rows: list[list[float]] = []
    smooth_l1_rows: list[list[float]] = []
    for query_index, predicted in enumerate(predicted_future):
        cosine_row: list[float] = []
        smooth_l1_row: list[float] = []
        candidate_targets = (
            target_future
            if target_future.ndim == 3
            else target_future[query_index]
        )
        for target in candidate_targets:
            cosine_row.append(
                float(
                    1.0
                    - functional.cosine_similarity(
                        predicted, target, dim=-1
                    ).mean()
                )
            )
            smooth_l1_row.append(
                float(functional.smooth_l1_loss(predicted, target))
            )
        cosine_rows.append(cosine_row)
        smooth_l1_rows.append(smooth_l1_row)
    return {
        "mean_token_cosine_distance": cosine_rows,
        "smooth_l1": smooth_l1_rows,
    }


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
