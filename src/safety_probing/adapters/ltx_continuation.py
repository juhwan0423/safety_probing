from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Callable


TEMPORAL_COMPRESSION = 8


@dataclass(frozen=True)
class PrefixGeometry:
    prefix_frames: int
    source_total_frames: int
    internal_total_frames: int
    generated_future_frames: int
    evaluated_future_frames: int
    discarded_tail_frames: int


def validate_prefix_geometry(
    *, prefix_frames: int, source_total_frames: int, internal_total_frames: int
) -> PrefixGeometry:
    for name, value in (
        ("prefix_frames", prefix_frames),
        ("source_total_frames", source_total_frames),
        ("internal_total_frames", internal_total_frames),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if (prefix_frames - 1) % TEMPORAL_COMPRESSION:
        raise ValueError("LTX prefix_frames must satisfy (frames - 1) % 8 == 0")
    if (internal_total_frames - 1) % TEMPORAL_COMPRESSION:
        raise ValueError(
            "LTX internal_total_frames must satisfy (frames - 1) % 8 == 0"
        )
    if prefix_frames >= source_total_frames:
        raise ValueError("The prefix must leave at least one future source frame")
    if internal_total_frames < source_total_frames:
        raise ValueError("The internal sequence cannot be shorter than the source")

    generated = internal_total_frames - prefix_frames
    evaluated = source_total_frames - prefix_frames
    return PrefixGeometry(
        prefix_frames=prefix_frames,
        source_total_frames=source_total_frames,
        internal_total_frames=internal_total_frames,
        generated_future_frames=generated,
        evaluated_future_frames=evaluated,
        discarded_tail_frames=generated - evaluated,
    )


def activate_ltx_code(code_root: str | Path) -> Path:
    root = Path(code_root).expanduser().resolve()
    package_roots = [
        root / "packages" / "ltx-core" / "src",
        root / "packages" / "ltx-pipelines" / "src",
    ]
    missing = [path for path in package_roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Incomplete LTX checkout: {missing[0]}")
    for path in reversed(package_roots):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


class _SequentialPrefixConditioner:
    def __init__(
        self,
        delegate: Callable[[Callable[..., Any]], Any],
        *,
        prefix_video: Path,
        prefix_frames: int,
        total_frames: int,
        fps: float,
        stage_shapes: list[tuple[int, int]],
        device: Any,
        dtype: Any,
    ) -> None:
        self._delegate = delegate
        self._prefix_video = prefix_video
        self._prefix_frames = prefix_frames
        self._total_frames = total_frames
        self._fps = fps
        self._stage_shapes = stage_shapes
        self._device = device
        self._dtype = dtype
        self._call_index = 0

    def __call__(self, _unused_image_builder: Callable[..., Any]) -> list[Any]:
        if self._call_index >= len(self._stage_shapes):
            raise RuntimeError("LTX requested more conditioning stages than configured")
        width, height = self._stage_shapes[self._call_index]
        self._call_index += 1

        def build(video_encoder: Any) -> list[Any]:
            from ltx_core.conditioning import VideoConditionByLatentIndex
            from ltx_core.types import VideoLatentShape, VideoPixelShape
            from ltx_pipelines.utils.helpers import video_latent_from_file

            total_shape = VideoPixelShape(
                batch=1,
                frames=self._total_frames,
                height=height,
                width=width,
                fps=self._fps,
            )
            prefix_shape = total_shape._replace(frames=self._prefix_frames)
            latent = video_latent_from_file(
                video_encoder=video_encoder,
                file_path=str(self._prefix_video),
                output_shape=total_shape,
                device=self._device,
                dtype=self._dtype,
                max_duration=self._prefix_frames / self._fps,
            )
            prefix_latent_frames = VideoLatentShape.from_pixel_shape(prefix_shape).frames
            latent = latent[:, :, :prefix_latent_frames]
            return [
                VideoConditionByLatentIndex(
                    latent=latent,
                    strength=1.0,
                    latent_idx=0,
                )
            ]

        return self._delegate(build)


class _CapturingVideoDecoder:
    def __init__(self, delegate: Callable[..., Any]) -> None:
        self._delegate = delegate
        self.latent: Any | None = None

    def __call__(self, latent: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        self.latent = latent.detach().to(device="cpu", dtype=torch.float16)
        return self._delegate(latent, *args, **kwargs)


def run_continuation(
    *,
    code_root: str | Path,
    checkpoint: str | Path,
    text_encoder: str | Path,
    spatial_upscaler: str | Path,
    prefix_video: str | Path,
    output_path: str | Path,
    latent_output_path: str | Path | None = None,
    prompt: str,
    seed: int,
    width: int = 576,
    height: int = 320,
    fps: float = 24.0,
    prefix_frames: int = 57,
    source_total_frames: int = 120,
    internal_total_frames: int = 121,
    quantization: str = "fp8-scaled-mm",
    offload: str = "none",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    geometry = validate_prefix_geometry(
        prefix_frames=prefix_frames,
        source_total_frames=source_total_frames,
        internal_total_frames=internal_total_frames,
    )
    if width % 64 or height % 64:
        raise ValueError("The LTX two-stage output width and height must be divisible by 64")

    report("Importing the official LTX runtime")
    activate_ltx_code(code_root)
    import torch
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode

    report("LTX runtime import complete")

    checkpoint = Path(checkpoint).expanduser().resolve()
    prefix_video = Path(prefix_video).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    for path in (
        checkpoint,
        Path(text_encoder).expanduser().resolve(),
        Path(spatial_upscaler).expanduser().resolve(),
        prefix_video,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report(
        f"Loading LTX-2.3 pipeline ({quantization}, offload={offload}); "
        "large checkpoint reads can be quiet for a while"
    )
    quantization_policy = QuantizationKind(quantization).to_policy(
        checkpoint_path=str(checkpoint)
    )
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=str(checkpoint),
        gemma_root=str(Path(text_encoder).expanduser().resolve()),
        spatial_upsampler_path=str(Path(spatial_upscaler).expanduser().resolve()),
        loras=[],
        quantization=quantization_policy,
        offload_mode=OffloadMode(offload),
    )
    report(f"Pipeline loaded on {pipeline.device} with dtype={pipeline.dtype}")
    original_conditioner = pipeline.image_conditioner
    original_decoder = pipeline.video_decoder
    capturing_decoder = _CapturingVideoDecoder(original_decoder)
    pipeline.image_conditioner = _SequentialPrefixConditioner(
        original_conditioner,
        prefix_video=prefix_video,
        prefix_frames=prefix_frames,
        total_frames=internal_total_frames,
        fps=fps,
        stage_shapes=[(width // 2, height // 2), (width, height)],
        device=pipeline.device,
        dtype=pipeline.dtype,
    )
    pipeline.video_decoder = capturing_decoder
    tiling_config = TilingConfig.default()
    torch.cuda.reset_peak_memory_stats()
    try:
        report(
            f"Generating {internal_total_frames} frames at {width}x{height}, "
            f"{fps:g} FPS, seed={seed}"
        )
        with torch.inference_mode():
            video, _audio = pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=internal_total_frames,
                frame_rate=fps,
                images=[],
                tiling_config=tiling_config,
                enhance_prompt=False,
            )
            report("Generation complete; encoding MP4")
            encode_video(
                video=video,
                fps=int(fps),
                audio=None,
                output_path=str(output_path),
                video_chunks_number=get_video_chunks_number(
                    internal_total_frames, tiling_config
                ),
            )
            report(f"Video written to {output_path}")
            if latent_output_path is not None:
                if capturing_decoder.latent is None:
                    raise RuntimeError("LTX decoder did not receive a final video latent")
                latent_output_path = Path(latent_output_path).expanduser().resolve()
                latent_output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "latent": capturing_decoder.latent,
                        "prefix_latent_frames": (prefix_frames - 1)
                        // TEMPORAL_COMPRESSION
                        + 1,
                        "geometry": asdict(geometry),
                    },
                    latent_output_path,
                )
                report(f"Latent written to {latent_output_path}")
    finally:
        pipeline.image_conditioner = original_conditioner
        pipeline.video_decoder = original_decoder

    report("LTX continuation smoke complete")

    return {
        "output_path": str(output_path),
        "latent_output_path": (
            str(Path(latent_output_path).expanduser().resolve())
            if latent_output_path is not None
            else None
        ),
        "latent_shape": (
            list(capturing_decoder.latent.shape)
            if capturing_decoder.latent is not None
            else None
        ),
        "seed": seed,
        "width": width,
        "height": height,
        "fps": fps,
        "geometry": asdict(geometry),
        "quantization": quantization,
        "offload": offload,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
