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


class LTXContinuationRunner:
    def __init__(
        self,
        *,
        code_root: str | Path,
        checkpoint: str | Path,
        text_encoder: str | Path,
        spatial_upscaler: str | Path,
        quantization: str = "fp8-scaled-mm",
        offload: str = "none",
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._progress = progress
        self.quantization = quantization
        self.offload = offload
        self._report("Importing the official LTX runtime")
        activate_ltx_code(code_root)

        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_pipelines.utils.types import OffloadMode

        self._torch = torch
        self._tiling_config_type = TilingConfig
        self._get_video_chunks_number = get_video_chunks_number
        self._encode_video = encode_video
        self._report("LTX runtime import complete")

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        text_encoder_path = Path(text_encoder).expanduser().resolve()
        upscaler_path = Path(spatial_upscaler).expanduser().resolve()
        for path in (checkpoint_path, text_encoder_path, upscaler_path):
            if not path.exists():
                raise FileNotFoundError(path)

        self._report(
            f"Loading LTX-2.3 pipeline ({quantization}, offload={offload}); "
            "large checkpoint reads can be quiet for a while"
        )
        quantization_policy = QuantizationKind(quantization).to_policy(
            checkpoint_path=str(checkpoint_path)
        )
        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=str(checkpoint_path),
            gemma_root=str(text_encoder_path),
            spatial_upsampler_path=str(upscaler_path),
            loras=[],
            quantization=quantization_policy,
            offload_mode=OffloadMode(offload),
        )
        self._original_conditioner = self.pipeline.image_conditioner
        self._original_decoder = self.pipeline.video_decoder
        self._report(
            f"Pipeline loaded on {self.pipeline.device} "
            f"with dtype={self.pipeline.dtype}"
        )

    def _report(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        return path.with_name(f".{path.stem}.partial{path.suffix}")

    def run(
        self,
        *,
        prefix_video: str | Path,
        output_path: str | Path,
        latent_output_path: str | Path | None = None,
        prompt: str,
        seed: int,
        width: int = 576,
        height: int = 320,
        fps: float = 24.0,
        prefix_frames: int = 65,
        source_total_frames: int = 120,
        internal_total_frames: int = 121,
    ) -> dict[str, Any]:
        geometry = validate_prefix_geometry(
            prefix_frames=prefix_frames,
            source_total_frames=source_total_frames,
            internal_total_frames=internal_total_frames,
        )
        if width % 64 or height % 64:
            raise ValueError(
                "The LTX two-stage output width and height must be divisible by 64"
            )

        prefix_path = Path(prefix_video).expanduser().resolve()
        if not prefix_path.is_file():
            raise FileNotFoundError(prefix_path)
        final_video_path = Path(output_path).expanduser().resolve()
        final_video_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_video_path = self._temporary_path(final_video_path)
        temporary_video_path.unlink(missing_ok=True)

        final_latent_path = (
            Path(latent_output_path).expanduser().resolve()
            if latent_output_path is not None
            else None
        )
        temporary_latent_path = (
            self._temporary_path(final_latent_path)
            if final_latent_path is not None
            else None
        )
        if final_latent_path is not None:
            final_latent_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_latent_path.unlink(missing_ok=True)

        capturing_decoder = _CapturingVideoDecoder(self._original_decoder)
        self.pipeline.image_conditioner = _SequentialPrefixConditioner(
            self._original_conditioner,
            prefix_video=prefix_path,
            prefix_frames=prefix_frames,
            total_frames=internal_total_frames,
            fps=fps,
            stage_shapes=[(width // 2, height // 2), (width, height)],
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
        )
        self.pipeline.video_decoder = capturing_decoder
        tiling_config = self._tiling_config_type.default()
        self._torch.cuda.reset_peak_memory_stats()
        video = None
        try:
            self._report(
                f"Generating {internal_total_frames} frames at {width}x{height}, "
                f"{fps:g} FPS, seed={seed}"
            )
            with self._torch.inference_mode():
                video, _audio = self.pipeline(
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
                self._report("Generation complete; encoding MP4")
                self._encode_video(
                    video=video,
                    fps=int(fps),
                    audio=None,
                    output_path=str(temporary_video_path),
                    video_chunks_number=self._get_video_chunks_number(
                        internal_total_frames, tiling_config
                    ),
                )
                temporary_video_path.replace(final_video_path)
                self._report(f"Video written to {final_video_path}")
                if final_latent_path is not None:
                    if capturing_decoder.latent is None:
                        raise RuntimeError(
                            "LTX decoder did not receive a final video latent"
                        )
                    self._torch.save(
                        {
                            "latent": capturing_decoder.latent,
                            "prefix_latent_frames": (prefix_frames - 1)
                            // TEMPORAL_COMPRESSION
                            + 1,
                            "geometry": asdict(geometry),
                        },
                        temporary_latent_path,
                    )
                    temporary_latent_path.replace(final_latent_path)
                    self._report(f"Latent written to {final_latent_path}")
            peak_memory = self._torch.cuda.max_memory_allocated()
        except BaseException:
            temporary_video_path.unlink(missing_ok=True)
            if temporary_latent_path is not None:
                temporary_latent_path.unlink(missing_ok=True)
            raise
        finally:
            self.pipeline.image_conditioner = self._original_conditioner
            self.pipeline.video_decoder = self._original_decoder
            del video
            self._torch.cuda.empty_cache()

        self._report("LTX continuation complete")
        return {
            "output_path": str(final_video_path),
            "latent_output_path": (
                str(final_latent_path) if final_latent_path is not None else None
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
            "quantization": self.quantization,
            "offload": self.offload,
            "peak_gpu_memory_bytes": peak_memory,
        }


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
    prefix_frames: int = 65,
    source_total_frames: int = 120,
    internal_total_frames: int = 121,
    quantization: str = "fp8-scaled-mm",
    offload: str = "none",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    runner = LTXContinuationRunner(
        code_root=code_root,
        checkpoint=checkpoint,
        text_encoder=text_encoder,
        spatial_upscaler=spatial_upscaler,
        quantization=quantization,
        offload=offload,
        progress=progress,
    )
    return runner.run(
        prefix_video=prefix_video,
        output_path=output_path,
        latent_output_path=latent_output_path,
        prompt=prompt,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        prefix_frames=prefix_frames,
        source_total_frames=source_total_frames,
        internal_total_frames=internal_total_frames,
    )
