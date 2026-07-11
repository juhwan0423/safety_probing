from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.ltx_continuation import run_continuation, validate_prefix_geometry
from .io_utils import load_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an LTX CCTV continuation smoke test.")
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/ltx_2_3_continuation.json"),
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=Path("artifacts/smoke_inputs/smoke_inputs.json"),
    )
    parser.add_argument("--outcome", default="confirmed_recovered")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/smoke/ltx_confirmed_recovered.mp4")
    )
    parser.add_argument(
        "--latent-output",
        type=Path,
        default=Path("outputs/smoke/ltx_confirmed_recovered_latent.pt"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate temporal geometry and paths without loading the model.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    registry_path = args.registry.resolve()
    project_root = registry_path.parent.parent
    registry = load_json(registry_path)
    experiment = load_json(args.experiment)
    smoke = load_json(args.smoke_manifest)
    model = registry["models"]["ltx_2_3"]
    runtime = model["runtime"]
    example = next(
        row for row in smoke["examples"] if row["safety_outcome"] == args.outcome
    )
    input_spec = experiment["input"]
    generation = experiment["generation"]
    width, height = input_spec["resize"]
    geometry = validate_prefix_geometry(
        prefix_frames=int(input_spec["frame_count"]),
        source_total_frames=120,
        internal_total_frames=int(generation["internal_total_frame_count"]),
    )
    paths = {
        "code_root": project_root / registry["code_root"] / model["code"]["directory"],
        "checkpoint": project_root / runtime["checkpoint"],
        "text_encoder": project_root / runtime["text_encoder"],
        "spatial_upscaler": project_root / runtime["spatial_upscaler"],
        "prefix_video": Path(example["prefix_video"]),
    }
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "geometry": geometry.__dict__,
                    "paths": {
                        name: {"path": str(path), "exists": path.exists()}
                        for name, path in paths.items()
                    },
                    "resolution": [width, height],
                },
                indent=2,
            )
        )
        return

    result = run_continuation(
        **paths,
        output_path=args.output,
        latent_output_path=args.latent_output,
        prompt=generation["prompt"],
        seed=args.seed,
        width=int(width),
        height=int(height),
        fps=float(input_spec["fps"]),
        prefix_frames=int(input_spec["frame_count"]),
        source_total_frames=120,
        internal_total_frames=int(generation["internal_total_frame_count"]),
        quantization=runtime["quantization"],
        offload=runtime["offload"],
        progress=lambda message: print(f"[LTX] {message}", flush=True),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
