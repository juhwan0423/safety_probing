from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.qwen_direct import inspect_local_inputs, run_local_inference
from .io_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3.6 CCTV safety inference.")
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/qwen_3_6_direct.json"),
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=Path("artifacts/smoke_inputs/smoke_inputs.json"),
    )
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    project_root = args.registry.resolve().parent.parent
    registry = load_json(args.registry)
    runtime = registry["models"]["qwen_3_6"]["runtime"]
    experiment = load_json(args.experiment)
    smoke = load_json(args.smoke_manifest)
    example = next(
        row
        for row in smoke["examples"]
        if row["safety_outcome"] == "confirmed_recovered"
    )
    inference = experiment["inference"]
    common = {
        "model_path": project_root / runtime["model_path"],
        "video_path": example["prefix_video"],
        "prompt": experiment["prompt"],
        "video_num_frames": int(inference["video_num_frames"]),
        "max_total_video_pixels": int(inference["max_total_video_pixels"]),
    }
    if args.preprocess_only:
        result = inspect_local_inputs(**common)
        print(json.dumps({"example": example, "preprocessing": result}, indent=2))
        return
    result = run_local_inference(
        **common,
        max_new_tokens=int(runtime["max_new_tokens"]),
        gpu_memory_limit=str(runtime["gpu_memory_limit"]),
    )
    print(json.dumps({"example": example, "prediction": result}, indent=2))


if __name__ == "__main__":
    main()
