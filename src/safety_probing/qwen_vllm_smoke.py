from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters.qwen_vllm import run_vllm_inference, server_ready
from .io_utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Qwen vLLM smoke server.")
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
    parser.add_argument("--outcome", default="confirmed_recovered")
    args = parser.parse_args()

    registry = load_json(args.registry)
    runtime = registry["models"]["qwen_3_6"]["runtime"]
    experiment = load_json(args.experiment)
    smoke = load_json(args.smoke_manifest)
    server_url = os.environ.get(
        str(runtime.get("server_url_env", "")), str(runtime["server_url"])
    )
    if not server_ready(server_url):
        raise RuntimeError(
            f"Qwen vLLM server is not ready at {server_url}. Start "
            "scripts/start_qwen_vllm.sh on the DGX Spark first."
        )
    example = next(
        row for row in smoke["examples"] if row["safety_outcome"] == args.outcome
    )
    result = run_vllm_inference(
        server_url=server_url,
        served_model_name=runtime["served_model_name"],
        video_path=example["prefix_video"],
        prompt=experiment["prompt"],
        output_schema=experiment["output_schema"],
        video_num_frames=int(runtime["video_num_frames"]),
        max_total_video_pixels=int(
            experiment["inference"]["max_total_video_pixels"]
        ),
        max_new_tokens=int(runtime["max_new_tokens"]),
        media_transport=str(
            experiment["inference"].get("media_transport", "base64_data_uri")
        ),
    )
    print(json.dumps({"example": example, "prediction": result}, indent=2))


if __name__ == "__main__":
    main()
