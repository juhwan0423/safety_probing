from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


VALID_OUTCOMES = {"observed_tip", "confirmed_recovered"}


def build_messages(video_path: str | Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(Path(video_path).resolve())},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Qwen output does not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen output JSON must be an object")
    return value


def parse_prediction(text: str) -> dict[str, Any]:
    value = _json_object(text)
    outcome = str(value.get("predicted_outcome"))
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"Invalid predicted_outcome: {outcome}")
    probability = float(value["tip_probability"])
    if not 0.0 <= probability <= 1.0:
        raise ValueError("tip_probability must be in [0, 1]")
    return {
        "predicted_outcome": outcome,
        "tip_probability": probability,
        "raw_json": value,
    }


def prepare_local_inputs(
    *,
    model_path: str | Path,
    video_path: str | Path,
    prompt: str,
    video_num_frames: int = 20,
    max_total_video_pixels: int = 2_293_760,
):
    try:
        from transformers import AutoProcessor
        from transformers.video_utils import load_video
    except ImportError as exc:
        raise RuntimeError("Install the inference-core dependencies first") from exc

    processor = AutoProcessor.from_pretrained(
        Path(model_path).resolve(), local_files_only=True
    )
    processor.video_processor.fps = None

    def fetch_videos_pyav(video_or_videos: Any, sample_indices_fn: Any = None):
        if isinstance(video_or_videos, list):
            return list(
                zip(
                    *[
                        fetch_videos_pyav(item, sample_indices_fn=sample_indices_fn)
                        for item in video_or_videos
                    ]
                )
            )
        return load_video(
            video_or_videos,
            backend="pyav",
            sample_indices_fn=sample_indices_fn,
        )

    # Transformers 5.13 otherwise falls back to a removed torchvision.io API.
    processor.video_processor.fetch_videos = fetch_videos_pyav
    inputs = processor.apply_chat_template(
        build_messages(video_path, prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs={
            "num_frames": video_num_frames,
            "size": {
                "shortest_edge": 4096,
                "longest_edge": max_total_video_pixels,
            },
        },
    )
    return processor, inputs


def inspect_local_inputs(**kwargs: Any) -> dict[str, Any]:
    processor, inputs = prepare_local_inputs(**kwargs)
    video_grid = inputs["video_grid_thw"]
    return {
        "input_names": sorted(inputs),
        "tensor_shapes": {
            key: list(value.shape)
            for key, value in inputs.items()
            if hasattr(value, "shape")
        },
        "text_and_vision_token_count": int(inputs["input_ids"].shape[1]),
        "video_grid_thw": video_grid.tolist(),
        "sampled_video_frame_count": int(
            video_grid[0, 0] * processor.video_processor.temporal_patch_size
        ),
    }


def run_local_inference(
    *,
    model_path: str | Path,
    video_path: str | Path,
    prompt: str,
    max_new_tokens: int = 160,
    gpu_memory_limit: str = "24GiB",
    video_num_frames: int = 20,
    max_total_video_pixels: int = 2_293_760,
    offload_directory: str | Path = "artifacts/offload/qwen_3_6",
) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise RuntimeError("Install the inference-core dependencies first") from exc

    model_path = Path(model_path).resolve()
    offload_directory = Path(offload_directory).resolve()
    offload_directory.mkdir(parents=True, exist_ok=True)
    processor, inputs = prepare_local_inputs(
        model_path=model_path,
        video_path=video_path,
        prompt=prompt,
        video_num_frames=video_num_frames,
        max_total_video_pixels=max_total_video_pixels,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        dtype="auto",
        device_map="auto",
        max_memory={0: gpu_memory_limit, "cpu": "40GiB"},
        offload_folder=offload_directory,
        attn_implementation="sdpa",
    ).eval()
    input_device = model.get_input_embeddings().weight.device
    inputs = {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_length = int(inputs["input_ids"].shape[1])
    includes_prompt = bool(
        generated.shape[1] >= prompt_length
        and torch.equal(
            generated[:, :prompt_length].to(inputs["input_ids"].device),
            inputs["input_ids"],
        )
    )
    new_tokens = generated[:, prompt_length:] if includes_prompt else generated
    output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    result: dict[str, Any] = {
        "raw_text": output_text,
        "generated_token_count": int(new_tokens.shape[1]),
        "generation_included_prompt": includes_prompt,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    try:
        result.update(parse_prediction(output_text))
        result["parse_valid"] = True
    except (KeyError, TypeError, ValueError) as exc:
        result["parse_valid"] = False
        result["parse_error"] = str(exc)
    return result
