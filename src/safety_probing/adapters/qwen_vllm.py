from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .qwen_direct import parse_prediction


def encode_video_url(
    video_path: str | Path, media_transport: str = "base64_data_uri"
) -> str:
    path = Path(video_path).expanduser().resolve()
    if media_transport == "file_uri":
        return path.as_uri()
    if media_transport != "base64_data_uri":
        raise ValueError(f"Unsupported Qwen media transport: {media_transport}")
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach vLLM server at {url}: {exc}") from exc


def server_ready(server_url: str, timeout_s: float = 5.0) -> bool:
    try:
        with urlopen(f"{server_url.rstrip('/')}/health", timeout=timeout_s) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def run_vllm_inference(
    *,
    server_url: str,
    served_model_name: str,
    video_path: str | Path,
    prompt: str,
    output_schema: dict[str, Any],
    video_num_frames: int = 20,
    max_total_video_pixels: int = 2_293_760,
    max_new_tokens: int = 160,
    media_transport: str = "base64_data_uri",
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    video_uri = encode_video_url(video_path, media_transport)
    payload = {
        "model": served_model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "mm_processor_kwargs": {
            "num_frames": video_num_frames,
            "do_sample_frames": True,
            "size": {
                "shortest_edge": 4096,
                "longest_edge": max_total_video_pixels,
            },
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "forklift_safety_prediction",
                "schema": output_schema,
                "strict": True,
            },
        },
    }
    response = _post_json(
        f"{server_url.rstrip('/')}/v1/chat/completions", payload, timeout_s
    )
    message = response["choices"][0]["message"]
    raw_text = message.get("content") or message.get("reasoning_content") or ""
    result = parse_prediction(raw_text)
    result.update(
        {
            "raw_text": raw_text,
            "response_id": response.get("id"),
            "usage": response.get("usage"),
            "backend": "vllm",
        }
    )
    return result
