from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from .io_utils import load_json, write_json


OUTCOMES = ("observed_tip", "confirmed_recovered", "right_censored")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_examples(
    rows: list[dict[str, Any]], *, prefix_end_time_s: float
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for outcome in OUTCOMES:
        candidates = sorted(
            (
                row
                for row in rows
                if row.get("safety_outcome") == outcome
                and (
                    row.get("failure_time_s") is None
                    or float(row["failure_time_s"]) > prefix_end_time_s
                )
            ),
            key=lambda row: str(row["run_id"]),
        )
        if candidates:
            selected.append(candidates[0])
    return selected


def encode_prefix_video(
    frame_directory: Path, output_path: Path, *, frame_count: int, fps: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(frame_directory / "frame_%06d.png"),
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )


def prepare_smoke_inputs(
    manifest: Path,
    output_directory: Path,
    *,
    camera: str = "main_fixed_northwest",
    prefix_frame_count: int = 65,
    prefix_fps: int = 24,
) -> dict[str, Any]:
    prefix_end_time_s = prefix_frame_count / prefix_fps
    rows = _read_jsonl(manifest)
    examples: list[dict[str, Any]] = []
    for row in _select_examples(rows, prefix_end_time_s=prefix_end_time_s):
        outcome = str(row["safety_outcome"])
        frame_directory = Path(row["rgb_dirs"][camera])
        output_path = output_directory / f"{outcome}.mp4"
        encode_prefix_video(
            frame_directory,
            output_path,
            frame_count=prefix_frame_count,
            fps=prefix_fps,
        )
        examples.append(
            {
                "run_id": row["run_id"],
                "safety_outcome": outcome,
                "binary_tip_label": row.get("binary_tip_label"),
                "failure_time_s": row.get("failure_time_s"),
                "source_frame_directory": str(frame_directory),
                "prefix_video": str(output_path.resolve()),
                "prefix_frame_count": prefix_frame_count,
                "prefix_fps": prefix_fps,
                "prefix_end_time_s": prefix_end_time_s,
            }
        )
    payload = {
        "schema_version": 1,
        "source_manifest": str(manifest.resolve()),
        "camera": camera,
        "examples": examples,
    }
    write_json(output_directory / "smoke_inputs.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deterministic CCTV prefix videos.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/current/manifests/canonical_runs.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/smoke_inputs")
    )
    parser.add_argument("--camera", default="main_fixed_northwest")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("configs/comparison_v2.json"),
        help="Shared comparison contract that defines prefix frames and FPS.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    comparison = load_json(args.comparison)
    observation = comparison["observation"]
    payload = prepare_smoke_inputs(
        args.manifest.resolve(),
        args.output_dir.resolve(),
        camera=args.camera,
        prefix_frame_count=int(observation["source_frame_count"]),
        prefix_fps=int(observation["source_fps"]),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
