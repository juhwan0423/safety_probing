from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters.vjepa import encoder_smoke_test
from .io_utils import load_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model-family load/forward smoke tests.")
    parser.add_argument("model", choices=("vjepa_2", "vjepa_2_1"))
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--smoke-manifest",
        type=Path,
        default=Path("artifacts/smoke_inputs/smoke_inputs.json"),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = args.registry.resolve().parent.parent
    registry = load_json(args.registry)
    spec = registry["models"][args.model]
    runtime = spec["runtime"]
    smoke = load_json(args.smoke_manifest)
    example = next(
        row
        for row in smoke["examples"]
        if row["safety_outcome"] == "confirmed_recovered"
    )
    payload = encoder_smoke_test(
        code_root=project_root / registry["code_root"] / "vjepa2",
        checkpoint=project_root / runtime["checkpoint"],
        version="2.1" if args.model == "vjepa_2_1" else "2",
        frame_directory=example["source_frame_directory"],
        device=args.device,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
