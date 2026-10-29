from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .io_utils import load_json, write_json


LOCKED_PATHS = (
    "configs/local_dataset.json",
    "configs/model_registry.json",
    "configs/experiments/vjepa_safety_query.json",
    "configs/experiments/ltx_safety_query.json",
    "configs/experiments/ltx_2_3_continuation_prompt_v1.json",
    "artifacts/splits/leakage_safe_splits.json",
    "src/safety_probing/vjepa_features.py",
    "src/safety_probing/vjepa_spatial_features.py",
    "src/safety_probing/safety_query_targets.py",
    "src/safety_probing/safety_query_probe.py",
    "src/safety_probing/safety_query_controls.py",
    "src/safety_probing/ltx_batch.py",
    "src/safety_probing/ltx_real_latents.py",
    "src/safety_probing/ltx_rollout_cache.py",
    "src/safety_probing/ltx_safety_query.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_lock(root: Path) -> dict[str, Any]:
    files = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    dataset_config = load_json(root / "configs/local_dataset.json")
    vjepa = load_json(root / "configs/experiments/vjepa_safety_query.json")
    ltx = load_json(root / "configs/experiments/ltx_safety_query.json")
    status = _git(root, "status", "--porcelain")
    return {
        "schema_version": 1,
        "protocol_id": "physical-safety-query-workshop-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Architecture, targets, prompts, seeds, metrics, and split are frozen after "
            "the shard-01 exploratory analysis. Part 02/03 complete the same generation "
            "grid and are not independent confirmatory test shards."
        ),
        "paper_scope": "V-JEPA prefix Safety Query versus prompt-v1 LTX generated Safety Query",
        "label_schema": dataset_config["labeling"]["schema_version"],
        "query_seeds": {
            "vjepa": list(vjepa["training"]["seeds"]),
            "ltx": list(ltx["training"]["seeds"]),
            "ltx_rollout": list(ltx["rollout"]["seeds"]),
        },
        "source_git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "branch", "--show-current"),
            "dirty": bool(status),
            "status_sha256": (
                hashlib.sha256(status.encode("utf-8")).hexdigest()
                if status is not None
                else None
            ),
        },
        "files": files,
    }


def verify_lock(root: Path, lock: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for item in lock["files"]:
        path = root / str(item["path"])
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "matches": actual == item["sha256"],
            }
        )
    return {
        "protocol_id": lock["protocol_id"],
        "valid": all(item["matches"] for item in checks),
        "checked_file_count": len(checks),
        "mismatch_count": sum(not item["matches"] for item in checks),
        "mismatches": [item for item in checks if not item["matches"]],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify the frozen physical Safety Query protocol lock."
    )
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/protocols/physical_safety_query_workshop_v1.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    if args.command == "create":
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Protocol lock already exists: {output}")
        lock = build_lock(root)
        write_json(output, lock)
        report = verify_lock(root, lock)
        print(json.dumps(report | {"output": str(output)}, indent=2))
        return

    if not output.is_file():
        raise FileNotFoundError(output)
    report = verify_lock(root, load_json(output))
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
