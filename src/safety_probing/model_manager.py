from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from .io_utils import load_json, write_json


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _clone_code(code_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    destination = code_root / str(spec["directory"])
    revision = str(spec["revision"])
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                str(spec["url"]),
                str(destination),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        check=True,
    )
    actual = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"path": str(destination), "revision": actual}


def _download_hub(storage_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Install model preparation dependencies with "
            "`uv sync --locked --extra model-prep`."
        ) from exc

    destination = storage_root / str(spec["local_directory"])
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(spec["repo_id"]),
        revision=str(spec["revision"]),
        local_dir=destination,
        allow_patterns=list(spec.get("allow_patterns", ["*"])),
    )
    downloaded_bytes = _directory_bytes(destination)
    expected_min_bytes = int(spec.get("expected_min_bytes", 0))
    if downloaded_bytes < expected_min_bytes:
        raise RuntimeError(
            f"Incomplete Hugging Face snapshot at {destination}: "
            f"{downloaded_bytes} < {expected_min_bytes} bytes"
        )
    return {
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "path": str(destination),
        "bytes": downloaded_bytes,
    }


def _download_direct(storage_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    destination = storage_root / str(spec["local_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(spec.get("expected_bytes", 0))
    if (
        destination.is_file()
        and expected_bytes
        and destination.stat().st_size == expected_bytes
    ):
        return {
            "url": spec["url"],
            "path": str(destination),
            "bytes": expected_bytes,
        }
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            "10",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--output",
            str(destination),
            str(spec["url"]),
        ],
        check=True,
    )
    if expected_bytes and destination.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Incomplete direct artifact at {destination}: "
            f"{destination.stat().st_size} != {expected_bytes} bytes"
        )
    return {
        "url": spec["url"],
        "path": str(destination),
        "bytes": destination.stat().st_size,
    }


def load_registry(path: str | Path) -> tuple[dict[str, Any], Path, Path, Path]:
    registry_path = Path(path).expanduser().resolve()
    registry = load_json(registry_path)
    project_root = registry_path.parent.parent
    storage_root = _resolve(project_root, str(registry.get("storage_root", "models")))
    code_root = _resolve(project_root, str(registry.get("code_root", "external")))
    return registry, registry_path, storage_root, code_root


def download_models(
    registry_path: str | Path, selected: list[str]
) -> dict[str, Any]:
    registry, resolved_registry, storage_root, code_root = load_registry(registry_path)
    models = dict(registry["models"])
    names = list(models) if selected == ["all"] else selected
    unknown = sorted(set(names) - set(models))
    if unknown:
        raise ValueError(f"Unknown model names: {', '.join(unknown)}")

    manifest_path = storage_root / "download_manifest.json"
    if manifest_path.is_file():
        previous = load_json(manifest_path)
        results: dict[str, Any] = dict(previous.get("models", {}))
    else:
        results = {}
    cloned_directories: set[str] = set()
    for name in names:
        spec = dict(models[name])
        result: dict[str, Any] = {"display_name": spec["display_name"]}
        code = spec.get("code")
        if code and str(code["directory"]) not in cloned_directories:
            result["code"] = _clone_code(code_root, dict(code))
            cloned_directories.add(str(code["directory"]))
        result["hub_artifacts"] = [
            _download_hub(storage_root, dict(item))
            for item in spec.get("hub_artifacts", [])
        ]
        result["direct_artifacts"] = [
            _download_direct(storage_root, dict(item))
            for item in spec.get("direct_artifacts", [])
        ]
        results[name] = result
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "registry": str(resolved_registry),
            "models": results,
        }
        write_json(manifest_path, manifest)
    return manifest


def model_status(registry_path: str | Path) -> dict[str, Any]:
    registry, resolved_registry, storage_root, code_root = load_registry(registry_path)
    status: dict[str, Any] = {}
    for name, raw_spec in registry["models"].items():
        spec = dict(raw_spec)
        artifacts: list[dict[str, Any]] = []
        for item in spec.get("hub_artifacts", []):
            path = storage_root / str(item["local_directory"])
            size = _directory_bytes(path)
            expected_min = int(item.get("expected_min_bytes", 0))
            artifacts.append(
                {
                    "kind": "huggingface",
                    "path": str(path),
                    "exists": path.exists(),
                    "bytes": size,
                    "complete": path.exists() and size >= expected_min,
                }
            )
        for item in spec.get("direct_artifacts", []):
            path = storage_root / str(item["local_path"])
            size = path.stat().st_size if path.is_file() else 0
            expected = int(item.get("expected_bytes", 0))
            artifacts.append(
                {
                    "kind": "direct",
                    "path": str(path),
                    "exists": path.is_file(),
                    "bytes": size,
                    "complete": path.is_file() and (not expected or size == expected),
                }
            )
        code = spec.get("code")
        code_path = code_root / str(code["directory"]) if code else None
        status[name] = {
            "display_name": spec["display_name"],
            "code_exists": code_path is None or (code_path / ".git").is_dir(),
            "artifacts": artifacts,
        }
    return {
        "registry": str(resolved_registry),
        "storage_root": str(storage_root),
        "models": status,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and audit pinned model assets.")
    parser.add_argument("command", choices=("download", "status"))
    parser.add_argument(
        "--registry", default="configs/model_registry.json", type=Path
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Registry model key. Repeat for multiple models; defaults to all.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "download":
        payload = download_models(args.registry, args.models or ["all"])
    else:
        payload = model_status(args.registry)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
