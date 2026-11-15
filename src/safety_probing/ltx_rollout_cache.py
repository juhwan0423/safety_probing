from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .io_utils import load_json, write_json, write_torch_atomic
from .vjepa_features import _sha256


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_future_feature(payload, *, prefix_latent_frames: int):
    import torch

    latent = payload["latent"]
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise RuntimeError("Expected rollout latent [1,C,T,H,W]")
    if int(payload["prefix_latent_frames"]) != prefix_latent_frames:
        raise RuntimeError("Rollout prefix latent geometry changed")
    feature = latent[0, :, prefix_latent_frames:].permute(1, 2, 3, 0)
    feature = feature.to(dtype=torch.float16, device="cpu")
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError("Rollout contains non-finite LTX latent values")
    return feature


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate saved LTX continuation latents for Safety Query."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/ltx_safety_query.json"),
    )
    parser.add_argument(
        "--real-cache",
        type=Path,
        default=Path("artifacts/features/ltx_2_3_part_01_real_future.pt"),
    )
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/features/ltx_2_3_part_01_rollout_future.pt"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    experiment_path = args.experiment.resolve()
    real_path = args.real_cache.resolve()
    output_path = args.output.resolve()
    experiment = load_json(experiment_path)
    rollout_config = experiment["rollout"]
    latent_config = experiment["real_latent"]
    rollout_root = (
        args.rollout_root.resolve()
        if args.rollout_root is not None
        else Path(rollout_config["root"]).resolve()
    )
    manifest_path = rollout_root / "batch_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    project_root = experiment_path.parent.parent.parent
    rollout_experiment_path = (
        project_root / str(rollout_config["experiment"])
    ).resolve()
    expected_rollout_experiment_sha256 = _sha256(rollout_experiment_path)
    real_cache = torch.load(
        real_path, map_location="cpu", mmap=True, weights_only=True
    )
    real_records = {
        str(record["condition_id"]): record for record in real_cache["records"]
    }
    requested_splits = set(rollout_config["splits"])
    requested_seeds = {int(seed) for seed in rollout_config["seeds"]}
    rows = [
        row
        for row in _read_jsonl(manifest_path)
        if row.get("status") == "complete"
        and str(row.get("condition_id")) in real_records
        and str(real_records[str(row["condition_id"])]["split"])
        in requested_splits
        and int(row.get("seed", -1)) in requested_seeds
    ]
    rows.sort(
        key=lambda row: (
            str(real_records[str(row["condition_id"])]["split"]),
            str(row["condition_id"]),
            int(row["seed"]),
        )
    )
    if not rows:
        raise RuntimeError("No completed LTX rollout records matched the contract")
    keys = [(str(row["condition_id"]), int(row["seed"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate condition/seed rollout records")
    expected_conditions = {
        condition_id
        for condition_id, record in real_records.items()
        if record["split"] in requested_splits
    }
    expected_keys = {
        (condition_id, seed)
        for condition_id in expected_conditions
        for seed in requested_seeds
    }
    missing = sorted(expected_keys - set(keys))
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"LTX rollout cache is missing {len(missing)} condition/seed jobs"
        )
    display = {
        "rollout_root": str(rollout_root),
        "selected_jobs": len(rows),
        "selected_conditions": len({key[0] for key in keys}),
        "missing_jobs": len(missing),
        "output": str(output_path),
    }
    print(json.dumps(display, indent=2), flush=True)
    if args.dry_run:
        return

    prefix_latent_frames = int(latent_config["prefix_latent_frames"])
    features = []
    records = []
    rollout_experiment_hashes = set()
    for index, row in enumerate(rows, start=1):
        latent_path = rollout_root / str(row["latent_path"])
        payload = torch.load(latent_path, map_location="cpu", weights_only=True)
        feature = extract_future_feature(
            payload, prefix_latent_frames=prefix_latent_frames
        )
        expected_tail = (
            int(latent_config["future_latent_frames"]),
            int(latent_config["latent_height"]),
            int(latent_config["latent_width"]),
            int(latent_config["latent_channels"]),
        )
        if tuple(feature.shape) != expected_tail:
            raise RuntimeError(
                f"Expected rollout feature {expected_tail}, got {tuple(feature.shape)}"
            )
        condition_id = str(row["condition_id"])
        source = real_records[condition_id]
        features.append(feature)
        records.append(
            {
                "condition_id": condition_id,
                "run_id": str(source["run_id"]),
                "split": str(source["split"]),
                "safety_outcome": str(source["safety_outcome"]),
                "rollout_seed": int(row["seed"]),
                "latent_path": str(latent_path),
            }
        )
        rollout_experiment_hashes.add(str(row["experiment_sha256"]))
        if index % 100 == 0 or index == len(rows):
            print(f"[LTX rollout cache] {index}/{len(rows)}", flush=True)
    if len(rollout_experiment_hashes) != 1:
        raise RuntimeError("Selected rollouts mix experiment hashes")
    if rollout_experiment_hashes != {expected_rollout_experiment_sha256}:
        raise RuntimeError(
            "Saved rollout experiment hash differs from configured prompt protocol"
        )
    feature_tensor = torch.stack(features)
    contract = {
        "schema_version": 1,
        "experiment_sha256": _sha256(experiment_path),
        "real_cache_contract": real_cache["contract"],
        "rollout_manifest_sha256": _sha256(manifest_path),
        "rollout_experiment_path": str(rollout_experiment_path),
        "rollout_experiment_sha256": next(iter(rollout_experiment_hashes)),
        "job_count": len(records),
        "condition_count": len({record["condition_id"] for record in records}),
        "condition_seeds": [
            [record["condition_id"], record["rollout_seed"]]
            for record in records
        ],
        "missing_job_count": len(missing),
    }
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "records": records,
        "features": feature_tensor,
    }
    write_torch_atomic(output_path, output)
    write_json(
        output_path.with_suffix(".json"),
        {
            key: value for key, value in output.items() if key != "features"
        }
        | {"feature_shape": list(feature_tensor.shape)},
    )
    print(f"[LTX rollout cache] Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
