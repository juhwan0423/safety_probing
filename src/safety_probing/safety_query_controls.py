from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .io_utils import load_json, write_json
from .safety_query_probe import (
    _aggregate,
    _model_factory,
    _predict,
    _read_jsonl,
    _sha256,
    evaluate_predictions,
)


CONTROL_NAMES = (
    "identity",
    "zero_memory",
    "condition_permutation",
    "spatial_mean_repeat",
    "temporal_mean_repeat",
    "temporal_reverse",
)


class IntervenedFeatures:
    """Apply a deterministic diagnostic intervention at cache read time."""

    def __init__(self, features, intervention: str, permutation=None) -> None:
        if intervention not in CONTROL_NAMES:
            raise ValueError(f"Unknown feature intervention: {intervention}")
        self.features = features
        self.intervention = intervention
        self.permutation = permutation
        self.shape = features.shape

    def __getitem__(self, indices):
        import torch

        source_indices = indices
        if self.intervention == "condition_permutation":
            if self.permutation is None:
                raise RuntimeError("condition_permutation requires an index map")
            source_indices = self.permutation[indices]
        values = self.features[source_indices]
        if self.intervention in ("identity", "condition_permutation"):
            return values
        if self.intervention == "zero_memory":
            return torch.zeros_like(values)
        if values.ndim != 5:
            raise ValueError("Control evaluation expects batched [B,T,H,W,D] features")
        if self.intervention == "spatial_mean_repeat":
            return values.mean(dim=(2, 3), keepdim=True).expand_as(values)
        if self.intervention == "temporal_mean_repeat":
            return values.mean(dim=1, keepdim=True).expand_as(values)
        if self.intervention == "temporal_reverse":
            return values.flip(1)
        raise AssertionError("Unreachable feature intervention")


def split_preserving_permutation(records, seed: int):
    import torch

    mapping = torch.arange(len(records), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    for split in ("train", "val", "test"):
        indices = torch.tensor(
            [
                index
                for index, record in enumerate(records)
                if record["split"] == split
            ],
            dtype=torch.long,
        )
        if len(indices) > 1:
            mapping[indices] = indices[
                torch.randperm(len(indices), generator=generator)
            ]
    return mapping


CONTROL_METRICS = (
    ("splits", "test", "static", "load_mass_kg", "mae"),
    ("splits", "test", "static", "load_com_lateral_m", "mae"),
    (
        "splits",
        "test",
        "observed_state",
        "zmp_outward_margin_m",
        "mae",
    ),
    ("splits", "test", "future_state", "roll_deg", "mae"),
    (
        "splits",
        "test",
        "future_summary",
        "min_zmp_outward_margin_m",
        "mae",
    ),
    ("splits", "test", "future_tip", "auroc"),
    (
        "pairs",
        "test",
        "hidden_property_delta",
        "load_mass",
        "direction_accuracy",
    ),
    (
        "pairs",
        "test",
        "hidden_property_delta",
        "com_shift",
        "direction_accuracy",
    ),
)


def _max_prediction_difference(first, second) -> float:
    import torch

    differences = []
    for key in sorted(first):
        if key not in second:
            raise RuntimeError(f"Checkpoint prediction is missing {key}")
        differences.append(float((first[key] - second[key]).abs().max()))
    return max(differences, default=0.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Safety Query decoders under feature interventions."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_safety_query.json"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            "artifacts/features/vjepa_2_1_part_01_observation_spatial12.pt"
        ),
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path("artifacts/targets/safety_query_part_01.pt"),
    )
    parser.add_argument(
        "--pairs-manifest",
        type=Path,
        default=Path("artifacts/current/manifests/pairs.jsonl"),
    )
    parser.add_argument(
        "--runs-manifest",
        type=Path,
        default=Path("artifacts/current/manifests/runs.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/safety_query/part_01"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/safety_query/part_01/controls.json"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--control", action="append", choices=CONTROL_NAMES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--permutation-seed", type=int, default=20260712)
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    experiment_path = args.experiment.resolve()
    feature_path = args.feature_cache.resolve()
    target_path = args.target_cache.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_path = args.output.resolve()
    config = load_json(experiment_path)
    feature_cache = torch.load(
        feature_path, map_location="cpu", mmap=True, weights_only=True
    )
    targets = torch.load(
        target_path, map_location="cpu", mmap=True, weights_only=True
    )
    features = feature_cache["features"]
    records = targets["records"]
    if feature_cache["contract"]["condition_ids"] != targets["contract"][
        "condition_ids"
    ]:
        raise RuntimeError("Feature and target condition order differs")
    if tuple(features.shape[:1]) != (len(records),):
        raise RuntimeError("Feature and target counts differ")

    if args.seed:
        seeds = list(args.seed)
    else:
        seeds = sorted(
            int(path.stem.removeprefix("seed_"))
            for path in checkpoint_root.glob("seed_*.pt")
        )
    if not seeds:
        raise RuntimeError(f"No Safety Query checkpoints found in {checkpoint_root}")
    controls = list(args.control or CONTROL_NAMES)
    permutation = split_preserving_permutation(records, args.permutation_seed)
    pair_rows = _read_jsonl(args.pairs_manifest.resolve())
    run_rows = _read_jsonl(args.runs_manifest.resolve())
    per_seed = []

    for seed in seeds:
        checkpoint_path = checkpoint_root / f"seed_{seed:04d}.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if checkpoint["experiment_sha256"] != _sha256(experiment_path):
            raise RuntimeError(f"Experiment hash mismatch: {checkpoint_path}")
        model = _model_factory(features, targets, config).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        seed_controls: dict[str, Any] = {}
        identity_difference = None
        for control in controls:
            intervened = IntervenedFeatures(
                features,
                control,
                permutation=permutation,
            )
            predictions = _predict(
                model,
                intervened,
                torch.arange(len(records), dtype=torch.long),
                args.batch_size,
                args.device,
            )
            evaluation, _ = evaluate_predictions(
                predictions,
                targets,
                records,
                targets["channels"],
                checkpoint["normalizers"],
                pair_rows,
                run_rows,
            )
            seed_controls[control] = evaluation
            if control == "identity":
                identity_difference = _max_prediction_difference(
                    predictions, checkpoint["predictions_normalized"]
                )
            test = evaluation["splits"]["test"]
            print(
                f"[Safety Query control] seed={seed} control={control} "
                f"mass_mae={test['static']['load_mass_kg']['mae']:.2f}kg "
                f"com_mae={test['static']['load_com_lateral_m']['mae']:.4f}m "
                f"tip_auc={test['future_tip']['auroc']:.4f}",
                flush=True,
            )
        per_seed.append(
            {
                "seed": seed,
                "identity_checkpoint_max_abs_difference": identity_difference,
                "controls": seed_controls,
            }
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = {
        control: _aggregate(
            [item["controls"][control] for item in per_seed],
            CONTROL_METRICS,
        )
        for control in controls
    }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "experiment_sha256": _sha256(experiment_path),
        "feature_cache": str(feature_path),
        "target_cache": str(target_path),
        "checkpoint_root": str(checkpoint_root),
        "seeds": seeds,
        "controls": controls,
        "permutation_seed": args.permutation_seed,
        "control_semantics": {
            "identity": "Unmodified frozen V-JEPA feature cache.",
            "zero_memory": "All encoder values are zero; learned query and position priors remain.",
            "condition_permutation": "Features are permuted within each split, breaking feature-target alignment.",
            "spatial_mean_repeat": "Each tubelet's spatial mean is repeated over the 12x12 grid.",
            "temporal_mean_repeat": "The temporal mean at each spatial cell is repeated over ten tubelets.",
            "temporal_reverse": "The ten tubelets are supplied in reverse order at test time.",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    write_json(output_path, result)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
