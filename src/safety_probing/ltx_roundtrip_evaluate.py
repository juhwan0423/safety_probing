from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .io_utils import load_json, write_json, write_torch_atomic
from .ltx_safety_query import (
    _model_factory,
    aggregate_rollout_predictions,
    evaluate_ltx_predictions,
    subset_targets,
)
from .safety_query_probe import _aggregate, _indices, _predict
from .vjepa_features import _read_jsonl, _sha256


PRIMARY_METRIC_PATHS = [
    ("splits", "test", "static", "load_mass_kg", "mae"),
    ("splits", "test", "static", "load_com_lateral_m", "mae"),
    ("splits", "test", "future_state", "roll_deg", "mae"),
    (
        "splits",
        "test",
        "future_summary",
        "min_zmp_outward_margin_m",
        "mae",
    ),
    ("splits", "test", "future_tip", "auroc"),
    ("splits", "test", "future_tip", "auprc"),
    ("splits", "test", "future_tip", "balanced_accuracy"),
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
]


def roundtrip_layout(cache, target_records):
    condition_to_target = {
        str(record["condition_id"]): index
        for index, record in enumerate(target_records)
    }
    by_seed: dict[int, list[int]] = {}
    conditions_by_seed: dict[int, list[str]] = {}
    for index, record in enumerate(cache["records"]):
        seed = int(record["decoder_seed"])
        condition_id = str(record["condition_id"])
        if condition_id not in condition_to_target:
            raise RuntimeError(f"Unknown round-trip condition {condition_id}")
        by_seed.setdefault(seed, []).append(index)
        conditions_by_seed.setdefault(seed, []).append(condition_id)
    if not by_seed:
        raise RuntimeError("Round-trip cache has no records")
    seeds = sorted(by_seed)
    reference = conditions_by_seed[seeds[0]]
    if len(reference) != len(set(reference)):
        raise RuntimeError("A decoder seed repeats a condition")
    for seed in seeds[1:]:
        if conditions_by_seed[seed] != reference:
            raise RuntimeError("Decoder seeds do not cover the same condition order")
    target_indices = [condition_to_target[condition_id] for condition_id in reference]
    records = [target_records[index] for index in target_indices]
    return seeds, by_seed, target_indices, records


def _subset_predictions(predictions, indices):
    return {
        key: value[indices]
        for key, value in predictions.items()
        if hasattr(value, "shape")
    }


def _path_value(report: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = report
    for key in path:
        value = value[key]
    return float(value)


def evaluate_control_gate(
    gt_native_aggregate: dict[str, Any],
    gt_roundtrip_aggregate: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    def mean(path: str, report: dict[str, Any]) -> float:
        return float(report[path]["mean"])

    auroc_path = "splits.test.future_tip.auroc"
    mae_paths = {
        "load_mass_kg": "splits.test.static.load_mass_kg.mae",
        "load_com_lateral_m": "splits.test.static.load_com_lateral_m.mae",
        "roll_deg": "splits.test.future_state.roll_deg.mae",
        "min_zmp_outward_margin_m": (
            "splits.test.future_summary.min_zmp_outward_margin_m.mae"
        ),
    }
    direction_paths = {
        "load_mass": (
            "pairs.test.hidden_property_delta.load_mass.direction_accuracy"
        ),
        "com_shift": (
            "pairs.test.hidden_property_delta.com_shift.direction_accuracy"
        ),
    }
    auroc_drop = mean(auroc_path, gt_native_aggregate) - mean(
        auroc_path, gt_roundtrip_aggregate
    )
    mae_ratios = {
        name: mean(path, gt_roundtrip_aggregate)
        / max(mean(path, gt_native_aggregate), 1e-12)
        for name, path in mae_paths.items()
    }
    direction_drops = {
        name: mean(path, gt_native_aggregate)
        - mean(path, gt_roundtrip_aggregate)
        for name, path in direction_paths.items()
    }
    checks = {
        "tip_auroc": auroc_drop
        <= float(gate_config["max_tip_auroc_drop"]),
        **{
            f"mae_ratio_{name}": value
            <= float(gate_config["max_primary_mae_ratio"])
            for name, value in mae_ratios.items()
        },
        **{
            f"direction_drop_{name}": value
            <= float(
                gate_config["max_intervention_direction_accuracy_drop"]
            )
            for name, value in direction_drops.items()
        },
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "tip_auroc_drop": auroc_drop,
            "primary_mae_ratios": mae_ratios,
            "intervention_direction_accuracy_drops": direction_drops,
        },
        "thresholds": gate_config,
    }


def pathway_gap_persistence(
    aggregate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metric_paths = {
        "load_mass_kg_mae": "splits.test.static.load_mass_kg.mae",
        "load_com_lateral_m_mae": (
            "splits.test.static.load_com_lateral_m.mae"
        ),
        "roll_deg_mae": "splits.test.future_state.roll_deg.mae",
        "min_zmp_outward_margin_m_mae": (
            "splits.test.future_summary.min_zmp_outward_margin_m.mae"
        ),
    }
    output = {}
    for name, path in metric_paths.items():
        gt_native = float(aggregate["gt_native"][path]["mean"])
        gt_roundtrip = float(aggregate["gt_roundtrip"][path]["mean"])
        generated_native = float(aggregate["generated_native"][path]["mean"])
        generated_roundtrip = float(
            aggregate["generated_roundtrip"][path]["mean"]
        )
        native_gap = generated_native - gt_native
        roundtrip_gap = generated_roundtrip - gt_roundtrip
        output[name] = {
            "gt_roundtrip_minus_native": gt_roundtrip - gt_native,
            "generated_roundtrip_minus_native": (
                generated_roundtrip - generated_native
            ),
            "native_generated_minus_gt_gap": native_gap,
            "roundtrip_generated_minus_gt_gap": roundtrip_gap,
            "gap_persistence_ratio": (
                roundtrip_gap / native_gap if abs(native_gap) > 1e-12 else None
            ),
        }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply frozen GT-trained LTX PSQs to decode/re-encode controls."
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        default=Path("configs/experiments/ltx_roundtrip_control.json"),
    )
    parser.add_argument(
        "--query-experiment",
        type=Path,
        default=Path("configs/experiments/ltx_safety_query.json"),
    )
    parser.add_argument(
        "--gt-roundtrip-cache",
        type=Path,
        default=Path("artifacts/features/ltx_2_3_full_gt_roundtrip_future.pt"),
    )
    parser.add_argument(
        "--generated-roundtrip-cache",
        type=Path,
        default=Path(
            "artifacts/features/ltx_2_3_full_generated_roundtrip_future.pt"
        ),
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path("artifacts/targets/ltx_safety_query_full.pt"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/ltx_safety_query/full_prompt_v1"),
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
        "--output-root",
        type=Path,
        default=Path("outputs/ltx_roundtrip_control/full_prompt_v1"),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:  # noqa: PLR0915
    args = _parser().parse_args()
    import torch

    control_path = args.control_config.resolve()
    query_path = args.query_experiment.resolve()
    gt_path = args.gt_roundtrip_cache.resolve()
    generated_path = args.generated_roundtrip_cache.resolve()
    target_path = args.target_cache.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    control = load_json(control_path)
    query = load_json(query_path)
    gt_cache = torch.load(gt_path, map_location="cpu", mmap=True, weights_only=True)
    generated_cache = torch.load(
        generated_path, map_location="cpu", mmap=True, weights_only=True
    )
    targets = torch.load(
        target_path, map_location="cpu", mmap=True, weights_only=True
    )
    records = targets["records"]
    pair_rows = _read_jsonl(args.pairs_manifest.resolve())
    run_rows = _read_jsonl(args.runs_manifest.resolve())

    expected_control_sha = _sha256(control_path)
    expected_query_sha = _sha256(query_path)
    for name, cache, source in (
        ("GT", gt_cache, "gt"),
        ("generated", generated_cache, "generated"),
    ):
        contract = cache["contract"]
        if contract["control_config_sha256"] != expected_control_sha:
            raise RuntimeError(f"{name} round-trip control hash differs")
        if contract["query_experiment_sha256"] != expected_query_sha:
            raise RuntimeError(f"{name} round-trip query hash differs")
        if contract["source"] != source:
            raise RuntimeError(f"{name} round-trip source differs")

    gt_seeds, gt_by_seed, gt_target_indices, gt_records = roundtrip_layout(
        gt_cache, records
    )
    (
        generated_seeds,
        generated_by_seed,
        generated_target_indices,
        generated_records,
    ) = roundtrip_layout(generated_cache, records)
    if gt_seeds != generated_seeds:
        raise RuntimeError("GT and generated decoder seeds differ")
    gt_condition_ids = [str(record["condition_id"]) for record in gt_records]
    generated_condition_ids = [
        str(record["condition_id"]) for record in generated_records
    ]
    if gt_condition_ids != generated_condition_ids:
        raise RuntimeError("GT and generated round-trip condition order differs")
    if gt_target_indices != generated_target_indices:
        raise RuntimeError("GT and generated target indices differ")
    control_targets = subset_targets(
        targets, torch.tensor(gt_target_indices, dtype=torch.long)
    )

    training = query["training"]
    batch_size = int(training["batch_size"])
    query_seeds = [int(seed) for seed in control["fixed_query_checkpoints"]]
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[LTX round-trip evaluation] conditions={len(gt_records)} "
        f"decoder_seeds={gt_seeds} query_seeds={query_seeds}",
        flush=True,
    )

    per_query_seed = []
    for query_seed in query_seeds:
        checkpoint_path = checkpoint_root / f"seed_{query_seed:04d}.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if checkpoint["experiment_sha256"] != expected_query_sha:
            raise RuntimeError(
                f"PSQ checkpoint query hash differs: {checkpoint_path}"
            )
        normalizers = checkpoint["normalizers"]
        model = _model_factory(gt_cache["features"], targets, query)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(args.device).eval()

        native_full_evaluation, _ = evaluate_ltx_predictions(
            checkpoint["real_predictions_normalized"],
            targets,
            records,
            targets["channels"],
            normalizers,
            pair_rows,
            run_rows,
        )
        threshold = native_full_evaluation["validation_tip_threshold"]
        native_gt_predictions = _subset_predictions(
            checkpoint["real_predictions_normalized"], gt_target_indices
        )
        native_gt_evaluation, _ = evaluate_ltx_predictions(
            native_gt_predictions,
            control_targets,
            gt_records,
            targets["channels"],
            normalizers,
            pair_rows,
            run_rows,
            threshold=threshold,
        )

        checkpoint_generated_ids = list(checkpoint["rollout_condition_ids"])
        generated_lookup = {
            condition_id: index
            for index, condition_id in enumerate(checkpoint_generated_ids)
        }
        missing = [
            condition_id
            for condition_id in generated_condition_ids
            if condition_id not in generated_lookup
        ]
        if missing:
            raise RuntimeError(
                f"Native generated predictions miss condition {missing[0]}"
            )
        native_generated_indices = [
            generated_lookup[condition_id]
            for condition_id in generated_condition_ids
        ]
        native_generated_predictions = _subset_predictions(
            checkpoint["rollout_ensemble_predictions_normalized"],
            native_generated_indices,
        )
        native_generated_evaluation, _ = evaluate_ltx_predictions(
            native_generated_predictions,
            control_targets,
            generated_records,
            targets["channels"],
            normalizers,
            pair_rows,
            run_rows,
            threshold=threshold,
        )

        path_results = {}
        prediction_payload = {}
        for path_name, cache, by_seed in (
            ("gt_roundtrip", gt_cache, gt_by_seed),
            ("generated_roundtrip", generated_cache, generated_by_seed),
        ):
            predictions_by_seed = []
            per_decoder_seed = {}
            for decoder_seed in gt_seeds:
                indices = by_seed[decoder_seed]
                features = cache["features"][indices]
                predictions = _predict(
                    model,
                    features,
                    torch.arange(len(features), dtype=torch.long),
                    batch_size,
                    args.device,
                )
                evaluation, _ = evaluate_ltx_predictions(
                    predictions,
                    control_targets,
                    gt_records,
                    targets["channels"],
                    normalizers,
                    pair_rows,
                    run_rows,
                    threshold=threshold,
                )
                predictions_by_seed.append(predictions)
                per_decoder_seed[str(decoder_seed)] = evaluation
            ensemble_predictions = aggregate_rollout_predictions(
                predictions_by_seed
            )
            ensemble_evaluation, ensemble_physical = evaluate_ltx_predictions(
                ensemble_predictions,
                control_targets,
                gt_records,
                targets["channels"],
                normalizers,
                pair_rows,
                run_rows,
                threshold=threshold,
            )
            path_results[path_name] = {
                "per_decoder_seed": per_decoder_seed,
                "ensemble": ensemble_evaluation,
            }
            prediction_payload[path_name] = {
                "by_decoder_seed_normalized": {
                    seed: predictions
                    for seed, predictions in zip(
                        gt_seeds, predictions_by_seed, strict=True
                    )
                },
                "ensemble_normalized": ensemble_predictions,
                "ensemble_physical": ensemble_physical,
            }

        seed_result = {
            "query_seed": query_seed,
            "validation_tip_threshold": threshold,
            "gt_native": native_gt_evaluation,
            "generated_native": native_generated_evaluation,
            **path_results,
        }
        per_query_seed.append(seed_result)
        write_torch_atomic(
            output_root / f"seed_{query_seed:04d}.pt",
            {
                "query_seed": query_seed,
                "condition_ids": gt_condition_ids,
                "decoder_seeds": gt_seeds,
                "predictions": prediction_payload,
                "control_config_sha256": expected_control_sha,
                "query_experiment_sha256": expected_query_sha,
            },
        )
        summary = {
            name: _path_value(
                result["splits"]["test"],
                ("future_tip", "auroc"),
            )
            for name, result in (
                ("GT native", native_gt_evaluation),
                ("GT round-trip", path_results["gt_roundtrip"]["ensemble"]),
                ("Generated native", native_generated_evaluation),
                (
                    "Generated round-trip",
                    path_results["generated_roundtrip"]["ensemble"],
                ),
            )
        }
        print(
            f"[LTX round-trip evaluation] query_seed={query_seed} "
            f"tip_AUROC={json.dumps(summary)}",
            flush=True,
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = {
        path_name: _aggregate(
            [
                item[path_name]["ensemble"]
                if path_name.endswith("roundtrip")
                else item[path_name]
                for item in per_query_seed
            ],
            PRIMARY_METRIC_PATHS,
        )
        for path_name in (
            "gt_native",
            "gt_roundtrip",
            "generated_native",
            "generated_roundtrip",
        )
    }
    gate = evaluate_control_gate(
        aggregate["gt_native"],
        aggregate["gt_roundtrip"],
        control["control_validity_gate"],
    )
    gap_persistence = pathway_gap_persistence(aggregate)
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": control["experiment"],
        "protocol_version": control["protocol_version"],
        "control_config_sha256": expected_control_sha,
        "query_experiment_sha256": expected_query_sha,
        "foundation_parameters_frozen": True,
        "query_parameters_frozen": True,
        "query_training_source": "native_gt_future_latents_only",
        "counts": {
            "conditions": len(gt_records),
            "validation": len(_indices(gt_records, "val")),
            "test": len(_indices(gt_records, "test")),
            "decoder_seeds": gt_seeds,
            "query_seeds": query_seeds,
            "jobs_per_path": len(gt_cache["records"]),
        },
        "paths": {
            "gt_roundtrip_cache": str(gt_path),
            "generated_roundtrip_cache": str(generated_path),
            "target_cache": str(target_path),
            "checkpoint_root": str(checkpoint_root),
        },
        "per_query_seed": per_query_seed,
        "aggregate": aggregate,
        "control_validity_gate": gate,
        "pathway_gap_persistence": gap_persistence,
    }
    write_json(output_root / "result.json", result)
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "gate": gate,
                "pathway_gap_persistence": gap_persistence,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
