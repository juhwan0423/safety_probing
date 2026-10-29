from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .io_utils import load_json, write_json, write_torch_atomic


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_auroc(labels, scores) -> float:
    import torch

    labels = torch.as_tensor(labels, dtype=torch.long).flatten()
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).double() + 0.5 * (comparisons == 0).double()).mean()
    )


def average_precision(labels, scores) -> float:
    import torch

    labels = torch.as_tensor(labels, dtype=torch.long).flatten()
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    positive_count = int((labels == 1).sum())
    if positive_count == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order]
    cumulative_positive = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    precision = cumulative_positive.double() / ranks
    return float(precision[sorted_labels == 1].sum() / positive_count)


def _binary_counts(labels, predictions) -> tuple[int, int, int, int]:
    import torch

    labels = torch.as_tensor(labels, dtype=torch.long)
    predictions = torch.as_tensor(predictions, dtype=torch.long)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return tp, tn, fp, fn


def balanced_accuracy(labels, predictions) -> float:
    tp, tn, fp, fn = _binary_counts(labels, predictions)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return (tpr + tnr) / 2


def select_balanced_threshold(labels, probabilities) -> float:
    import torch

    probabilities = torch.as_tensor(probabilities, dtype=torch.float64).flatten()
    unique = torch.unique(probabilities).sort().values
    if len(unique) == 1:
        return float(unique[0])
    candidates = torch.cat(
        [
            torch.tensor([0.0], dtype=torch.float64),
            (unique[:-1] + unique[1:]) / 2,
            torch.tensor([1.0], dtype=torch.float64),
        ]
    )
    best = (-float("inf"), -float("inf"), 0.5)
    for threshold in candidates.tolist():
        predictions = (probabilities >= threshold).long()
        balanced = balanced_accuracy(labels, predictions)
        accuracy = float(
            (predictions == torch.as_tensor(labels, dtype=torch.long)).float().mean()
        )
        candidate = (balanced, accuracy, -abs(threshold - 0.5))
        if candidate > best[:3]:
            best = (balanced, accuracy, -abs(threshold - 0.5))
            best_threshold = threshold
    return float(best_threshold)


def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
    import torch

    labels = torch.as_tensor(labels, dtype=torch.float64).flatten()
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64).flatten()
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if bool(mask.any()):
            confidence = float(probabilities[mask].mean())
            accuracy = float(labels[mask].mean())
            error += float(mask.double().mean()) * abs(confidence - accuracy)
    return error


def classification_metrics(labels, logits, threshold: float) -> dict[str, Any]:
    import torch

    labels = torch.as_tensor(labels, dtype=torch.long).flatten()
    logits = torch.as_tensor(logits, dtype=torch.float64).flatten()
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).long()
    tp, tn, fp, fn = _binary_counts(labels, predictions)
    return {
        "count": len(labels),
        "positive_count": int((labels == 1).sum()),
        "auroc": binary_auroc(labels, logits),
        "auprc": average_precision(labels, logits),
        "accuracy": float((predictions == labels).double().mean()),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "brier": float(((probabilities - labels.double()) ** 2).mean()),
        "ece_10_bin": expected_calibration_error(labels, probabilities),
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _average_ranks(values):
    import torch

    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty(len(values), dtype=torch.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_correlation(targets, predictions) -> float:
    import torch

    target_ranks = _average_ranks(targets)
    prediction_ranks = _average_ranks(predictions)
    target_centered = target_ranks - target_ranks.mean()
    prediction_centered = prediction_ranks - prediction_ranks.mean()
    denominator = torch.sqrt(
        (target_centered**2).sum() * (prediction_centered**2).sum()
    )
    if float(denominator) == 0.0:
        return float("nan")
    return float((target_centered * prediction_centered).sum() / denominator)


def regression_metrics(targets, predictions) -> dict[str, Any]:
    import torch

    targets = torch.as_tensor(targets, dtype=torch.float64).flatten()
    predictions = torch.as_tensor(predictions, dtype=torch.float64).flatten()
    errors = predictions - targets
    return {
        "count": len(targets),
        "mae": float(errors.abs().mean()),
        "rmse": float(torch.sqrt((errors**2).mean())),
        "spearman": spearman_correlation(targets, predictions),
    }


def pairwise_accuracy(scores, pairs: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    correct = 0.0
    tie_count = 0
    for pair in pairs:
        difference = float(scores[pair["tip_index"]] - scores[pair["stable_index"]])
        if difference > 0:
            correct += 1.0
        elif difference == 0:
            correct += 0.5
            tie_count += 1
    return {
        "pair_count": len(pairs),
        "correct_credit": correct,
        "tie_count": tie_count,
        "accuracy": correct / len(pairs) if pairs else None,
    }


def build_gold_pair_indices(
    pair_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    condition_to_index: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    runs = {
        (str(row["pair_id"]), str(row["run_label"])): row for row in run_rows
    }
    output: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for pair in pair_rows:
        if pair.get("quality_tier") != "gold":
            continue
        outcomes = {
            pair.get("run_A_safety_outcome"),
            pair.get("run_B_safety_outcome"),
        }
        if outcomes != {"observed_tip", "confirmed_recovered"}:
            continue
        pair_id = str(pair["pair_id"])
        endpoints = {label: runs[(pair_id, label)] for label in ("A", "B")}
        indices = {
            label: condition_to_index[str(endpoints[label]["condition_id"])]
            for label in ("A", "B")
        }
        tip_label = (
            "A" if pair["run_A_safety_outcome"] == "observed_tip" else "B"
        )
        stable_label = "B" if tip_label == "A" else "A"
        split = str(pair["splits"]["main"])
        output[split].append(
            {
                "pair_id": pair_id,
                "tip_index": indices[tip_label],
                "stable_index": indices[stable_label],
                "turn_direction": pair["turn_direction"],
                "intervention_type": pair["intervention_type"],
            }
        )
    return output


def _pair_reports(scores, pairs_by_split) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, pairs in pairs_by_split.items():
        split_report = pairwise_accuracy(scores, pairs)
        split_report["by_direction"] = {
            direction: pairwise_accuracy(
                scores,
                [pair for pair in pairs if pair["turn_direction"] == direction],
            )
            for direction in ("left", "right")
        }
        split_report["by_intervention"] = {
            intervention: pairwise_accuracy(
                scores,
                [
                    pair
                    for pair in pairs
                    if pair["intervention_type"] == intervention
                ],
            )
            for intervention in ("load_mass", "com_shift")
        }
        output[split] = split_report
    return output


def _train_classifier(
    features,
    labels,
    train_indices,
    val_indices,
    val_pairs,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
):
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    x = features.to(device)
    y = labels.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)
    model = torch.nn.Linear(x.shape[1], 1).to(device)
    negative_count = int((y[train_indices] == 0).sum())
    positive_count = int((y[train_indices] == 1).sum())
    pos_weight = torch.tensor(
        negative_count / positive_count, dtype=torch.float32, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_key = None
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(x[train_indices]).squeeze(1)
        loss = criterion(train_logits, y[train_indices])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x).squeeze(1)
            val_loss = float(criterion(logits[val_indices], y[val_indices]))
            val_logits_cpu = logits[val_indices].cpu()
            val_labels_cpu = y[val_indices].cpu()
            val_auc = binary_auroc(val_labels_cpu, val_logits_cpu)
            val_pair = pairwise_accuracy(logits.cpu(), val_pairs)["accuracy"]
        key = (float(val_pair), val_auc, -val_loss)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("Classifier did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(1).cpu()
    return logits, best_state, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "best_validation_pairwise_accuracy": best_key[0],
        "best_validation_auroc": best_key[1],
        "best_validation_loss": best_val_loss,
        "positive_class_weight": float(pos_weight),
    }


def _train_regressor(
    features,
    targets,
    train_indices,
    val_indices,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
):
    import torch

    torch.manual_seed(seed + 1000)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed + 1000)
    x = features.to(device)
    y = targets.to(device)
    train_indices = train_indices.to(device)
    val_indices = val_indices.to(device)
    target_mean = y[train_indices].mean()
    target_std = y[train_indices].std(unbiased=False).clamp_min(1e-6)
    normalized_targets = (y - target_mean) / target_std
    model = torch.nn.Linear(x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[train_indices]).squeeze(1)
        loss = torch.nn.functional.mse_loss(
            prediction, normalized_targets[train_indices]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            normalized_prediction = model(x).squeeze(1)
            raw_prediction = normalized_prediction * target_std + target_mean
            val_rmse = float(
                torch.sqrt(
                    torch.nn.functional.mse_loss(
                        raw_prediction[val_indices], y[val_indices]
                    )
                )
            )
        if val_rmse < best_rmse - 1e-10:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("Regressor did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = (
            model(x).squeeze(1) * target_std + target_mean
        ).cpu()
    return predictions, best_state, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "best_validation_rmse": best_rmse,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }


def _indices_for(records, *, split: str, binary: bool):
    import torch

    indices = []
    for index, record in enumerate(records):
        if record["split"] != split:
            continue
        if binary and record["binary_tip_label"] is None:
            continue
        if not binary and record["future_min_outward_zmp_margin_m"] is None:
            continue
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long)


def _split_indices(records, split: str):
    import torch

    return torch.tensor(
        [index for index, record in enumerate(records) if record["split"] == split],
        dtype=torch.long,
    )


def empirical_prevalence_scores(
    records: list[dict[str, Any]], fields: tuple[str, ...]
):
    """Return a train-only categorical prevalence control for dataset bias audits."""
    import torch

    train_records = [
        record
        for record in records
        if record["split"] == "train"
        and record["binary_tip_label"] is not None
    ]
    if not train_records:
        raise RuntimeError("Cannot fit prevalence control without train labels")
    positive_count = sum(int(record["binary_tip_label"]) for record in train_records)
    global_rate = positive_count / len(train_records)
    counts: dict[tuple[str, ...], list[int]] = {}
    for record in train_records:
        key = tuple(str(record[field]) for field in fields)
        group = counts.setdefault(key, [0, 0])
        group[0] += int(record["binary_tip_label"])
        group[1] += 1
    scores = []
    seen = []
    for record in records:
        key = tuple(str(record[field]) for field in fields)
        if key in counts:
            positive, total = counts[key]
            scores.append((positive + 1) / (total + 2))
            seen.append(True)
        else:
            scores.append(global_rate)
            seen.append(False)
    return torch.tensor(scores, dtype=torch.float32), torch.tensor(seen), {
        "fields": list(fields),
        "train_group_count": len(counts),
        "train_global_tip_rate": global_rate,
        "smoothing": "beta_1_1",
        "unseen_group_fallback": "train_global_tip_rate",
    }


def _aggregate(per_seed: list[dict[str, Any]], paths: list[tuple[str, ...]]):
    output: dict[str, Any] = {}
    for path in paths:
        values = []
        for seed_result in per_seed:
            value: Any = seed_result
            for key in path:
                value = value[key]
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
        name = ".".join(path)
        output[name] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train J1 linear probes on cached frozen V-JEPA features."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_linear_probe.json"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path(
            "artifacts/features/vjepa_2_1_part_01_observation.pt"
        ),
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
        default=Path("outputs/vjepa_linear_probe/part_01"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    experiment_path = args.experiment.resolve()
    feature_cache_path = args.feature_cache.resolve()
    pairs_manifest_path = args.pairs_manifest.resolve()
    runs_manifest_path = args.runs_manifest.resolve()
    output_root = args.output_root.resolve()
    experiment = load_json(experiment_path)
    cache = torch.load(
        feature_cache_path, map_location="cpu", mmap=True, weights_only=True
    )
    records = cache["records"]
    temporal_features = cache["features"].float()
    if temporal_features.ndim != 3:
        raise RuntimeError("Expected cached features with shape [N,T,D]")
    if len(records) != temporal_features.shape[0]:
        raise RuntimeError("Feature cache record count mismatch")
    expected_model = str(experiment["primary_model_registry_key"])
    if cache["model"]["registry_key"] != expected_model:
        raise RuntimeError(
            f"Expected cache for {expected_model}, got {cache['model']['registry_key']}"
        )

    pooled_features = temporal_features.mean(dim=1)
    train_all = _split_indices(records, "train")
    feature_mean = pooled_features[train_all].mean(dim=0)
    feature_std = pooled_features[train_all].std(
        dim=0, unbiased=False
    ).clamp_min(1e-6)
    features = (pooled_features - feature_mean) / feature_std

    labels = torch.tensor(
        [
            float("nan")
            if record["binary_tip_label"] is None
            else float(record["binary_tip_label"])
            for record in records
        ],
        dtype=torch.float32,
    )
    margin_targets = torch.tensor(
        [
            float("nan")
            if record["future_min_outward_zmp_margin_m"] is None
            else float(record["future_min_outward_zmp_margin_m"])
            for record in records
        ],
        dtype=torch.float32,
    )
    binary_indices = {
        split: _indices_for(records, split=split, binary=True)
        for split in ("train", "val", "test")
    }
    margin_indices = {
        split: _indices_for(records, split=split, binary=False)
        for split in ("train", "val", "test")
    }
    condition_to_index = {
        str(record["condition_id"]): index
        for index, record in enumerate(records)
    }
    pairs_by_split = build_gold_pair_indices(
        _read_jsonl(pairs_manifest_path),
        _read_jsonl(runs_manifest_path),
        condition_to_index,
    )

    metadata_controls = {}
    for name, fields in {
        "speed_only": ("speed_label",),
        "route": ("turn_direction", "speed_label", "steering_label"),
    }.items():
        scores, seen, specification = empirical_prevalence_scores(records, fields)
        control_logits = torch.logit(scores.clamp(1e-6, 1 - 1e-6))
        val_threshold = select_balanced_threshold(
            labels[binary_indices["val"]], scores[binary_indices["val"]]
        )
        metadata_controls[name] = {
            "specification": specification,
            "classification": {
                split: {
                    **classification_metrics(
                        labels[indices], control_logits[indices], val_threshold
                    ),
                    "unseen_group_count": int((~seen[indices]).sum()),
                }
                for split, indices in binary_indices.items()
            },
            "gold_pairwise": _pair_reports(scores, pairs_by_split),
        }

    training = experiment["training"]
    seeds = list(args.seed or training["seeds"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    patience = int(args.patience or training["patience"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    output_root.mkdir(parents=True, exist_ok=True)
    per_seed = []
    print(
        f"[J1 linear] Features={tuple(features.shape)} seeds={seeds} "
        f"binary train/val/test="
        f"{len(binary_indices['train'])}/{len(binary_indices['val'])}/"
        f"{len(binary_indices['test'])}",
        flush=True,
    )
    for seed in seeds:
        print(f"[J1 linear] Training seed {seed}", flush=True)
        logits, classifier_state, classifier_training = _train_classifier(
            features,
            labels,
            binary_indices["train"],
            binary_indices["val"],
            pairs_by_split["val"],
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            device=args.device,
        )
        val_probabilities = torch.sigmoid(logits[binary_indices["val"]])
        threshold = select_balanced_threshold(
            labels[binary_indices["val"]], val_probabilities
        )
        classification = {
            split: classification_metrics(
                labels[indices], logits[indices], threshold
            )
            for split, indices in binary_indices.items()
        }
        classification_pairs = _pair_reports(logits, pairs_by_split)

        margin_predictions, regressor_state, regression_training = _train_regressor(
            features,
            margin_targets,
            margin_indices["train"],
            margin_indices["val"],
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            device=args.device,
        )
        regression = {
            split: regression_metrics(
                margin_targets[indices], margin_predictions[indices]
            )
            for split, indices in margin_indices.items()
        }
        regression_pairs = _pair_reports(-margin_predictions, pairs_by_split)
        seed_result = {
            "seed": seed,
            "classification_training": classifier_training,
            "classification": classification,
            "classification_pairwise": classification_pairs,
            "regression_training": regression_training,
            "margin_regression": regression,
            "margin_pairwise": regression_pairs,
        }
        per_seed.append(seed_result)
        write_torch_atomic(
            output_root / f"seed_{seed:04d}.pt",
            {
                "seed": seed,
                "classifier_state_dict": classifier_state,
                "regressor_state_dict": regressor_state,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "classification_threshold": threshold,
                "regression_target_mean": regression_training["target_mean"],
                "regression_target_std": regression_training["target_std"],
                "classification_logits": logits,
                "margin_predictions": margin_predictions,
                "feature_cache_contract": cache["contract"],
            },
        )
        print(
            f"[J1 linear] seed={seed} "
            f"test AUROC={classification['test']['auroc']:.4f} "
            f"Gold pair={classification_pairs['test']['accuracy']:.4f} "
            f"margin MAE={regression['test']['mae']:.4f}",
            flush=True,
        )

    aggregate_paths = [
        ("classification", "test", "auroc"),
        ("classification", "test", "auprc"),
        ("classification", "test", "balanced_accuracy"),
        ("classification", "test", "f1"),
        ("classification", "test", "brier"),
        ("classification", "test", "ece_10_bin"),
        ("classification_pairwise", "val", "accuracy"),
        ("classification_pairwise", "test", "accuracy"),
        ("margin_regression", "test", "mae"),
        ("margin_regression", "test", "rmse"),
        ("margin_regression", "test", "spearman"),
        ("margin_pairwise", "test", "accuracy"),
    ]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment["experiment"],
        "experiment_sha256": _sha256(experiment_path),
        "feature_cache": str(feature_cache_path),
        "feature_cache_contract": cache["contract"],
        "pairs_manifest_sha256": _sha256(pairs_manifest_path),
        "runs_manifest_sha256": _sha256(runs_manifest_path),
        "foundation_parameters_frozen": True,
        "probe_type": "linear",
        "feature_shape_before_temporal_pooling": list(temporal_features.shape),
        "feature_shape_after_temporal_pooling": list(pooled_features.shape),
        "counts": {
            "binary": {
                split: len(indices) for split, indices in binary_indices.items()
            },
            "margin": {
                split: len(indices) for split, indices in margin_indices.items()
            },
            "gold_pairs": {
                split: len(pairs) for split, pairs in pairs_by_split.items()
            },
        },
        "metadata_prevalence_controls": metadata_controls,
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed, aggregate_paths),
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
