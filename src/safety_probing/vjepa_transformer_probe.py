from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from .io_utils import load_json, write_json, write_torch_atomic
from .vjepa_linear_probe import (
    _indices_for,
    _pair_reports,
    _read_jsonl,
    _sha256,
    _split_indices,
    binary_auroc,
    build_gold_pair_indices,
    classification_metrics,
    empirical_prevalence_scores,
    pairwise_accuracy,
    regression_metrics,
    select_balanced_threshold,
)


def build_temporal_transformer_probe(
    *,
    input_dim: int,
    temporal_tokens: int,
    model_dim: int,
    layers: int,
    heads: int,
    feedforward_dim: int,
    dropout: float,
):
    import torch

    class TemporalTransformerProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.temporal_tokens = temporal_tokens
            self.input_projection = torch.nn.Linear(input_dim, model_dim)
            self.position_embedding = torch.nn.Parameter(
                torch.empty(1, temporal_tokens, model_dim)
            )
            layer = torch.nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = torch.nn.TransformerEncoder(
                layer,
                num_layers=layers,
                enable_nested_tensor=False,
            )
            self.output_norm = torch.nn.LayerNorm(model_dim)
            self.head = torch.nn.Linear(model_dim, 1)
            torch.nn.init.trunc_normal_(self.position_embedding, std=0.02)

        def forward(self, features):
            if features.ndim != 3:
                raise ValueError("Expected temporal features with shape [B,T,D]")
            if features.shape[1] != self.temporal_tokens:
                raise ValueError(
                    f"Expected {self.temporal_tokens} temporal tokens, "
                    f"got {features.shape[1]}"
                )
            if features.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected feature dimension {self.input_dim}, "
                    f"got {features.shape[2]}"
                )
            tokens = self.input_projection(features)
            tokens = tokens + self.position_embedding
            tokens = self.encoder(tokens)
            pooled = self.output_norm(tokens).mean(dim=1)
            return self.head(pooled).squeeze(1)

    return TemporalTransformerProbe()


def build_mean_pool_mlp_probe(
    *,
    input_dim: int,
    temporal_tokens: int,
    hidden_dim: int,
    dropout: float,
):
    import torch

    class MeanPoolMLPProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.temporal_tokens = temporal_tokens
            self.hidden = torch.nn.Linear(input_dim, hidden_dim)
            self.activation = torch.nn.GELU()
            self.dropout = torch.nn.Dropout(dropout)
            self.output_norm = torch.nn.LayerNorm(hidden_dim)
            self.head = torch.nn.Linear(hidden_dim, 1)

        def forward(self, features):
            if features.ndim != 3:
                raise ValueError("Expected temporal features with shape [B,T,D]")
            if features.shape[1] != self.temporal_tokens:
                raise ValueError(
                    f"Expected {self.temporal_tokens} temporal tokens, "
                    f"got {features.shape[1]}"
                )
            if features.shape[2] != self.input_dim:
                raise ValueError(
                    f"Expected feature dimension {self.input_dim}, "
                    f"got {features.shape[2]}"
                )
            pooled = features.mean(dim=1)
            hidden = self.hidden(pooled)
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
            hidden = self.output_norm(hidden)
            return self.head(hidden).squeeze(1)

    return MeanPoolMLPProbe()


def fit_token_channel_normalization(features, train_indices):
    train_features = features[train_indices]
    mean = train_features.mean(dim=(0, 1))
    std = train_features.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    return mean, std


def _state_dict_cpu(model) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _batch_indices(indices, batch_size: int, generator):
    import torch

    order = torch.randperm(len(indices), generator=generator)
    for start in range(0, len(indices), batch_size):
        yield indices[order[start : start + batch_size]]


def _build_model(features, probe_config):
    return build_temporal_transformer_probe(
        input_dim=features.shape[2],
        temporal_tokens=features.shape[1],
        model_dim=int(probe_config["model_dim"]),
        layers=int(probe_config["layers"]),
        heads=int(probe_config["heads"]),
        feedforward_dim=int(probe_config["feedforward_dim"]),
        dropout=float(probe_config["dropout"]),
    )


def _train_classifier(
    features,
    labels,
    train_indices,
    val_indices,
    val_pairs,
    model_factory,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    device: str,
):
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    x = features.to(device)
    y = labels.to(device)
    train_device = train_indices.to(device)
    val_device = val_indices.to(device)
    model = model_factory().to(device)
    negative_count = int((y[train_device] == 0).sum())
    positive_count = int((y[train_device] == 1).sum())
    pos_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_key = None
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in _batch_indices(train_indices, batch_size, generator):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]), y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            logits = model(x)
            val_loss = float(criterion(logits[val_device], y[val_device]))
            val_logits = logits[val_device].cpu()
            val_labels = y[val_device].cpu()
            val_auc = binary_auroc(val_labels, val_logits)
            val_pair = pairwise_accuracy(logits.cpu(), val_pairs)["accuracy"]
        key = (float(val_pair), val_auc, -val_loss)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_val_loss = val_loss
            best_state = _state_dict_cpu(model)
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("Transformer classifier did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        logits = model(x).cpu()
        reversed_logits = model(x.flip(dims=(1,))).cpu()
    return logits, reversed_logits, best_state, {
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
    model_factory,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    gradient_clip_norm: float,
    device: str,
):
    import torch

    torch.manual_seed(seed + 1000)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed + 1000)
    generator = torch.Generator().manual_seed(seed + 1000)
    x = features.to(device)
    y = targets.to(device)
    train_device = train_indices.to(device)
    val_device = val_indices.to(device)
    target_mean = y[train_device].mean()
    target_std = y[train_device].std(unbiased=False).clamp_min(1e-6)
    normalized_targets = (y - target_mean) / target_std
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_rmse = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in _batch_indices(train_indices, batch_size, generator):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[batch])
            loss = torch.nn.functional.mse_loss(
                prediction, normalized_targets[batch]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            normalized_prediction = model(x)
            raw_prediction = normalized_prediction * target_std + target_mean
            val_rmse = float(
                torch.sqrt(
                    torch.nn.functional.mse_loss(
                        raw_prediction[val_device], y[val_device]
                    )
                )
            )
        if val_rmse < best_rmse - 1e-10:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
        elif epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("Transformer regressor did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = (model(x) * target_std + target_mean).cpu()
        reversed_predictions = (
            model(x.flip(dims=(1,))) * target_std + target_mean
        ).cpu()
    return predictions, reversed_predictions, best_state, {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "best_validation_rmse": best_rmse,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
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
        output[".".join(path)] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return output


def _j1_comparison(j1_result, j2_aggregate) -> dict[str, Any]:
    mappings = {
        "classification.test.auroc": "higher_is_better",
        "classification.test.auprc": "higher_is_better",
        "classification_pairwise.test.accuracy": "higher_is_better",
        "margin_regression.test.mae": "lower_is_better",
        "margin_regression.test.spearman": "higher_is_better",
        "margin_pairwise.test.accuracy": "higher_is_better",
    }
    output = {}
    for metric, direction in mappings.items():
        j1_mean = j1_result["aggregate"][metric]["mean"]
        j2_mean = j2_aggregate[metric]["mean"]
        raw_delta = j2_mean - j1_mean
        improvement = raw_delta if direction == "higher_is_better" else -raw_delta
        output[metric] = {
            "direction": direction,
            "j1_mean": j1_mean,
            "j2_mean": j2_mean,
            "j2_minus_j1": raw_delta,
            "signed_improvement": improvement,
        }
    return output


def _mean_pool_mlp_comparison(aggregate) -> dict[str, Any]:
    metrics = {
        "classification.test.auroc": "higher_is_better",
        "classification.test.auprc": "higher_is_better",
        "classification_pairwise.test.accuracy": "higher_is_better",
        "margin_regression.test.mae": "lower_is_better",
        "margin_regression.test.spearman": "higher_is_better",
        "margin_pairwise.test.accuracy": "higher_is_better",
    }
    output = {}
    for metric, direction in metrics.items():
        transformer_mean = aggregate[metric]["mean"]
        control_key = f"mean_pool_mlp_control.{metric}"
        control_mean = aggregate[control_key]["mean"]
        raw_delta = transformer_mean - control_mean
        improvement = raw_delta if direction == "higher_is_better" else -raw_delta
        output[metric] = {
            "direction": direction,
            "mean_pool_mlp_mean": control_mean,
            "transformer_mean": transformer_mean,
            "transformer_minus_control": raw_delta,
            "signed_transformer_improvement": improvement,
        }
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train J2 temporal Transformer probes on frozen V-JEPA features."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_transformer_probe.json"),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("artifacts/features/vjepa_2_1_part_01_observation.pt"),
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
        "--linear-result",
        type=Path,
        default=Path("outputs/vjepa_linear_probe/part_01/result.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/vjepa_transformer_probe/part_01"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
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
    linear_result_path = args.linear_result.resolve()
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
    expected_tokens = int(experiment["representation"]["cache_temporal_tokens"])
    if temporal_features.shape[1] != expected_tokens:
        raise RuntimeError(
            f"Expected {expected_tokens} cached temporal tokens, "
            f"got {temporal_features.shape[1]}"
        )

    train_all = _split_indices(records, "train")
    feature_mean, feature_std = fit_token_channel_normalization(
        temporal_features, train_all
    )
    features = (temporal_features - feature_mean) / feature_std
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
        threshold = select_balanced_threshold(
            labels[binary_indices["val"]], scores[binary_indices["val"]]
        )
        metadata_controls[name] = {
            "specification": specification,
            "classification": {
                split: {
                    **classification_metrics(
                        labels[indices], control_logits[indices], threshold
                    ),
                    "unseen_group_count": int((~seen[indices]).sum()),
                }
                for split, indices in binary_indices.items()
            },
            "gold_pairwise": _pair_reports(scores, pairs_by_split),
        }

    probe_config = experiment["probe"]
    mean_pool_mlp_config = experiment["controls"]["mean_pool_mlp"]
    training = experiment["training"]
    seeds = list(args.seed or training["seeds"])
    batch_size = int(args.batch_size or training["batch_size"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    patience = int(args.patience or training["patience"])
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    gradient_clip_norm = float(training["gradient_clip_norm"])
    if batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("Batch size, epochs, and patience must be positive")

    def transformer_model_factory():
        return _build_model(features, probe_config)

    def mean_pool_mlp_model_factory():
        return build_mean_pool_mlp_probe(
            input_dim=features.shape[2],
            temporal_tokens=features.shape[1],
            hidden_dim=int(mean_pool_mlp_config["hidden_dim"]),
            dropout=float(mean_pool_mlp_config["dropout"]),
        )

    reference_model = transformer_model_factory()
    trainable_parameters = sum(
        parameter.numel()
        for parameter in reference_model.parameters()
        if parameter.requires_grad
    )
    del reference_model
    reference_control = mean_pool_mlp_model_factory()
    control_trainable_parameters = sum(
        parameter.numel()
        for parameter in reference_control.parameters()
        if parameter.requires_grad
    )
    del reference_control

    output_root.mkdir(parents=True, exist_ok=True)
    per_seed = []
    print(
        f"[J2 transformer] Features={tuple(features.shape)} "
        f"parameters/task={trainable_parameters:,} "
        f"mean-MLP/task={control_trainable_parameters:,} seeds={seeds} "
        f"binary train/val/test={len(binary_indices['train'])}/"
        f"{len(binary_indices['val'])}/{len(binary_indices['test'])}",
        flush=True,
    )
    for seed in seeds:
        print(f"[J2 transformer] Training seed {seed}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.monotonic()
        logits, reversed_logits, classifier_state, classifier_training = (
            _train_classifier(
                features,
                labels,
                binary_indices["train"],
                binary_indices["val"],
                pairs_by_split["val"],
                transformer_model_factory,
                seed=seed,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                gradient_clip_norm=gradient_clip_norm,
                device=args.device,
            )
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
        reversed_classification = {
            split: classification_metrics(
                labels[indices], reversed_logits[indices], threshold
            )
            for split, indices in binary_indices.items()
        }
        reversed_classification_pairs = _pair_reports(
            reversed_logits, pairs_by_split
        )

        (
            margin_predictions,
            reversed_margin_predictions,
            regressor_state,
            regression_training,
        ) = _train_regressor(
            features,
            margin_targets,
            margin_indices["train"],
            margin_indices["val"],
            transformer_model_factory,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            gradient_clip_norm=gradient_clip_norm,
            device=args.device,
        )
        regression = {
            split: regression_metrics(
                margin_targets[indices], margin_predictions[indices]
            )
            for split, indices in margin_indices.items()
        }
        regression_pairs = _pair_reports(-margin_predictions, pairs_by_split)
        reversed_regression = {
            split: regression_metrics(
                margin_targets[indices], reversed_margin_predictions[indices]
            )
            for split, indices in margin_indices.items()
        }
        reversed_regression_pairs = _pair_reports(
            -reversed_margin_predictions, pairs_by_split
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        duration_s = time.monotonic() - started
        peak_gpu_memory = (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else 0
        )

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        control_started = time.monotonic()
        (
            control_logits,
            _,
            control_classifier_state,
            control_classifier_training,
        ) = _train_classifier(
            features,
            labels,
            binary_indices["train"],
            binary_indices["val"],
            pairs_by_split["val"],
            mean_pool_mlp_model_factory,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            gradient_clip_norm=gradient_clip_norm,
            device=args.device,
        )
        control_val_probabilities = torch.sigmoid(
            control_logits[binary_indices["val"]]
        )
        control_threshold = select_balanced_threshold(
            labels[binary_indices["val"]], control_val_probabilities
        )
        control_classification = {
            split: classification_metrics(
                labels[indices], control_logits[indices], control_threshold
            )
            for split, indices in binary_indices.items()
        }
        control_classification_pairs = _pair_reports(
            control_logits, pairs_by_split
        )
        (
            control_margin_predictions,
            _,
            control_regressor_state,
            control_regression_training,
        ) = _train_regressor(
            features,
            margin_targets,
            margin_indices["train"],
            margin_indices["val"],
            mean_pool_mlp_model_factory,
            seed=seed,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            gradient_clip_norm=gradient_clip_norm,
            device=args.device,
        )
        control_regression = {
            split: regression_metrics(
                margin_targets[indices], control_margin_predictions[indices]
            )
            for split, indices in margin_indices.items()
        }
        control_regression_pairs = _pair_reports(
            -control_margin_predictions, pairs_by_split
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        control_duration_s = time.monotonic() - control_started
        control_peak_gpu_memory = (
            torch.cuda.max_memory_allocated()
            if args.device.startswith("cuda")
            else 0
        )
        seed_result = {
            "seed": seed,
            "classification_training": classifier_training,
            "classification": classification,
            "classification_pairwise": classification_pairs,
            "regression_training": regression_training,
            "margin_regression": regression,
            "margin_pairwise": regression_pairs,
            "reversed_order": {
                "classification": reversed_classification,
                "classification_pairwise": reversed_classification_pairs,
                "margin_regression": reversed_regression,
                "margin_pairwise": reversed_regression_pairs,
            },
            "mean_pool_mlp_control": {
                "classification_training": control_classifier_training,
                "classification": control_classification,
                "classification_pairwise": control_classification_pairs,
                "regression_training": control_regression_training,
                "margin_regression": control_regression,
                "margin_pairwise": control_regression_pairs,
                "runtime": {
                    "duration_s": control_duration_s,
                    "peak_gpu_memory_bytes": control_peak_gpu_memory,
                },
            },
            "runtime": {
                "duration_s": duration_s,
                "peak_gpu_memory_bytes": peak_gpu_memory,
            },
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
                "reversed_order_classification_logits": reversed_logits,
                "margin_predictions": margin_predictions,
                "reversed_order_margin_predictions": reversed_margin_predictions,
                "feature_cache_contract": cache["contract"],
                "probe_config": probe_config,
                "mean_pool_mlp_control": {
                    "config": mean_pool_mlp_config,
                    "classifier_state_dict": control_classifier_state,
                    "regressor_state_dict": control_regressor_state,
                    "classification_threshold": control_threshold,
                    "regression_target_mean": control_regression_training[
                        "target_mean"
                    ],
                    "regression_target_std": control_regression_training[
                        "target_std"
                    ],
                    "classification_logits": control_logits,
                    "margin_predictions": control_margin_predictions,
                },
            },
        )
        print(
            f"[J2 transformer] seed={seed} "
            f"test AUROC={classification['test']['auroc']:.4f} "
            f"Gold pair={classification_pairs['test']['accuracy']:.4f} "
            f"margin MAE={regression['test']['mae']:.4f} "
            f"mean-MLP pair="
            f"{control_classification_pairs['test']['accuracy']:.4f} "
            f"runtime={duration_s:.1f}s+{control_duration_s:.1f}s",
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
        ("reversed_order", "classification", "test", "auroc"),
        (
            "reversed_order",
            "classification_pairwise",
            "test",
            "accuracy",
        ),
        ("reversed_order", "margin_regression", "test", "mae"),
        ("reversed_order", "margin_pairwise", "test", "accuracy"),
        ("mean_pool_mlp_control", "classification", "test", "auroc"),
        ("mean_pool_mlp_control", "classification", "test", "auprc"),
        (
            "mean_pool_mlp_control",
            "classification",
            "test",
            "balanced_accuracy",
        ),
        ("mean_pool_mlp_control", "classification", "test", "f1"),
        ("mean_pool_mlp_control", "classification", "test", "brier"),
        (
            "mean_pool_mlp_control",
            "classification",
            "test",
            "ece_10_bin",
        ),
        (
            "mean_pool_mlp_control",
            "classification_pairwise",
            "test",
            "accuracy",
        ),
        ("mean_pool_mlp_control", "margin_regression", "test", "mae"),
        ("mean_pool_mlp_control", "margin_regression", "test", "rmse"),
        ("mean_pool_mlp_control", "margin_regression", "test", "spearman"),
        (
            "mean_pool_mlp_control",
            "margin_pairwise",
            "test",
            "accuracy",
        ),
        ("mean_pool_mlp_control", "runtime", "duration_s"),
        ("runtime", "duration_s"),
        ("runtime", "peak_gpu_memory_bytes"),
    ]
    aggregate = _aggregate(per_seed, aggregate_paths)
    mean_pool_mlp_comparison = _mean_pool_mlp_comparison(aggregate)
    linear_result = None
    comparison = None
    if linear_result_path.is_file():
        linear_result = load_json(linear_result_path)
        if linear_result["feature_cache_contract"] != cache["contract"]:
            raise RuntimeError("J1 result used a different feature cache contract")
        comparison = _j1_comparison(linear_result, aggregate)
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
        "probe_type": "temporal_transformer",
        "probe_config": probe_config,
        "trainable_parameters_per_task": trainable_parameters,
        "mean_pool_mlp_trainable_parameters_per_task": (
            control_trainable_parameters
        ),
        "feature_shape": list(temporal_features.shape),
        "normalization": "train_split_token_channel_zscore",
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
        "aggregate": aggregate,
        "j1_result": str(linear_result_path) if linear_result is not None else None,
        "j1_comparison": comparison,
        "mean_pool_mlp_comparison": mean_pool_mlp_comparison,
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(aggregate, indent=2), flush=True)
    if comparison is not None:
        print("[J2 transformer] J2 versus J1", flush=True)
        print(json.dumps(comparison, indent=2), flush=True)
    print("[J2 transformer] Transformer versus mean-pool MLP", flush=True)
    print(json.dumps(mean_pool_mlp_comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
