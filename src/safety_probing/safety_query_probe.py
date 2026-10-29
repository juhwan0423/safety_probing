from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Callable

from .io_utils import load_json, write_json, write_torch_atomic
from .vjepa_linear_probe import (
    _read_jsonl,
    _sha256,
    binary_auroc,
    classification_metrics,
    regression_metrics,
    select_balanced_threshold,
)


CONTINUOUS_TARGETS = {
    "static": "static_mask",
    "observed_state": "observed_state_mask",
    "future_state": "future_state_mask",
    "observed_contact_xy": "observed_contact_xy_mask",
    "future_contact_xy": "future_contact_xy_mask",
    "future_summary": "future_summary_mask",
}


def build_safety_query_decoder(
    *,
    feature_dim: int,
    temporal_tokens: int,
    spatial_height: int,
    spatial_width: int,
    state_dim: int,
    contact_count: int,
    summary_dim: int,
    model_dim: int,
    layers: int,
    heads: int,
    feedforward_dim: int,
    dropout: float,
):
    import torch

    class PhysicalSafetyQueryDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_dim = feature_dim
            self.temporal_tokens = temporal_tokens
            self.spatial_height = spatial_height
            self.spatial_width = spatial_width
            self.contact_count = contact_count
            self.memory_projection = torch.nn.Linear(feature_dim, model_dim)
            self.memory_norm = torch.nn.LayerNorm(model_dim)
            self.memory_time_embedding = torch.nn.Parameter(
                torch.empty(temporal_tokens, model_dim)
            )
            self.memory_space_embedding = torch.nn.Parameter(
                torch.empty(spatial_height * spatial_width, model_dim)
            )
            self.static_queries = torch.nn.Parameter(torch.empty(2, model_dim))
            self.observed_state_queries = torch.nn.Parameter(
                torch.empty(temporal_tokens, model_dim)
            )
            self.future_state_queries = torch.nn.Parameter(
                torch.empty(temporal_tokens, model_dim)
            )
            self.observed_contact_queries = torch.nn.Parameter(
                torch.empty(temporal_tokens, contact_count, model_dim)
            )
            self.future_contact_queries = torch.nn.Parameter(
                torch.empty(temporal_tokens, contact_count, model_dim)
            )
            self.summary_query = torch.nn.Parameter(torch.empty(1, model_dim))
            layer = torch.nn.TransformerDecoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = torch.nn.TransformerDecoder(layer, num_layers=layers)
            self.decoder_norm = torch.nn.LayerNorm(model_dim)
            self.static_head = torch.nn.Linear(model_dim, 1)
            self.observed_state_head = torch.nn.Linear(model_dim, state_dim)
            self.future_state_head = torch.nn.Linear(model_dim, state_dim)
            self.observed_contact_head = torch.nn.Linear(model_dim, 3)
            self.future_contact_head = torch.nn.Linear(model_dim, 3)
            self.summary_head = torch.nn.Linear(model_dim, summary_dim)
            self.tip_head = torch.nn.Linear(model_dim, 1)
            for parameter in (
                self.memory_time_embedding,
                self.memory_space_embedding,
                self.static_queries,
                self.observed_state_queries,
                self.future_state_queries,
                self.observed_contact_queries,
                self.future_contact_queries,
                self.summary_query,
            ):
                torch.nn.init.trunc_normal_(parameter, std=0.02)

        def forward(self, features):
            if features.ndim != 5:
                raise ValueError("Expected features with shape [B,T,H,W,D]")
            expected = (
                self.temporal_tokens,
                self.spatial_height,
                self.spatial_width,
                self.feature_dim,
            )
            if tuple(features.shape[1:]) != expected:
                raise ValueError(
                    f"Expected feature tail {expected}, got {tuple(features.shape[1:])}"
                )
            batch_size = features.shape[0]
            memory = self.memory_projection(features)
            positions = (
                self.memory_time_embedding[:, None, :]
                + self.memory_space_embedding[None, :, :]
            ).reshape(
                self.temporal_tokens
                * self.spatial_height
                * self.spatial_width,
                -1,
            )
            memory = memory.reshape(batch_size, positions.shape[0], -1)
            memory = self.memory_norm(memory + positions)

            query_groups = [
                self.static_queries,
                self.observed_state_queries,
                self.future_state_queries,
                self.observed_contact_queries.reshape(-1, memory.shape[-1]),
                self.future_contact_queries.reshape(-1, memory.shape[-1]),
                self.summary_query,
            ]
            queries = torch.cat(query_groups, dim=0).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            decoded = self.decoder(queries, memory)
            decoded = self.decoder_norm(decoded)

            offset = 0

            def take(count: int):
                nonlocal offset
                result = decoded[:, offset : offset + count]
                offset += count
                return result

            static_tokens = take(2)
            observed_state_tokens = take(self.temporal_tokens)
            future_state_tokens = take(self.temporal_tokens)
            observed_contact_tokens = take(
                self.temporal_tokens * self.contact_count
            ).reshape(batch_size, self.temporal_tokens, self.contact_count, -1)
            future_contact_tokens = take(
                self.temporal_tokens * self.contact_count
            ).reshape(batch_size, self.temporal_tokens, self.contact_count, -1)
            summary_token = take(1)
            if offset != decoded.shape[1]:
                raise RuntimeError("Safety Query layout did not consume all queries")

            observed_contact = self.observed_contact_head(
                observed_contact_tokens
            )
            future_contact = self.future_contact_head(future_contact_tokens)
            return {
                "static": self.static_head(static_tokens).squeeze(-1),
                "observed_state": self.observed_state_head(
                    observed_state_tokens
                ),
                "future_state": self.future_state_head(future_state_tokens),
                "observed_contact_logits": observed_contact[..., 0],
                "observed_contact_xy": observed_contact[..., 1:],
                "future_contact_logits": future_contact[..., 0],
                "future_contact_xy": future_contact[..., 1:],
                "future_summary": self.summary_head(summary_token).squeeze(1),
                "future_tip_logits": self.tip_head(summary_token).squeeze((1, 2)),
            }

    return PhysicalSafetyQueryDecoder()


def _masked_stats(values, mask, train_indices):
    import torch

    values = values[train_indices]
    mask = mask[train_indices]
    if mask.ndim == values.ndim - 1:
        mask = mask.unsqueeze(-1).expand_as(values)
    reduce_dims = tuple(range(values.ndim - 1))
    weights = mask.to(dtype=torch.float64)
    values64 = values.to(dtype=torch.float64)
    count = weights.sum(dim=reduce_dims).clamp_min(1.0)
    mean = (values64 * weights).sum(dim=reduce_dims) / count
    variance = (
        ((values64 - mean) ** 2) * weights
    ).sum(dim=reduce_dims) / count
    return mean.float(), variance.sqrt().clamp_min(1e-6).float()


def fit_normalizers(targets, train_indices) -> dict[str, dict[str, Any]]:
    return {
        key: dict(
            zip(
                ("mean", "std"),
                _masked_stats(targets[key], targets[mask_key], train_indices),
                strict=True,
            )
        )
        for key, mask_key in CONTINUOUS_TARGETS.items()
    }


def _normalize(values, normalizer):
    return (values - normalizer["mean"]) / normalizer["std"]


def _denormalize(values, normalizer):
    return values * normalizer["std"] + normalizer["mean"]


def _masked_mse(prediction, target, mask):
    import torch

    if mask.ndim == target.ndim - 1:
        mask = mask.unsqueeze(-1).expand_as(target)
    mask = mask.to(dtype=target.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def safety_query_losses(
    outputs,
    targets,
    normalizers,
    loss_weights,
    *,
    tip_pos_weight,
):
    import torch

    losses = {}
    for output_key, target_key in (
        ("static", "static"),
        ("observed_state", "observed_state"),
        ("future_state", "future_state"),
        ("observed_contact_xy", "observed_contact_xy"),
        ("future_contact_xy", "future_contact_xy"),
        ("future_summary", "future_summary"),
    ):
        mask_key = CONTINUOUS_TARGETS[target_key]
        normalized_target = _normalize(
            targets[target_key], normalizers[target_key]
        )
        losses[target_key] = _masked_mse(
            outputs[output_key], normalized_target, targets[mask_key]
        )

    losses["observed_contact_presence"] = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["observed_contact_logits"],
            targets["observed_contact"].float(),
        )
    )
    losses["future_contact_presence"] = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["future_contact_logits"],
            targets["future_contact"].float(),
        )
    )
    tip_mask = targets["future_tip_mask"]
    if bool(tip_mask.any()):
        losses["future_tip"] = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["future_tip_logits"][tip_mask],
            targets["future_tip"][tip_mask],
            pos_weight=tip_pos_weight,
        )
    else:
        losses["future_tip"] = outputs["future_tip_logits"].sum() * 0

    weighted_terms = {
        "static": losses["static"] * float(loss_weights["static"]),
        "observed_state": losses["observed_state"]
        * float(loss_weights["observed_state"]),
        "future_state": losses["future_state"]
        * float(loss_weights["future_state"]),
        "observed_contact_presence": losses["observed_contact_presence"]
        * float(loss_weights["observed_contact_presence"]),
        "observed_contact_position": losses["observed_contact_xy"]
        * float(loss_weights["observed_contact_position"]),
        "future_contact_presence": losses["future_contact_presence"]
        * float(loss_weights["future_contact_presence"]),
        "future_contact_position": losses["future_contact_xy"]
        * float(loss_weights["future_contact_position"]),
        "future_summary": losses["future_summary"]
        * float(loss_weights["future_summary"]),
        "future_tip": losses["future_tip"]
        * float(loss_weights["future_tip"]),
    }
    losses["total"] = sum(weighted_terms.values())
    return losses


def _target_batch(targets, indices, device):
    return {
        key: value[indices].to(device)
        for key, value in targets.items()
        if hasattr(value, "shape") and key != "features"
    }


def _model_factory(features, targets, config):
    decoder = config["query_decoder"]
    return build_safety_query_decoder(
        feature_dim=features.shape[-1],
        temporal_tokens=features.shape[1],
        spatial_height=features.shape[2],
        spatial_width=features.shape[3],
        state_dim=targets["observed_state"].shape[-1],
        contact_count=targets["observed_contact"].shape[-1],
        summary_dim=targets["future_summary"].shape[-1],
        model_dim=int(decoder["model_dim"]),
        layers=int(decoder["layers"]),
        heads=int(decoder["heads"]),
        feedforward_dim=int(decoder["feedforward_dim"]),
        dropout=float(decoder["dropout"]),
    )


def _indices(records, split: str):
    import torch

    return torch.tensor(
        [index for index, record in enumerate(records) if record["split"] == split],
        dtype=torch.long,
    )


def _batches(indices, batch_size: int, generator=None):
    import torch

    if generator is None:
        order = torch.arange(len(indices))
    else:
        order = torch.randperm(len(indices), generator=generator)
    for start in range(0, len(indices), batch_size):
        yield indices[order[start : start + batch_size]]


def _predict(model, features, indices, batch_size: int, device: str):
    import torch

    collected: dict[str, list[Any]] = {}
    model.eval()
    with torch.inference_mode():
        for batch in _batches(indices, batch_size):
            output = model(features[batch].float().to(device))
            for key, value in output.items():
                collected.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def _state_dict_cpu(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_one_seed(
    features,
    targets,
    records,
    config,
    normalizers,
    *,
    seed: int,
    device: str,
    max_epochs: int,
    patience: int,
):
    import torch

    training = config["training"]
    batch_size = int(training["batch_size"])
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(seed)
    model = _model_factory(features, targets, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_indices = _indices(records, "train")
    val_indices = _indices(records, "val")
    binary_train = targets["future_tip_mask"][train_indices]
    train_labels = targets["future_tip"][train_indices][binary_train]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    tip_pos_weight = torch.tensor(
        negatives / positives, dtype=torch.float32, device=device
    )
    device_normalizers = {
        key: {
            name: value.to(device) for name, value in normalizer.items()
        }
        for key, normalizer in normalizers.items()
    }
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for batch in _batches(train_indices, batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            output = model(features[batch].float().to(device))
            target = _target_batch(targets, batch, device)
            losses = safety_query_losses(
                output,
                target,
                device_normalizers,
                training["loss_weights"],
                tip_pos_weight=tip_pos_weight,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            optimizer.step()
            train_loss_sum += float(losses["total"].detach()) * len(batch)
            train_examples += len(batch)

        val_outputs = _predict(
            model, features, val_indices, batch_size, device
        )
        val_target = _target_batch(targets, val_indices, "cpu")
        val_losses = safety_query_losses(
            val_outputs,
            val_target,
            normalizers,
            training["loss_weights"],
            tip_pos_weight=tip_pos_weight.cpu(),
        )
        val_loss = float(val_losses["total"])
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
        elif epoch - best_epoch >= patience:
            break
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[Safety Query] seed={seed} epoch={epoch} "
                f"train={train_loss_sum / train_examples:.4f} "
                f"val={val_loss:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("Safety Query training produced no checkpoint")
    model.load_state_dict(best_state)
    all_indices = torch.arange(len(records), dtype=torch.long)
    predictions = _predict(model, features, all_indices, batch_size, device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime = {
        "duration_s": time.monotonic() - started,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        ),
    }
    training_report = {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "best_validation_total_normalized_loss": best_loss,
        "future_tip_positive_weight": float(tip_pos_weight),
    }
    return predictions, best_state, training_report, runtime


def _continuous_report(
    prediction,
    target,
    mask,
    channel_names,
    baseline_mean,
):
    if mask.ndim == target.ndim - 1:
        mask = mask.unsqueeze(-1).expand_as(target)
    report = {}
    for channel, name in enumerate(channel_names):
        channel_mask = mask[..., channel]
        actual = target[..., channel][channel_mask]
        predicted = prediction[..., channel][channel_mask]
        baseline = baseline_mean[channel].expand_as(actual)
        model_metrics = regression_metrics(actual, predicted)
        baseline_metrics = regression_metrics(actual, baseline)
        report[name] = {
            **model_metrics,
            "train_mean_baseline_mae": baseline_metrics["mae"],
            "train_mean_baseline_rmse": baseline_metrics["rmse"],
            "mae_improvement_over_train_mean": (
                baseline_metrics["mae"] - model_metrics["mae"]
            ),
        }
    return report


def _contact_report(logits, xy_prediction, target_contact, target_xy, xy_mask):
    import torch

    predicted_contact = logits >= 0
    target_contact = target_contact.bool()
    tp = int((predicted_contact & target_contact).sum())
    tn = int((~predicted_contact & ~target_contact).sum())
    fp = int((predicted_contact & ~target_contact).sum())
    fn = int((~predicted_contact & target_contact).sum())
    expanded_mask = xy_mask.unsqueeze(-1).expand_as(target_xy)
    errors = (xy_prediction - target_xy).abs()[expanded_mask]
    return {
        "count": target_contact.numel(),
        "active_count": int(target_contact.sum()),
        "accuracy": (tp + tn) / target_contact.numel(),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "active_position_coordinate_mae_m": float(errors.mean()),
    }


def _convex_hull(points: list[tuple[float, float]]):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area_centroid(points):
    if len(points) < 3:
        return None
    twice_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for index, first in enumerate(points):
        second = points[(index + 1) % len(points)]
        cross = first[0] * second[1] - second[0] * first[1]
        twice_area += cross
        centroid_x += (first[0] + second[0]) * cross
        centroid_y += (first[1] + second[1]) * cross
    if abs(twice_area) < 1e-12:
        return None
    return (
        abs(twice_area) / 2,
        centroid_x / (3 * twice_area),
        centroid_y / (3 * twice_area),
    )


def _support_polygon_report(logits, predicted_xy, target_contact, target_xy):
    import torch

    predicted_contact = logits >= 0
    area_errors = []
    centroid_errors = []
    eligible = 0
    both_valid = 0
    flat_shape = target_contact.shape[:-1]
    for flat_index in range(math.prod(flat_shape)):
        actual_mask = target_contact.reshape(-1, target_contact.shape[-1])[
            flat_index
        ]
        predicted_mask = predicted_contact.reshape(
            -1, predicted_contact.shape[-1]
        )[flat_index]
        actual_points_tensor = target_xy.reshape(-1, target_xy.shape[-2], 2)[
            flat_index
        ][actual_mask]
        predicted_points_tensor = predicted_xy.reshape(
            -1, predicted_xy.shape[-2], 2
        )[flat_index][predicted_mask]
        actual = _polygon_area_centroid(
            _convex_hull([tuple(point.tolist()) for point in actual_points_tensor])
        )
        if actual is None:
            continue
        eligible += 1
        predicted = _polygon_area_centroid(
            _convex_hull(
                [tuple(point.tolist()) for point in predicted_points_tensor]
            )
        )
        if predicted is None:
            continue
        both_valid += 1
        area_errors.append(abs(predicted[0] - actual[0]))
        centroid_errors.append(
            math.hypot(predicted[1] - actual[1], predicted[2] - actual[2])
        )
    return {
        "actual_valid_polygon_count": eligible,
        "both_valid_polygon_count": both_valid,
        "predicted_polygon_coverage": both_valid / eligible if eligible else None,
        "area_mae_m2": statistics.fmean(area_errors) if area_errors else None,
        "centroid_error_mae_m": (
            statistics.fmean(centroid_errors) if centroid_errors else None
        ),
    }


def _nearest_level_accuracy(prediction, target, levels):
    import torch

    levels = torch.tensor(levels, dtype=prediction.dtype)
    predicted_level = levels[(prediction[:, None] - levels[None, :]).abs().argmin(1)]
    return float((predicted_level == target).float().mean())


def _pair_property_report(
    predictions,
    targets,
    records,
    pair_rows,
    run_rows,
    split: str,
):
    condition_to_index = {
        str(record["condition_id"]): index
        for index, record in enumerate(records)
    }
    runs = {
        (str(row["pair_id"]), str(row["run_label"])): row for row in run_rows
    }
    output = {}
    for intervention, channel in (("load_mass", 0), ("com_shift", 1)):
        rows = [
            row
            for row in pair_rows
            if row["splits"]["main"] == split
            and row["intervention_type"] == intervention
        ]
        correct = 0.0
        delta_errors = []
        by_tier: dict[str, list[float]] = {}
        for pair in rows:
            pair_id = str(pair["pair_id"])
            a = condition_to_index[
                str(runs[(pair_id, "A")]["condition_id"])
            ]
            b = condition_to_index[
                str(runs[(pair_id, "B")]["condition_id"])
            ]
            actual_delta = float(targets[b, channel] - targets[a, channel])
            predicted_delta = float(
                predictions[b, channel] - predictions[a, channel]
            )
            credit = 1.0 if actual_delta * predicted_delta > 0 else 0.0
            if predicted_delta == 0:
                credit = 0.5
            correct += credit
            delta_errors.append(abs(predicted_delta - actual_delta))
            by_tier.setdefault(str(pair["quality_tier"]), []).append(credit)
        output[intervention] = {
            "pair_count": len(rows),
            "direction_accuracy": correct / len(rows) if rows else None,
            "delta_mae": statistics.fmean(delta_errors) if rows else None,
            "by_quality_tier": {
                tier: {
                    "pair_count": len(values),
                    "direction_accuracy": statistics.fmean(values),
                }
                for tier, values in sorted(by_tier.items())
            },
        }
    return output


def _pair_margin_report(
    predictions,
    targets,
    records,
    pair_rows,
    run_rows,
    split: str,
):
    condition_to_index = {
        str(record["condition_id"]): index
        for index, record in enumerate(records)
    }
    runs = {
        (str(row["pair_id"]), str(row["run_label"])): row for row in run_rows
    }
    by_tier: dict[str, list[float]] = {}
    for pair in pair_rows:
        if pair["splits"]["main"] != split:
            continue
        pair_id = str(pair["pair_id"])
        a = condition_to_index[str(runs[(pair_id, "A")]["condition_id"])]
        b = condition_to_index[str(runs[(pair_id, "B")]["condition_id"])]
        actual_delta = float(targets[b] - targets[a])
        predicted_delta = float(predictions[b] - predictions[a])
        if abs(actual_delta) < 1e-12:
            credit = 0.5 if abs(predicted_delta) < 1e-12 else 0.0
        else:
            credit = 1.0 if actual_delta * predicted_delta > 0 else 0.0
            if predicted_delta == 0:
                credit = 0.5
        by_tier.setdefault(str(pair["quality_tier"]), []).append(credit)
    return {
        tier: {
            "pair_count": len(values),
            "ordering_accuracy": statistics.fmean(values),
        }
        for tier, values in sorted(by_tier.items())
    }


def evaluate_predictions(
    predictions,
    targets,
    records,
    channels,
    normalizers,
    pair_rows,
    run_rows,
):
    import torch

    denormalized = {
        key: _denormalize(predictions[key], normalizers[key])
        for key in CONTINUOUS_TARGETS
    }
    split_reports = {}
    for split in ("train", "val", "test"):
        indices = _indices(records, split)
        report = {
            "static": _continuous_report(
                denormalized["static"][indices],
                targets["static"][indices],
                targets["static_mask"][indices],
                channels["static"],
                normalizers["static"]["mean"],
            ),
            "observed_state": _continuous_report(
                denormalized["observed_state"][indices],
                targets["observed_state"][indices],
                targets["observed_state_mask"][indices],
                channels["state"],
                normalizers["observed_state"]["mean"],
            ),
            "future_state": _continuous_report(
                denormalized["future_state"][indices],
                targets["future_state"][indices],
                targets["future_state_mask"][indices],
                channels["state"],
                normalizers["future_state"]["mean"],
            ),
            "future_summary": _continuous_report(
                denormalized["future_summary"][indices],
                targets["future_summary"][indices],
                targets["future_summary_mask"][indices],
                channels["future_summary"],
                normalizers["future_summary"]["mean"],
            ),
        }
        for scope in ("observed", "future"):
            xy_key = f"{scope}_contact_xy"
            logits_key = f"{scope}_contact_logits"
            contact_key = f"{scope}_contact"
            mask_key = f"{scope}_contact_xy_mask"
            xy = denormalized[xy_key][indices]
            report[f"{scope}_contacts"] = {
                **_contact_report(
                    predictions[logits_key][indices],
                    xy,
                    targets[contact_key][indices],
                    targets[xy_key][indices],
                    targets[mask_key][indices],
                ),
                "support_polygon": _support_polygon_report(
                    predictions[logits_key][indices],
                    xy,
                    targets[contact_key][indices],
                    targets[xy_key][indices],
                ),
            }
        binary_mask = targets["future_tip_mask"][indices]
        labels = targets["future_tip"][indices][binary_mask]
        logits = predictions["future_tip_logits"][indices][binary_mask]
        if split == "val":
            val_threshold = select_balanced_threshold(labels, torch.sigmoid(logits))
        split_reports[split] = report

    for split in ("train", "val", "test"):
        indices = _indices(records, split)
        binary_mask = targets["future_tip_mask"][indices]
        split_reports[split]["future_tip"] = classification_metrics(
            targets["future_tip"][indices][binary_mask],
            predictions["future_tip_logits"][indices][binary_mask],
            val_threshold,
        )

    static_predictions = denormalized["static"]
    split_reports["test"]["static"]["load_mass_kg"][
        "nearest_level_accuracy"
    ] = _nearest_level_accuracy(
        static_predictions[_indices(records, "test"), 0],
        targets["static"][_indices(records, "test"), 0],
        [240.0, 360.0, 480.0, 600.0, 720.0],
    )
    split_reports["test"]["static"]["load_com_lateral_m"][
        "nearest_level_accuracy"
    ] = _nearest_level_accuracy(
        static_predictions[_indices(records, "test"), 1],
        targets["static"][_indices(records, "test"), 1],
        [-0.25, 0.0, 0.25],
    )
    pair_reports = {
        split: {
            "hidden_property_delta": _pair_property_report(
                static_predictions,
                targets["static"],
                records,
                pair_rows,
                run_rows,
                split,
            ),
            "zmp_summary_ordering_by_tier": _pair_margin_report(
                denormalized["future_summary"][:, 0],
                targets["future_summary"][:, 0],
                records,
                pair_rows,
                run_rows,
                split,
            ),
            "com_summary_ordering_by_tier": _pair_margin_report(
                denormalized["future_summary"][:, 1],
                targets["future_summary"][:, 1],
                records,
                pair_rows,
                run_rows,
                split,
            ),
        }
        for split in ("train", "val", "test")
    }
    return {
        "splits": split_reports,
        "pairs": pair_reports,
        "validation_tip_threshold": val_threshold,
    }, denormalized


def _aggregate(per_seed, paths):
    output = {}
    for path in paths:
        values = []
        for item in per_seed:
            value: Any = item
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train physical Safety Query decoders on frozen V-JEPA tokens."
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
        "--output-root",
        type=Path,
        default=Path("outputs/safety_query/part_01"),
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
    feature_path = args.feature_cache.resolve()
    target_path = args.target_cache.resolve()
    pairs_path = args.pairs_manifest.resolve()
    runs_path = args.runs_manifest.resolve()
    output_root = args.output_root.resolve()
    config = load_json(experiment_path)
    feature_cache = torch.load(
        feature_path, map_location="cpu", mmap=True, weights_only=True
    )
    targets = torch.load(
        target_path, map_location="cpu", mmap=True, weights_only=True
    )
    features = feature_cache["features"]
    if features.ndim != 5:
        raise RuntimeError("Safety Query requires [N,T,H,W,D] features")
    feature_ids = feature_cache["contract"]["condition_ids"]
    target_ids = targets["contract"]["condition_ids"]
    if feature_ids != target_ids:
        raise RuntimeError("Feature and target condition order differs")
    records = targets["records"]
    if len(records) != features.shape[0]:
        raise RuntimeError("Safety Query feature/target count mismatch")

    train_indices = _indices(records, "train")
    normalizers = fit_normalizers(targets, train_indices)
    reference = _model_factory(features, targets, config)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in reference.parameters()
        if parameter.requires_grad
    )
    del reference
    training = config["training"]
    seeds = list(args.seed or training["seeds"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    patience = int(args.patience or training["patience"])
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("Epoch and patience values must be positive")
    pair_rows = _read_jsonl(pairs_path)
    run_rows = _read_jsonl(runs_path)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[Safety Query] features={tuple(features.shape)} "
        f"queries={sum(config['query_decoder']['query_layout'].values())} "
        f"parameters={trainable_parameters:,} seeds={seeds}",
        flush=True,
    )

    per_seed = []
    for seed in seeds:
        predictions, state, training_report, runtime = train_one_seed(
            features,
            targets,
            records,
            config,
            normalizers,
            seed=seed,
            device=args.device,
            max_epochs=max_epochs,
            patience=patience,
        )
        evaluation, denormalized = evaluate_predictions(
            predictions,
            targets,
            records,
            targets["channels"],
            normalizers,
            pair_rows,
            run_rows,
        )
        seed_result = {
            "seed": seed,
            "training": training_report,
            "runtime": runtime,
            **evaluation,
        }
        per_seed.append(seed_result)
        write_torch_atomic(
            output_root / f"seed_{seed:04d}.pt",
            {
                "seed": seed,
                "model_state_dict": state,
                "normalizers": normalizers,
                "predictions_normalized": predictions,
                "predictions_physical": denormalized,
                "feature_contract": feature_cache["contract"],
                "target_contract": targets["contract"],
                "experiment_sha256": _sha256(experiment_path),
            },
        )
        test = evaluation["splits"]["test"]
        print(
            f"[Safety Query] seed={seed} "
            f"mass_mae={test['static']['load_mass_kg']['mae']:.2f}kg "
            f"com_mae={test['static']['load_com_lateral_m']['mae']:.4f}m "
            f"zmp_mae={test['future_state']['zmp_body_x']['mae']:.4f}m "
            f"tip_auc={test['future_tip']['auroc']:.4f}",
            flush=True,
        )

    paths = [
        ("splits", "test", "static", "load_mass_kg", "mae"),
        (
            "splits",
            "test",
            "static",
            "load_mass_kg",
            "nearest_level_accuracy",
        ),
        ("splits", "test", "static", "load_com_lateral_m", "mae"),
        (
            "splits",
            "test",
            "static",
            "load_com_lateral_m",
            "nearest_level_accuracy",
        ),
        ("splits", "test", "observed_state", "roll_deg", "mae"),
        ("splits", "test", "observed_state", "zmp_body_x", "mae"),
        ("splits", "test", "observed_state", "zmp_body_y", "mae"),
        ("splits", "test", "future_state", "roll_deg", "mae"),
        ("splits", "test", "future_state", "zmp_body_x", "mae"),
        ("splits", "test", "future_state", "zmp_body_y", "mae"),
        ("splits", "test", "observed_contacts", "f1"),
        ("splits", "test", "future_contacts", "f1"),
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
        ("runtime", "duration_s"),
        ("runtime", "peak_gpu_memory_bytes"),
    ]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "experiment_sha256": _sha256(experiment_path),
        "feature_cache": str(feature_path),
        "feature_contract": feature_cache["contract"],
        "target_cache": str(target_path),
        "target_contract": targets["contract"],
        "foundation_parameters_frozen": True,
        "query_decoder_trainable_parameters": trainable_parameters,
        "counts": {
            split: len(_indices(records, split))
            for split in ("train", "val", "test")
        },
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed, paths),
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
