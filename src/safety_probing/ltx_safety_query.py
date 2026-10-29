from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from .io_utils import load_json, write_json, write_torch_atomic
from .safety_query_controls import (
    CONTROL_NAMES,
    IntervenedFeatures,
    split_preserving_permutation,
)
from .safety_query_probe import (
    _aggregate,
    _batches,
    _contact_report,
    _continuous_report,
    _denormalize,
    _indices,
    _masked_mse,
    _masked_stats,
    _nearest_level_accuracy,
    _normalize,
    _pair_margin_report,
    _pair_property_report,
    _predict,
    _state_dict_cpu,
    _support_polygon_report,
)
from .vjepa_linear_probe import (
    _read_jsonl,
    _sha256,
    classification_metrics,
    select_balanced_threshold,
)


CONTINUOUS_TARGETS = {
    "static": "static_mask",
    "future_state": "future_state_mask",
    "future_contact_xy": "future_contact_xy_mask",
    "future_summary": "future_summary_mask",
}


def build_ltx_safety_query_decoder(
    *,
    feature_dim: int,
    memory_temporal_tokens: int,
    spatial_height: int,
    spatial_width: int,
    target_temporal_tokens: int,
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

    class LTXPhysicalSafetyQueryDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_dim = feature_dim
            self.memory_temporal_tokens = memory_temporal_tokens
            self.spatial_height = spatial_height
            self.spatial_width = spatial_width
            self.target_temporal_tokens = target_temporal_tokens
            self.contact_count = contact_count
            self.memory_projection = torch.nn.Linear(feature_dim, model_dim)
            self.memory_norm = torch.nn.LayerNorm(model_dim)
            self.memory_time_embedding = torch.nn.Parameter(
                torch.empty(memory_temporal_tokens, model_dim)
            )
            self.memory_space_embedding = torch.nn.Parameter(
                torch.empty(spatial_height * spatial_width, model_dim)
            )
            self.static_queries = torch.nn.Parameter(torch.empty(2, model_dim))
            self.future_state_queries = torch.nn.Parameter(
                torch.empty(target_temporal_tokens, model_dim)
            )
            self.future_contact_queries = torch.nn.Parameter(
                torch.empty(target_temporal_tokens, contact_count, model_dim)
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
            self.future_state_head = torch.nn.Linear(model_dim, state_dim)
            self.future_contact_head = torch.nn.Linear(model_dim, 3)
            self.summary_head = torch.nn.Linear(model_dim, summary_dim)
            self.tip_head = torch.nn.Linear(model_dim, 1)
            for parameter in (
                self.memory_time_embedding,
                self.memory_space_embedding,
                self.static_queries,
                self.future_state_queries,
                self.future_contact_queries,
                self.summary_query,
            ):
                torch.nn.init.trunc_normal_(parameter, std=0.02)

        def forward(self, features):
            expected = (
                self.memory_temporal_tokens,
                self.spatial_height,
                self.spatial_width,
                self.feature_dim,
            )
            if features.ndim != 5 or tuple(features.shape[1:]) != expected:
                raise ValueError(
                    f"Expected LTX features [B,{','.join(map(str, expected))}], "
                    f"got {tuple(features.shape)}"
                )
            batch_size = features.shape[0]
            memory = self.memory_projection(features)
            positions = (
                self.memory_time_embedding[:, None, :]
                + self.memory_space_embedding[None, :, :]
            ).reshape(-1, memory.shape[-1])
            memory = memory.reshape(batch_size, len(positions), -1)
            memory = self.memory_norm(memory + positions)
            queries = torch.cat(
                [
                    self.static_queries,
                    self.future_state_queries,
                    self.future_contact_queries.reshape(-1, memory.shape[-1]),
                    self.summary_query,
                ],
                dim=0,
            ).unsqueeze(0).expand(batch_size, -1, -1)
            decoded = self.decoder(queries, memory)
            decoded = self.decoder_norm(decoded)
            offset = 0

            def take(count: int):
                nonlocal offset
                result = decoded[:, offset : offset + count]
                offset += count
                return result

            static = take(2)
            future_state = take(self.target_temporal_tokens)
            future_contact = take(
                self.target_temporal_tokens * self.contact_count
            ).reshape(
                batch_size,
                self.target_temporal_tokens,
                self.contact_count,
                -1,
            )
            summary = take(1)
            if offset != decoded.shape[1]:
                raise RuntimeError("LTX Safety Query layout did not consume all tokens")
            contact_output = self.future_contact_head(future_contact)
            return {
                "static": self.static_head(static).squeeze(-1),
                "future_state": self.future_state_head(future_state),
                "future_contact_logits": contact_output[..., 0],
                "future_contact_xy": contact_output[..., 1:],
                "future_summary": self.summary_head(summary).squeeze(1),
                "future_tip_logits": self.tip_head(summary).squeeze((1, 2)),
            }

    return LTXPhysicalSafetyQueryDecoder()


def fit_ltx_normalizers(targets, train_indices):
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


def ltx_safety_query_losses(
    outputs,
    targets,
    normalizers,
    loss_weights,
    *,
    tip_pos_weight,
):
    import torch

    losses = {}
    for key in CONTINUOUS_TARGETS:
        losses[key] = _masked_mse(
            outputs[key],
            _normalize(targets[key], normalizers[key]),
            targets[CONTINUOUS_TARGETS[key]],
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
    losses["total"] = (
        losses["static"] * float(loss_weights["static"])
        + losses["future_state"] * float(loss_weights["future_state"])
        + losses["future_contact_presence"]
        * float(loss_weights["future_contact_presence"])
        + losses["future_contact_xy"]
        * float(loss_weights["future_contact_position"])
        + losses["future_summary"] * float(loss_weights["future_summary"])
        + losses["future_tip"] * float(loss_weights["future_tip"])
    )
    return losses


def _model_factory(features, targets, config):
    decoder = config["query_decoder"]
    return build_ltx_safety_query_decoder(
        feature_dim=features.shape[-1],
        memory_temporal_tokens=features.shape[1],
        spatial_height=features.shape[2],
        spatial_width=features.shape[3],
        target_temporal_tokens=targets["future_state"].shape[1],
        state_dim=targets["future_state"].shape[-1],
        contact_count=targets["future_contact"].shape[-1],
        summary_dim=targets["future_summary"].shape[-1],
        model_dim=int(decoder["model_dim"]),
        layers=int(decoder["layers"]),
        heads=int(decoder["heads"]),
        feedforward_dim=int(decoder["feedforward_dim"]),
        dropout=float(decoder["dropout"]),
    )


def _target_batch(targets, indices, device):
    return {
        key: value[indices].to(device)
        for key, value in targets.items()
        if hasattr(value, "shape")
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
    train_binary_mask = targets["future_tip_mask"][train_indices]
    train_labels = targets["future_tip"][train_indices][train_binary_mask]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    tip_pos_weight = torch.tensor(
        negatives / positives, dtype=torch.float32, device=device
    )
    device_normalizers = {
        key: {name: value.to(device) for name, value in stats.items()}
        for key, stats in normalizers.items()
    }
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    started = time.monotonic()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for batch in _batches(train_indices, batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            output = model(features[batch].float().to(device))
            target = _target_batch(targets, batch, device)
            losses = ltx_safety_query_losses(
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
        val_output = _predict(model, features, val_indices, batch_size, device)
        val_losses = ltx_safety_query_losses(
            val_output,
            _target_batch(targets, val_indices, "cpu"),
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
                f"[LTX Safety Query] seed={seed} epoch={epoch} "
                f"train={train_loss_sum / train_examples:.4f} val={val_loss:.4f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("LTX Safety Query training produced no checkpoint")
    model.load_state_dict(best_state)
    predictions = _predict(
        model,
        features,
        torch.arange(len(records), dtype=torch.long),
        batch_size,
        device,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime = {
        "duration_s": time.monotonic() - started,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        ),
    }
    report = {
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "best_real_validation_total_normalized_loss": best_loss,
        "future_tip_positive_weight": float(tip_pos_weight),
    }
    return model, predictions, best_state, report, runtime


def evaluate_ltx_predictions(
    predictions,
    targets,
    records,
    channels,
    normalizers,
    pair_rows,
    run_rows,
    *,
    threshold: float | None = None,
):
    import torch

    denormalized = {
        key: _denormalize(predictions[key], normalizers[key])
        for key in CONTINUOUS_TARGETS
    }
    available_splits = [
        split
        for split in ("train", "val", "test")
        if any(record["split"] == split for record in records)
    ]
    if threshold is None:
        if "val" not in available_splits:
            raise RuntimeError("Validation split is required to select threshold")
        val_indices = _indices(records, "val")
        val_mask = targets["future_tip_mask"][val_indices]
        threshold = select_balanced_threshold(
            targets["future_tip"][val_indices][val_mask],
            torch.sigmoid(predictions["future_tip_logits"][val_indices][val_mask]),
        )
    split_reports = {}
    for split in available_splits:
        indices = _indices(records, split)
        contact_xy = denormalized["future_contact_xy"][indices]
        report = {
            "static": _continuous_report(
                denormalized["static"][indices],
                targets["static"][indices],
                targets["static_mask"][indices],
                channels["static"],
                normalizers["static"]["mean"],
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
            "future_contacts": {
                **_contact_report(
                    predictions["future_contact_logits"][indices],
                    contact_xy,
                    targets["future_contact"][indices],
                    targets["future_contact_xy"][indices],
                    targets["future_contact_xy_mask"][indices],
                ),
                "support_polygon": _support_polygon_report(
                    predictions["future_contact_logits"][indices],
                    contact_xy,
                    targets["future_contact"][indices],
                    targets["future_contact_xy"][indices],
                ),
            },
        }
        binary_mask = targets["future_tip_mask"][indices]
        report["future_tip"] = classification_metrics(
            targets["future_tip"][indices][binary_mask],
            predictions["future_tip_logits"][indices][binary_mask],
            threshold,
        )
        split_reports[split] = report
    if "test" in available_splits:
        test_indices = _indices(records, "test")
        split_reports["test"]["static"]["load_mass_kg"][
            "nearest_level_accuracy"
        ] = _nearest_level_accuracy(
            denormalized["static"][test_indices, 0],
            targets["static"][test_indices, 0],
            [240.0, 360.0, 480.0, 600.0, 720.0],
        )
        split_reports["test"]["static"]["load_com_lateral_m"][
            "nearest_level_accuracy"
        ] = _nearest_level_accuracy(
            denormalized["static"][test_indices, 1],
            targets["static"][test_indices, 1],
            [-0.25, 0.0, 0.25],
        )
    pair_reports = {}
    for split in available_splits:
        pair_reports[split] = {
            "hidden_property_delta": _pair_property_report(
                denormalized["static"],
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
    return {
        "splits": split_reports,
        "pairs": pair_reports,
        "validation_tip_threshold": threshold,
    }, denormalized


def subset_targets(targets, indices):
    return {
        key: value[indices]
        for key, value in targets.items()
        if hasattr(value, "shape")
    } | {"channels": targets["channels"]}


def aggregate_rollout_predictions(predictions_by_seed):
    import torch

    if not predictions_by_seed:
        raise ValueError("No rollout predictions to aggregate")
    output = {}
    for key in predictions_by_seed[0]:
        values = torch.stack([item[key] for item in predictions_by_seed])
        if key.endswith("logits"):
            probability = torch.sigmoid(values).mean(0).clamp(1e-6, 1 - 1e-6)
            output[key] = torch.logit(probability)
        else:
            output[key] = values.mean(0)
    return output


def _generated_layout(rollout_cache, target_records):
    condition_to_target = {
        str(record["condition_id"]): index
        for index, record in enumerate(target_records)
    }
    by_seed: dict[int, list[int]] = {}
    conditions_by_seed: dict[int, list[str]] = {}
    for index, record in enumerate(rollout_cache["records"]):
        seed = int(record["rollout_seed"])
        by_seed.setdefault(seed, []).append(index)
        conditions_by_seed.setdefault(seed, []).append(str(record["condition_id"]))
    seeds = sorted(by_seed)
    reference = conditions_by_seed[seeds[0]]
    for seed in seeds[1:]:
        if conditions_by_seed[seed] != reference:
            raise RuntimeError("Rollout seeds do not cover the same condition order")
    target_indices = [condition_to_target[condition_id] for condition_id in reference]
    records = [target_records[index] for index in target_indices]
    return seeds, by_seed, target_indices, records


def _predict_rollouts(
    model,
    features,
    by_seed,
    *,
    batch_size: int,
    device: str,
    reverse_time: bool = False,
):
    import torch

    output = []
    for seed in sorted(by_seed):
        seed_features = features[by_seed[seed]]
        if reverse_time:
            seed_features = seed_features.flip(1)
        predictions = _predict(
            model,
            seed_features,
            torch.arange(len(seed_features)),
            batch_size,
            device,
        )
        output.append(predictions)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a physical query on real LTX latents and evaluate rollouts."
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
    parser.add_argument(
        "--rollout-cache",
        type=Path,
        default=Path("artifacts/features/ltx_2_3_part_01_rollout_future.pt"),
    )
    parser.add_argument(
        "--target-cache",
        type=Path,
        default=Path("artifacts/targets/ltx_safety_query_part_01.pt"),
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
        default=Path("outputs/ltx_safety_query/part_01_prompt_v1"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--skip-controls", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch

    experiment_path = args.experiment.resolve()
    real_path = args.real_cache.resolve()
    rollout_path = args.rollout_cache.resolve()
    target_path = args.target_cache.resolve()
    output_root = args.output_root.resolve()
    config = load_json(experiment_path)
    real_cache = torch.load(
        real_path, map_location="cpu", mmap=True, weights_only=True
    )
    rollout_cache = torch.load(
        rollout_path, map_location="cpu", mmap=True, weights_only=True
    )
    targets = torch.load(
        target_path, map_location="cpu", mmap=True, weights_only=True
    )
    real_features = real_cache["features"]
    real_records = targets["records"]
    if real_cache["contract"]["condition_ids"] != targets["contract"][
        "condition_ids"
    ]:
        raise RuntimeError("LTX real feature and target condition order differs")
    if real_cache["contract"]["experiment_sha256"] != _sha256(experiment_path):
        raise RuntimeError("LTX real cache experiment hash differs")
    if rollout_cache["contract"]["experiment_sha256"] != _sha256(experiment_path):
        raise RuntimeError("LTX rollout cache experiment hash differs")
    rollout_seeds, rollout_indices, target_indices, rollout_records = (
        _generated_layout(rollout_cache, real_records)
    )
    rollout_targets = subset_targets(
        targets, torch.tensor(target_indices, dtype=torch.long)
    )
    train_indices = _indices(real_records, "train")
    normalizers = fit_ltx_normalizers(targets, train_indices)
    reference = _model_factory(real_features, targets, config)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in reference.parameters()
        if parameter.requires_grad
    )
    del reference
    training = config["training"]
    query_seeds = list(args.seed or training["seeds"])
    max_epochs = int(args.max_epochs or training["max_epochs"])
    patience = int(args.patience or training["patience"])
    pair_rows = _read_jsonl(args.pairs_manifest.resolve())
    run_rows = _read_jsonl(args.runs_manifest.resolve())
    real_permutation = split_preserving_permutation(real_records, 20260712)
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[LTX Safety Query] real={tuple(real_features.shape)} "
        f"rollouts={tuple(rollout_cache['features'].shape)} "
        f"parameters={trainable_parameters:,} query_seeds={query_seeds} "
        f"rollout_seeds={rollout_seeds}",
        flush=True,
    )

    per_seed = []
    for seed in query_seeds:
        model, real_predictions, state, training_report, runtime = train_one_seed(
            real_features,
            targets,
            real_records,
            config,
            normalizers,
            seed=seed,
            device=args.device,
            max_epochs=max_epochs,
            patience=patience,
        )
        real_evaluation, real_physical = evaluate_ltx_predictions(
            real_predictions,
            targets,
            real_records,
            targets["channels"],
            normalizers,
            pair_rows,
            run_rows,
        )
        threshold = real_evaluation["validation_tip_threshold"]
        real_controls = {}
        if not args.skip_controls:
            for control in CONTROL_NAMES:
                if control == "identity":
                    continue
                intervened = IntervenedFeatures(
                    real_features,
                    control,
                    permutation=real_permutation,
                )
                control_predictions = _predict(
                    model,
                    intervened,
                    torch.arange(len(real_records), dtype=torch.long),
                    int(training["batch_size"]),
                    args.device,
                )
                control_evaluation, _ = evaluate_ltx_predictions(
                    control_predictions,
                    targets,
                    real_records,
                    targets["channels"],
                    normalizers,
                    pair_rows,
                    run_rows,
                    threshold=threshold,
                )
                real_controls[control] = control_evaluation
        rollout_predictions = _predict_rollouts(
            model,
            rollout_cache["features"],
            rollout_indices,
            batch_size=int(training["batch_size"]),
            device=args.device,
        )
        per_rollout_seed = {}
        for rollout_seed, predictions in zip(
            rollout_seeds, rollout_predictions, strict=True
        ):
            evaluation, _ = evaluate_ltx_predictions(
                predictions,
                rollout_targets,
                rollout_records,
                targets["channels"],
                normalizers,
                pair_rows,
                run_rows,
                threshold=threshold,
            )
            per_rollout_seed[str(rollout_seed)] = evaluation
        ensemble_predictions = aggregate_rollout_predictions(rollout_predictions)
        ensemble_evaluation, ensemble_physical = evaluate_ltx_predictions(
            ensemble_predictions,
            rollout_targets,
            rollout_records,
            targets["channels"],
            normalizers,
            pair_rows,
            run_rows,
            threshold=threshold,
        )
        reverse_evaluation = None
        if not args.skip_controls:
            reverse_predictions = _predict_rollouts(
                model,
                rollout_cache["features"],
                rollout_indices,
                batch_size=int(training["batch_size"]),
                device=args.device,
                reverse_time=True,
            )
            reverse_ensemble = aggregate_rollout_predictions(reverse_predictions)
            reverse_evaluation, _ = evaluate_ltx_predictions(
                reverse_ensemble,
                rollout_targets,
                rollout_records,
                targets["channels"],
                normalizers,
                pair_rows,
                run_rows,
                threshold=threshold,
            )
        seed_result = {
            "seed": seed,
            "training": training_report,
            "runtime": runtime,
            "real_future_oracle": real_evaluation,
            "real_feature_controls": real_controls,
            "generated_rollout": {
                "per_rollout_seed": per_rollout_seed,
                "ensemble": ensemble_evaluation,
                "temporal_reverse_ensemble": reverse_evaluation,
            },
        }
        per_seed.append(seed_result)
        write_torch_atomic(
            output_root / f"seed_{seed:04d}.pt",
            {
                "seed": seed,
                "model_state_dict": state,
                "normalizers": normalizers,
                "real_predictions_normalized": real_predictions,
                "real_predictions_physical": real_physical,
                "rollout_predictions_by_seed_normalized": {
                    rollout_seed: predictions
                    for rollout_seed, predictions in zip(
                        rollout_seeds, rollout_predictions, strict=True
                    )
                },
                "rollout_ensemble_predictions_normalized": ensemble_predictions,
                "rollout_ensemble_predictions_physical": ensemble_physical,
                "rollout_condition_ids": [
                    str(record["condition_id"]) for record in rollout_records
                ],
                "real_cache_contract": real_cache["contract"],
                "rollout_cache_contract": rollout_cache["contract"],
                "target_contract": targets["contract"],
                "experiment_sha256": _sha256(experiment_path),
            },
        )
        oracle = real_evaluation["splits"]["test"]
        generated = ensemble_evaluation["splits"]["test"]
        print(
            f"[LTX Safety Query] seed={seed} "
            f"oracle_tip_auc={oracle['future_tip']['auroc']:.4f} "
            f"generated_tip_auc={generated['future_tip']['auroc']:.4f} "
            f"generated_mass_mae={generated['static']['load_mass_kg']['mae']:.2f}kg "
            f"generated_zmp_summary_mae="
            f"{generated['future_summary']['min_zmp_outward_margin_m']['mae']:.4f}m",
            flush=True,
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate_paths = [
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
    aggregate = {
        "real_future_oracle": _aggregate(
            [item["real_future_oracle"] for item in per_seed],
            aggregate_paths,
        ),
        "generated_rollout_ensemble": _aggregate(
            [item["generated_rollout"]["ensemble"] for item in per_seed],
            aggregate_paths,
        ),
        "runtime": _aggregate(per_seed, [("runtime", "duration_s")]),
    }
    if not args.skip_controls:
        aggregate["real_feature_controls"] = {
            control: _aggregate(
                [item["real_feature_controls"][control] for item in per_seed],
                aggregate_paths,
            )
            for control in CONTROL_NAMES
            if control != "identity"
        }
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "experiment_sha256": _sha256(experiment_path),
        "real_cache": str(real_path),
        "rollout_cache": str(rollout_path),
        "target_cache": str(target_path),
        "foundation_parameters_frozen": True,
        "query_training_source": "real_future_latents_only",
        "generated_latents_used_for_training": False,
        "query_decoder_trainable_parameters": trainable_parameters,
        "counts": {
            "real": {
                split: len(_indices(real_records, split))
                for split in ("train", "val", "test")
            },
            "rollout_conditions": len(rollout_records),
            "rollout_seeds": rollout_seeds,
            "rollout_jobs": len(rollout_cache["records"]),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
