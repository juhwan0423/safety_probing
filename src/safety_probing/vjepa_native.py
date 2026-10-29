from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from .adapters.vjepa import (
    FrozenVJEPANativePredictor,
    latent_candidate_energy,
    load_indexed_video_tensor,
    uniform_frame_indices,
)
from .io_utils import load_json, write_json


CANDIDATE_LABELS = ("A", "B")
DECISIVE_OUTCOMES = {"observed_tip", "confirmed_recovered"}
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index_runs(
    run_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    runs_by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in run_rows:
        label = str(row.get("run_label"))
        if label not in CANDIDATE_LABELS:
            continue
        pair_runs = runs_by_pair.setdefault(str(row["pair_id"]), {})
        if label in pair_runs:
            raise RuntimeError(
                f"Duplicate run {label} for pair {row['pair_id']} in run manifest"
            )
        pair_runs[label] = row
    return runs_by_pair


def select_gold_pairs(
    pair_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    *,
    splits: set[str],
    shard: str | None = None,
) -> list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]:
    runs_by_pair = _index_runs(run_rows)
    dataset_name = None
    if shard is not None:
        dataset_name = (
            f"eccv_workshop_grid_{shard}" if shard.startswith("part_") else shard
        )
    candidates: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    for pair in sorted(
        pair_rows,
        key=lambda row: (
            SPLIT_ORDER.get(str(row.get("splits", {}).get("main")), 99),
            str(row["pair_id"]),
        ),
    ):
        current_pair_id = str(pair["pair_id"])
        if str(pair.get("splits", {}).get("main")) not in splits:
            continue
        if dataset_name is not None and Path(str(pair["dataset_root"])).name != dataset_name:
            continue
        if pair.get("quality_tier") != "gold":
            continue
        outcomes = {
            pair.get("run_A_safety_outcome"),
            pair.get("run_B_safety_outcome"),
        }
        if outcomes != DECISIVE_OUTCOMES:
            continue
        pair_runs = runs_by_pair.get(current_pair_id, {})
        if set(pair_runs) != set(CANDIDATE_LABELS):
            continue
        for label in CANDIDATE_LABELS:
            expected = pair[f"run_{label}_safety_outcome"]
            actual = pair_runs[label].get("safety_outcome")
            if actual != expected:
                raise RuntimeError(
                    f"Pair/run outcome mismatch for {current_pair_id} run {label}: "
                    f"{expected} vs {actual}"
                )
        candidates.append((pair, pair_runs))
    return candidates


def select_gold_pair(
    pair_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    *,
    split: str = "val",
    pair_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if pair_id is not None:
        pair_rows = [row for row in pair_rows if str(row["pair_id"]) == pair_id]
        splits = {
            str(row.get("splits", {}).get("main")) for row in pair_rows
        }
    else:
        splits = {split}
    candidates = select_gold_pairs(pair_rows, run_rows, splits=splits)

    if not candidates:
        requested = f"pair_id={pair_id}" if pair_id else f"main split={split}"
        raise ValueError(f"No decisive Gold pair found for {requested}")
    return candidates[0]


def summarize_candidate_retrieval(
    energy_matrix: list[list[float]],
    outcomes: dict[str, str],
) -> dict[str, Any]:
    if len(energy_matrix) != len(CANDIDATE_LABELS) or any(
        len(row) != len(CANDIDATE_LABELS) for row in energy_matrix
    ):
        raise ValueError("Expected a 2x2 A/B energy matrix")

    queries: list[dict[str, Any]] = []
    identity_correct = 0
    outcome_correct = 0
    for query_index, query_label in enumerate(CANDIDATE_LABELS):
        row = energy_matrix[query_index]
        selected_index = min(range(len(row)), key=row.__getitem__)
        selected_label = CANDIDATE_LABELS[selected_index]
        alternative_index = 1 - query_index
        correct_energy = float(row[query_index])
        alternative_energy = float(row[alternative_index])
        correct_identity = selected_label == query_label
        correct_outcome = outcomes[selected_label] == outcomes[query_label]
        identity_correct += int(correct_identity)
        outcome_correct += int(correct_outcome)
        queries.append(
            {
                "query_run": query_label,
                "expected_outcome": outcomes[query_label],
                "selected_candidate": selected_label,
                "selected_outcome": outcomes[selected_label],
                "correct_identity": correct_identity,
                "correct_outcome": correct_outcome,
                "energy_to_A": float(row[0]),
                "energy_to_B": float(row[1]),
                "correct_vs_alternative_margin": (
                    alternative_energy - correct_energy
                ),
            }
        )
    denominator = float(len(CANDIDATE_LABELS))
    return {
        "queries": queries,
        "identity_accuracy": identity_correct / denominator,
        "outcome_accuracy": outcome_correct / denominator,
        "all_queries_correct": identity_correct == len(CANDIDATE_LABELS),
    }


def _save_latents(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pt", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def native_sampling_contract(experiment: dict[str, Any]) -> dict[str, Any]:
    input_spec = experiment["input"]
    future_spec = experiment["masked_future"]
    observation_source_frames = int(input_spec["source_frame_count"])
    total_source_frames = int(future_spec["total_source_frame_count"])
    context_model_frames = int(input_spec["sampled_frame_count"])
    future_model_frames = int(future_spec["sampled_frame_count"])
    context_source_indices = uniform_frame_indices(
        0, observation_source_frames - 1, context_model_frames
    )
    future_source_indices = uniform_frame_indices(
        observation_source_frames,
        total_source_frames - 1,
        future_model_frames,
    )
    return {
        "camera": str(input_spec["camera"]),
        "source_fps": float(input_spec["source_fps"]),
        "context_model_frames": context_model_frames,
        "future_model_frames": future_model_frames,
        "context_source_indices": context_source_indices,
        "future_source_indices": future_source_indices,
        "source_indices": context_source_indices + future_source_indices,
    }


def create_native_runner(
    *,
    experiment: dict[str, Any],
    registry: dict[str, Any],
    project_root: Path,
    device: str,
) -> tuple[FrozenVJEPANativePredictor, dict[str, Any], int]:
    model_key = str(experiment["native_model"]["model_registry_key"])
    if model_key != "vjepa_2":
        raise ValueError("J0 native target-energy evaluation requires model vjepa_2")
    model_spec = registry["models"][model_key]
    runtime = model_spec["runtime"]
    resolution = int(runtime["resolution"])
    sampling = native_sampling_contract(experiment)
    checkpoint = project_root / runtime["checkpoint"]
    runner = FrozenVJEPANativePredictor(
        code_root=(
            project_root
            / registry["code_root"]
            / model_spec["code"]["directory"]
        ),
        checkpoint=checkpoint,
        version="2",
        context_frame_count=sampling["context_model_frames"],
        future_frame_count=sampling["future_model_frames"],
        resolution=resolution,
        device=device,
    )
    model_record = {
        "registry_key": model_key,
        "display_name": model_spec["display_name"],
        "code_revision": model_spec["code"]["revision"],
        "checkpoint": str(checkpoint.resolve()),
        "device": device,
    }
    return runner, model_record, resolution


def evaluate_native_pair(
    *,
    pair: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    runner: FrozenVJEPANativePredictor,
    experiment: dict[str, Any],
    model_record: dict[str, Any],
    resolution: int,
    include_latents: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    import torch

    sampling = native_sampling_contract(experiment)
    videos = []
    for label in CANDIDATE_LABELS:
        frame_directory = Path(runs[label]["rgb_dirs"][sampling["camera"]])
        video, loaded_indices = load_indexed_video_tensor(
            frame_directory,
            frame_indices=sampling["source_indices"],
            resolution=resolution,
        )
        if loaded_indices != sampling["source_indices"]:
            raise RuntimeError("Loaded frame indices differ from the sampling contract")
        videos.append(video)

    video_batch = torch.cat(videos, dim=0)
    context_model_frames = int(sampling["context_model_frames"])
    target_video_batch = torch.stack(
        [
            torch.cat(
                [
                    video_batch[query_index, :, :context_model_frames],
                    video_batch[candidate_index, :, context_model_frames:],
                ],
                dim=1,
            )
            for query_index in range(len(CANDIDATE_LABELS))
            for candidate_index in range(len(CANDIDATE_LABELS))
        ]
    )
    latent_result = runner.predict_and_encode_targets(
        video_batch, target_video_batch=target_video_batch
    )
    target_future = latent_result["target_future"].reshape(
        len(CANDIDATE_LABELS),
        len(CANDIDATE_LABELS),
        *latent_result["target_future"].shape[1:],
    )
    energies = latent_candidate_energy(
        latent_result["predicted_future"], target_future
    )
    outcomes = {
        label: str(runs[label]["safety_outcome"])
        for label in CANDIDATE_LABELS
    }
    retrieval = {
        metric: summarize_candidate_retrieval(matrix, outcomes)
        for metric, matrix in energies.items()
    }
    primary_metric = str(experiment["evaluation"]["primary_energy"])
    payload = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "evaluation_scope": "paired_candidate_retrieval_not_standalone_classification",
        "trainable_parameter_count": 0,
        "model": model_record,
        "pair": {
            "pair_id": pair["pair_id"],
            "main_split": pair["splits"]["main"],
            "quality_tier": pair["quality_tier"],
            "tip_state_transition": pair["tip_state_transition"],
            "same_action_trajectory": pair["same_action_trajectory"],
            "turn_direction": pair.get("turn_direction"),
            "intervention_type": pair.get("intervention_type"),
            "runs": {
                label: {
                    "run_id": runs[label]["run_id"],
                    "safety_outcome": outcomes[label],
                    "binary_tip_label": runs[label]["binary_tip_label"],
                }
                for label in CANDIDATE_LABELS
            },
        },
        "sampling": {
            "source_fps": sampling["source_fps"],
            "observation_source_indices": sampling["context_source_indices"],
            "future_source_indices": sampling["future_source_indices"],
            "model_frame_count": len(sampling["source_indices"]),
            "observation_model_frame_count": sampling["context_model_frames"],
            "future_model_frame_count": sampling["future_model_frames"],
        },
        "latents": {
            "predicted_future_shape": list(
                latent_result["predicted_future"].shape
            ),
            "target_future_shape": list(target_future.shape),
            "target_context_control": (
                "query_prefix_fixed_while_candidate_future_is_swapped"
            ),
            "context_token_count": int(latent_result["context_mask"].shape[1]),
            "future_token_count": int(latent_result["future_mask"].shape[1]),
            "peak_gpu_memory_bytes": latent_result["peak_gpu_memory_bytes"],
        },
        "energy_matrices": {
            "row_queries": list(CANDIDATE_LABELS),
            "column_candidates": list(CANDIDATE_LABELS),
            **energies,
        },
        "retrieval": retrieval,
        "primary_result": {
            "energy": primary_metric,
            **retrieval[primary_metric],
        },
    }
    latent_payload = None
    if include_latents:
        latent_payload = {
            "pair_id": pair["pair_id"],
            "candidate_labels": CANDIDATE_LABELS,
            "source_frame_indices": sampling["source_indices"],
            "predicted_future": latent_result["predicted_future"].half(),
            "target_future": target_future.half(),
        }
    return payload, latent_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen V-JEPA native future-latent retrieval on one Gold pair."
        )
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_native_future.json"),
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
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
    parser.add_argument("--pair-id")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/vjepa_native/smoke_result.json"),
    )
    parser.add_argument("--latent-output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    experiment = load_json(args.experiment.resolve())
    registry_path = args.registry.resolve()
    registry = load_json(registry_path)
    project_root = registry_path.parent.parent

    pair, runs = select_gold_pair(
        _read_jsonl(args.pairs_manifest.resolve()),
        _read_jsonl(args.runs_manifest.resolve()),
        split=args.split,
        pair_id=args.pair_id,
    )
    print(
        f"[V-JEPA native] Pair {pair['pair_id']} "
        f"({runs['A']['safety_outcome']} vs {runs['B']['safety_outcome']})",
        flush=True,
    )
    print("[V-JEPA native] Loading frozen released encoder and predictor", flush=True)
    runner, model_record, resolution = create_native_runner(
        experiment=experiment,
        registry=registry,
        project_root=project_root,
        device=args.device,
    )
    print("[V-JEPA native] Predicting fully masked future latent tokens", flush=True)
    payload, latent_payload = evaluate_native_pair(
        pair=pair,
        runs=runs,
        runner=runner,
        experiment=experiment,
        model_record=model_record,
        resolution=resolution,
        include_latents=args.latent_output is not None,
    )
    write_json(args.output.resolve(), payload)
    if args.latent_output is not None:
        if latent_payload is None:
            raise RuntimeError("Latent output was requested but not returned")
        _save_latents(args.latent_output.resolve(), latent_payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
