from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback
from typing import Any

from .io_utils import load_json, write_json, write_jsonl
from .vjepa_native import (
    _read_jsonl,
    _save_latents,
    create_native_runner,
    evaluate_native_pair,
    select_gold_pairs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _record_complete(
    sidecar: Path,
    *,
    output_root: Path,
    experiment_sha256: str,
    registry_sha256: str,
    save_latents: bool,
) -> bool:
    if not sidecar.is_file():
        return False
    record = load_json(sidecar)
    if record.get("status") != "complete":
        return False
    if record.get("experiment_sha256") != experiment_sha256:
        return False
    if record.get("registry_sha256") != registry_sha256:
        return False
    if save_latents:
        latent_path = record.get("latent_path")
        return bool(latent_path) and (output_root / str(latent_path)).is_file()
    return True


def _collect_records(output_root: Path) -> list[dict[str, Any]]:
    records = [load_json(path) for path in output_root.rglob("result.json")]
    records.sort(
        key=lambda row: (
            str(row.get("split")),
            str(row.get("pair_id")),
        )
    )
    return records


def summarize_batch_records(
    records: list[dict[str, Any]],
    *,
    requested_pair_count: int,
    interrupted: bool = False,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in records)
    complete = [row for row in records if row.get("status") == "complete"]
    queries = [
        query
        for row in complete
        for query in row["primary_result"]["queries"]
    ]
    correct_queries = sum(bool(query["correct_outcome"]) for query in queries)
    margins = [float(query["correct_vs_alternative_margin"]) for query in queries]

    by_split: dict[str, Any] = {}
    for split in sorted({str(row.get("split")) for row in complete}):
        split_records = [row for row in complete if str(row.get("split")) == split]
        split_queries = [
            query
            for row in split_records
            for query in row["primary_result"]["queries"]
        ]
        split_correct = sum(
            bool(query["correct_outcome"]) for query in split_queries
        )
        by_split[split] = {
            "pair_count": len(split_records),
            "query_count": len(split_queries),
            "correct_query_count": split_correct,
            "outcome_retrieval_accuracy": (
                split_correct / len(split_queries) if split_queries else None
            ),
        }

    return {
        "schema_version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_pair_count": requested_pair_count,
        "record_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "completed_pair_count": len(complete),
        "primary_query_count": len(queries),
        "primary_correct_query_count": correct_queries,
        "primary_outcome_retrieval_accuracy": (
            correct_queries / len(queries) if queries else None
        ),
        "all_queries_correct_pair_count": sum(
            bool(row["primary_result"]["all_queries_correct"])
            for row in complete
        ),
        "mean_correct_vs_alternative_margin": (
            statistics.fmean(margins) if margins else None
        ),
        "median_correct_vs_alternative_margin": (
            statistics.median(margins) if margins else None
        ),
        "by_split": by_split,
        "interrupted": interrupted,
    }


def _write_batch_state(
    output_root: Path,
    *,
    requested_pair_count: int,
    interrupted: bool = False,
) -> dict[str, Any]:
    records = _collect_records(output_root)
    write_jsonl(output_root / "batch_manifest.jsonl", records)
    summary = summarize_batch_records(
        records,
        requested_pair_count=requested_pair_count,
        interrupted=interrupted,
    )
    write_json(output_root / "batch_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable frozen V-JEPA J0 retrieval with one persistent model load."
        )
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/model_registry.json")
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/vjepa_native_future.json"),
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
    parser.add_argument("--shard")
    parser.add_argument(
        "--split", action="append", choices=("train", "val", "test")
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max-pairs must be positive")

    registry_path = args.registry.resolve()
    experiment_path = args.experiment.resolve()
    pairs_manifest_path = args.pairs_manifest.resolve()
    runs_manifest_path = args.runs_manifest.resolve()
    project_root = registry_path.parent.parent
    registry = load_json(registry_path)
    experiment = load_json(experiment_path)
    splits = set(args.split or ["val", "test"])
    selected = select_gold_pairs(
        _read_jsonl(pairs_manifest_path),
        _read_jsonl(runs_manifest_path),
        splits=splits,
        shard=args.shard,
    )
    if args.max_pairs is not None:
        selected = selected[: args.max_pairs]
    if not selected:
        raise RuntimeError(
            f"No decisive Gold pairs found for shard={args.shard}, "
            f"splits={sorted(splits)}"
        )

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (
            project_root
            / "outputs"
            / "vjepa_native_batch"
            / (args.shard or "all_shards")
        ).resolve()
    )
    experiment_sha256 = _sha256(experiment_path)
    registry_sha256 = _sha256(registry_path)
    split_counts = Counter(
        str(pair["splits"]["main"]) for pair, _runs in selected
    )
    direction_counts = Counter(
        str(pair.get("turn_direction")) for pair, _runs in selected
    )
    plan = {
        "schema_version": 1,
        "experiment": experiment["experiment"],
        "shard": args.shard,
        "splits": sorted(splits),
        "pair_count": len(selected),
        "pair_ids": [str(pair["pair_id"]) for pair, _runs in selected],
        "pair_counts_by_split": dict(sorted(split_counts.items())),
        "pair_counts_by_direction": dict(sorted(direction_counts.items())),
        "save_latents": args.save_latents,
        "experiment_path": str(experiment_path),
        "experiment_sha256": experiment_sha256,
        "registry_sha256": registry_sha256,
        "output_root": str(output_root),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    contract_path = output_root / "batch_contract.json"
    if contract_path.is_file():
        previous = load_json(contract_path)
        for key in (
            "shard",
            "splits",
            "pair_ids",
            "save_latents",
            "experiment_sha256",
            "registry_sha256",
        ):
            if previous.get(key) != plan.get(key):
                raise RuntimeError(
                    f"Existing batch contract differs at {key}; use a new output root"
                )
    write_json(contract_path, plan)
    write_json(output_root / "experiment.snapshot.json", experiment)
    write_json(output_root / "model_registry.snapshot.json", registry)

    pending: list[
        tuple[dict[str, Any], dict[str, dict[str, Any]], Path, Path]
    ] = []
    for pair, runs in selected:
        split = str(pair["splits"]["main"])
        pair_root = output_root / split / str(pair["pair_id"])
        sidecar_path = pair_root / "result.json"
        latent_path = pair_root / "latents.pt"
        if not args.overwrite and _record_complete(
            sidecar_path,
            output_root=output_root,
            experiment_sha256=experiment_sha256,
            registry_sha256=registry_sha256,
            save_latents=args.save_latents,
        ):
            continue
        pending.append((pair, runs, sidecar_path, latent_path))

    print(
        f"[V-JEPA J0 batch] Complete={len(selected) - len(pending)}, "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        summary = _write_batch_state(
            output_root, requested_pair_count=len(selected)
        )
        print(json.dumps(summary, indent=2), flush=True)
        return

    print("[V-JEPA J0 batch] Loading frozen model once", flush=True)
    runner, model_record, resolution = create_native_runner(
        experiment=experiment,
        registry=registry,
        project_root=project_root,
        device=args.device,
    )
    interrupted = False
    try:
        for index, (pair, runs, sidecar_path, latent_path) in enumerate(
            pending, start=1
        ):
            pair_id = str(pair["pair_id"])
            split = str(pair["splits"]["main"])
            print(
                f"[V-JEPA J0 batch] Pair {index}/{len(pending)} "
                f"split={split} id={pair_id}",
                flush=True,
            )
            started = time.monotonic()
            base_record = {
                "status": "failed",
                "pair_id": pair_id,
                "split": split,
                "shard": args.shard,
                "experiment_sha256": experiment_sha256,
                "registry_sha256": registry_sha256,
                "latent_path": (
                    _relative(latent_path, output_root)
                    if args.save_latents
                    else None
                ),
            }
            try:
                payload, latent_payload = evaluate_native_pair(
                    pair=pair,
                    runs=runs,
                    runner=runner,
                    experiment=experiment,
                    model_record=model_record,
                    resolution=resolution,
                    include_latents=args.save_latents,
                )
                if args.save_latents:
                    if latent_payload is None:
                        raise RuntimeError(
                            "Latent output was requested but not returned"
                        )
                    _save_latents(latent_path, latent_payload)
                record = {
                    **payload,
                    **base_record,
                    "status": "complete",
                    "duration_s": round(time.monotonic() - started, 3),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:
                record = {
                    **base_record,
                    "duration_s": round(time.monotonic() - started, 3),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(sidecar_path, record)
                if args.fail_fast:
                    raise
            else:
                write_json(sidecar_path, record)
    except KeyboardInterrupt:
        interrupted = True
        print(
            "[V-JEPA J0 batch] Interrupted; completed pairs are resumable",
            flush=True,
        )
        raise
    finally:
        summary = _write_batch_state(
            output_root,
            requested_pair_count=len(selected),
            interrupted=interrupted,
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
