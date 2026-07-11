from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_dataset
from .config import DatasetConfig
from .reporting import write_statistics_document
from .split_builder import build_and_write_splits


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safety-probing",
        description="Audit ExoFork-CF datasets and prepare leakage-safe probing manifests.",
    )
    parser.add_argument(
        "command",
        choices=("build-splits", "analyze", "all"),
        help="Operation to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local_dataset.json"),
        help="Dataset analysis config.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip dense CSV scans for a quick inventory pass.",
    )
    return parser


def _split_counts(payload: dict) -> dict[str, dict[str, int]]:
    return {
        name: {
            "train": len(payload[name]["train_pair_ids"]),
            "val": len(payload[name]["val_pair_ids"]),
            "test": len(payload[name]["test_pair_ids"]),
            "condition_leaks": payload["audit"]["generated_manifest"][name][
                "cross_split_physical_condition_count"
            ],
        }
        for name in ("main", "left_to_right", "right_to_left")
    }


def main() -> None:
    args = _parser().parse_args()
    config = DatasetConfig.load(args.config)

    if args.command in {"build-splits", "all"}:
        split_payload = build_and_write_splits(config)
        print(f"Wrote {config.generated_split_manifest}")
        print(json.dumps(_split_counts(split_payload), indent=2))

    if args.command in {"analyze", "all"}:
        stats = analyze_dataset(config, scan_dense=not args.metadata_only)
        document = write_statistics_document(config, stats)
        print(f"Wrote {config.output_dir / 'dataset_statistics.json'}")
        print(f"Wrote {document}")
        print(
            json.dumps(
                {
                    "completed_pairs": stats["completion"]["completed_pair_count"],
                    "completed_runs": stats["completion"]["completed_run_count"],
                    "accepted_pairs": stats["pairs"]["accepted_count"],
                    "tip_runs": stats["runs"]["tip_count"],
                    "recovered_runs": stats["runs"]["stable_count"],
                    "right_censored_runs": stats["runs"]["censored_count"],
                    "rgb_mismatches": stats["rgb"]["frame_count_mismatch_count"],
                    "dense_row_mismatches": stats["dense_logs"][
                        "row_count_mismatch_run_count"
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
