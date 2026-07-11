from __future__ import annotations

import unittest

from safety_probing.ltx_batch import select_batch_rows


class LTXBatchSelectionTest(unittest.TestCase):
    def test_filters_shard_and_split_deterministically(self) -> None:
        rows = [
            {
                "condition_id": "b",
                "dataset_root": "/data/eccv_workshop_grid_part_01",
                "splits": {"main": "test"},
            },
            {
                "condition_id": "a",
                "dataset_root": "/data/eccv_workshop_grid_part_01",
                "splits": {"main": "val"},
            },
            {
                "condition_id": "c",
                "dataset_root": "/data/eccv_workshop_grid_part_02",
                "splits": {"main": "test"},
            },
            {
                "condition_id": "d",
                "dataset_root": "/data/eccv_workshop_grid_part_01",
                "splits": {"main": "train"},
            },
        ]
        selected = select_batch_rows(
            rows, shard="part_01", splits={"val", "test"}
        )
        self.assertEqual([row["condition_id"] for row in selected], ["b", "a"])

    def test_rejects_duplicate_canonical_conditions(self) -> None:
        row = {
            "condition_id": "same",
            "dataset_root": "/data/eccv_workshop_grid_part_01",
            "splits": {"main": "test"},
        }
        with self.assertRaises(RuntimeError):
            select_batch_rows(
                [row, dict(row)], shard="part_01", splits={"test"}
            )

    def test_limits_each_outcome_for_balanced_pilot(self) -> None:
        rows = []
        for outcome in ("observed_tip", "confirmed_recovered"):
            for index in range(3):
                rows.append(
                    {
                        "condition_id": f"{outcome}_{index}",
                        "dataset_root": "/data/eccv_workshop_grid_part_01",
                        "splits": {"main": "val"},
                        "safety_outcome": outcome,
                    }
                )
        selected = select_batch_rows(
            rows,
            shard="part_01",
            splits={"val"},
            max_per_outcome=1,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {row["safety_outcome"] for row in selected},
            {"observed_tip", "confirmed_recovered"},
        )


if __name__ == "__main__":
    unittest.main()
