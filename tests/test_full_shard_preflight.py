from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from safety_probing.full_shard_preflight import audit_pair_directory
from safety_probing.ltx_batch import select_batch_rows
from safety_probing.vjepa_features import select_canonical_rows


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class FullShardSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "condition_id": "b",
                "dataset_root": "/data/eccv_workshop_grid_part_02",
                "splits": {"main": "test"},
                "safety_outcome": "observed_tip",
            },
            {
                "condition_id": "a",
                "dataset_root": "/data/eccv_workshop_grid_part_01",
                "splits": {"main": "val"},
                "safety_outcome": "confirmed_recovered",
            },
        ]

    def test_full_alias_selects_every_canonical_condition(self) -> None:
        self.assertEqual(
            [row["condition_id"] for row in select_canonical_rows(self.rows, shard="full")],
            ["a", "b"],
        )
        self.assertEqual(len(select_canonical_rows(self.rows, shard="all")), 2)

    def test_ltx_full_alias_still_filters_requested_split(self) -> None:
        selected = select_batch_rows(
            self.rows, shard="full", splits={"test"}
        )
        self.assertEqual([row["condition_id"] for row in selected], ["b"])


class PairDirectoryAuditTest(unittest.TestCase):
    def test_deep_audit_checks_rows_and_camera_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pair_dir = Path(temporary) / "pair_1"
            _write(
                pair_dir / "metadata_pair.json",
                json.dumps({"pair_id": "pair_1"}) + "\n",
            )
            _write(pair_dir / "assumptions.json", "{}\n")
            for label in ("A", "B"):
                run = pair_dir / f"run_{label}"
                _write(run / "metadata_run.json", "{}\n")
                _write(run / "command_log.csv", "step\n")
                _write(run / "dense_log.csv", "step\n0\n1\n")
                for frame in range(2):
                    _write(
                        run / "rgb" / "main" / f"frame_{frame:06d}.png",
                        "png",
                    )
            issues = audit_pair_directory(
                pair_dir,
                pair_id="pair_1",
                expected_dense_rows=2,
                expected_rgb_frames=2,
                camera_names=("main",),
                deep=True,
            )
            self.assertEqual(issues, [])
            (pair_dir / "run_B" / "rgb" / "main" / "frame_000001.png").unlink()
            issues = audit_pair_directory(
                pair_dir,
                pair_id="pair_1",
                expected_dense_rows=2,
                expected_rgb_frames=2,
                camera_names=("main",),
                deep=True,
            )
            self.assertTrue(any("frames=1" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
