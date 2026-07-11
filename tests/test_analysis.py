from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from safety_probing.analysis import analyze_dense_log
from safety_probing.config import LabelingConfig


class DenseAnalysisTest(unittest.TestCase):
    def test_validity_margin_and_contact_integrity(self) -> None:
        fieldnames = [
            "episode_step",
            "sim_time",
            "roll_deg",
            "pitch_deg",
            "wheel_contact_count",
            "support_polygon_vertex_count",
            "imu_valid",
            "system_com_available",
            "zmp_valid",
            "zmp_support_margin_m",
            "zmp_outward_lateral_margin_m",
            "com_outward_lateral_margin_m",
            "outward_lateral_margin_valid",
            "back_wheel_contact",
            "back_wheel_contact_pos_x",
            "back_wheel_contact_pos_y",
        ]
        rows = [
            {
                "episode_step": 0,
                "sim_time": 2.6,
                "roll_deg": 1.0,
                "pitch_deg": 0.5,
                "wheel_contact_count": 3,
                "support_polygon_vertex_count": 3,
                "imu_valid": True,
                "system_com_available": True,
                "zmp_valid": True,
                "zmp_support_margin_m": 0.2,
                "zmp_outward_lateral_margin_m": 0.1,
                "com_outward_lateral_margin_m": 0.2,
                "outward_lateral_margin_valid": True,
                "back_wheel_contact": True,
                "back_wheel_contact_pos_x": 1.0,
                "back_wheel_contact_pos_y": 2.0,
            },
            {
                "episode_step": 1,
                "sim_time": 3.0,
                "roll_deg": 3.0,
                "pitch_deg": 1.0,
                "wheel_contact_count": 3,
                "support_polygon_vertex_count": 3,
                "imu_valid": True,
                "system_com_available": True,
                "zmp_valid": True,
                "zmp_support_margin_m": -0.1,
                "zmp_outward_lateral_margin_m": -0.3,
                "com_outward_lateral_margin_m": -0.1,
                "outward_lateral_margin_valid": True,
                "back_wheel_contact": True,
                "back_wheel_contact_pos_x": 1.1,
                "back_wheel_contact_pos_y": 2.1,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "dense_log.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = analyze_dense_log(path)

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["missing_required_columns"], [])
        self.assertEqual(result["min_outward_zmp_margin_m"], -0.3)
        self.assertEqual(result["future_min_outward_zmp_margin_m"], -0.3)
        self.assertEqual(result["active_contact_rows_missing_xy"], 0)
        self.assertEqual(result["validity"]["imu_valid"]["true_rows"], 2)
        self.assertEqual(result["longest_support_loss_streak_steps"], 0)
        self.assertAlmostEqual(result["terminal_abs_roll_deg"], 3.0)

    def test_posthoc_support_loss_event(self) -> None:
        fieldnames = [
            "episode_step",
            "sim_time",
            "roll_deg",
            "pitch_deg",
            "wheel_contact_count",
            "imu_valid",
            "system_com_available",
            "zmp_valid",
            "zmp_outward_lateral_margin_m",
            "com_outward_lateral_margin_m",
            "outward_lateral_margin_valid",
        ]
        rows = [
            {
                "episode_step": index,
                "sim_time": index / 72.0,
                "roll_deg": index,
                "pitch_deg": 0.0,
                "wheel_contact_count": 2,
                "imu_valid": True,
                "system_com_available": True,
                "zmp_valid": True,
                "outward_lateral_margin_valid": False,
            }
            for index in range(6)
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "dense_log.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            result = analyze_dense_log(path, LabelingConfig.from_dict({}))

        self.assertEqual(result["longest_support_loss_streak_steps"], 6)
        self.assertEqual(result["terminal_support_loss_streak_steps"], 6)
        self.assertAlmostEqual(result["posthoc_support_loss_time_s"], 5 / 72.0)


if __name__ == "__main__":
    unittest.main()
