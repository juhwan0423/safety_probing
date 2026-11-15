from __future__ import annotations

import math
import unittest

import torch

from safety_probing.safety_query_probe import (
    _masked_stats,
    _polygon_area_centroid,
    build_safety_query_decoder,
)
from safety_probing.safety_query_controls import (
    IntervenedFeatures,
    split_preserving_permutation,
)
from safety_probing.safety_query_targets import (
    STATE_NAMES,
    _state_from_row,
    tubelet_center_times,
    world_point_to_yaw_local,
)


class SafetyQueryTargetTest(unittest.TestCase):
    def test_rgb_frame_times_use_post_step_capture_schedule(self) -> None:
        result = tubelet_center_times([0, 3, 7, 10], 24.0)
        self.assertAlmostEqual(result[0], 2.5 / 24.0)
        self.assertAlmostEqual(result[1], 9.5 / 24.0)

    def test_world_point_is_rotated_into_yaw_local_frame(self) -> None:
        x, y = world_point_to_yaw_local(1.0, 2.0, 1.0, 1.0, 90.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0, places=7)

    def test_zmp_coordinates_follow_outward_validity_mask(self) -> None:
        row = {
            "forklift_pos_x": "0",
            "forklift_pos_y": "0",
            "forklift_pos_z": "0",
            "yaw_deg": "0",
            "roll_deg": "1",
            "pitch_deg": "2",
            "imu_valid": "False",
            "linear_vel_x": "0",
            "linear_vel_y": "0",
            "linear_vel_z": "0",
            "system_com_available": "False",
            "zmp_valid": "True",
            "zmp_x": "0.2",
            "zmp_y": "0.3",
            "outward_lateral_margin_valid": "False",
        }
        _, mask = _state_from_row(row)
        self.assertFalse(mask[STATE_NAMES.index("zmp_body_x")])
        row["outward_lateral_margin_valid"] = "True"
        _, mask = _state_from_row(row)
        self.assertTrue(mask[STATE_NAMES.index("zmp_body_x")])


class SafetyQueryDecoderTest(unittest.TestCase):
    def test_query_decoder_emits_every_physical_group(self) -> None:
        model = build_safety_query_decoder(
            feature_dim=8,
            temporal_tokens=2,
            spatial_height=2,
            spatial_width=2,
            state_dim=5,
            contact_count=3,
            summary_dim=4,
            model_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        model.eval()
        output = model(torch.randn(3, 2, 2, 2, 8))
        self.assertEqual(tuple(output["static"].shape), (3, 2))
        self.assertEqual(tuple(output["observed_state"].shape), (3, 2, 5))
        self.assertEqual(tuple(output["future_state"].shape), (3, 2, 5))
        self.assertEqual(
            tuple(output["observed_contact_logits"].shape), (3, 2, 3)
        )
        self.assertEqual(
            tuple(output["future_contact_xy"].shape), (3, 2, 3, 2)
        )
        self.assertEqual(tuple(output["future_summary"].shape), (3, 4))
        self.assertEqual(tuple(output["future_tip_logits"].shape), (3,))

    def test_masked_stats_ignore_invalid_values(self) -> None:
        values = torch.tensor(
            [
                [[1.0, 1000.0], [3.0, 1000.0]],
                [[5.0, 2.0], [7.0, 4.0]],
            ]
        )
        mask = torch.tensor(
            [
                [[True, False], [True, False]],
                [[True, True], [True, True]],
            ]
        )
        mean, std = _masked_stats(values, mask, torch.tensor([0, 1]))
        self.assertTrue(torch.allclose(mean, torch.tensor([4.0, 3.0])))
        self.assertAlmostEqual(float(std[0]), math.sqrt(5.0), places=6)
        self.assertAlmostEqual(float(std[1]), 1.0, places=6)

    def test_polygon_area_and_centroid(self) -> None:
        result = _polygon_area_centroid(
            [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result, (4.0, 1.0, 1.0))


class SafetyQueryControlTest(unittest.TestCase):
    def test_feature_interventions_remove_requested_axis(self) -> None:
        features = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(
            2, 3, 2, 2, 1
        )
        indices = torch.tensor([0, 1])
        spatial = IntervenedFeatures(
            features, "spatial_mean_repeat"
        )[indices]
        self.assertTrue(torch.equal(spatial[:, :, 0, 0], spatial[:, :, 1, 1]))
        temporal = IntervenedFeatures(
            features, "temporal_mean_repeat"
        )[indices]
        self.assertTrue(torch.equal(temporal[:, 0], temporal[:, 2]))
        reversed_features = IntervenedFeatures(
            features, "temporal_reverse"
        )[indices]
        self.assertTrue(torch.equal(reversed_features[:, 0], features[:, -1]))
        zero = IntervenedFeatures(features, "zero_memory")[indices]
        self.assertEqual(int(torch.count_nonzero(zero)), 0)

    def test_permutation_never_crosses_split(self) -> None:
        records = [
            {"split": "train"},
            {"split": "train"},
            {"split": "val"},
            {"split": "val"},
            {"split": "test"},
            {"split": "test"},
        ]
        permutation = split_preserving_permutation(records, 17)
        for target_index, source_index in enumerate(permutation.tolist()):
            self.assertEqual(
                records[target_index]["split"], records[source_index]["split"]
            )


if __name__ == "__main__":
    unittest.main()
