from __future__ import annotations

import unittest

from safety_probing.split_builder import build_split_manifest


class SplitBuilderTest(unittest.TestCase):
    def test_shared_endpoints_do_not_cross_generated_splits(self) -> None:
        routes = [
            {
                "route_id": direction,
                "turn_direction": direction,
                "mirror_group_id": "vslow_swide",
                "speed_label": "slow",
                "steering_label": "wide",
            }
            for direction in ("left", "right")
        ]
        scenes = [
            {
                "scenario_id": f"m{mass}",
                "load_base_mass_kg": float(mass),
                "load_base_com_offset_local_xyz_m": [0.0, 0.0, 0.0],
                "load_spawn_lift_m": 2.05,
                "load_size_m": [1.2, 0.9, 0.9],
                "load_base_offset_m": [0.0, 0.3, 0.01],
            }
            for mass in (240, 360, 480)
        ]
        pair_requests = []
        for direction in ("left", "right"):
            for index, (base, modified) in enumerate(((240, 360), (360, 480))):
                pair_requests.append(
                    {
                        "pair_id": f"{direction}_{base}_{modified}",
                        "route_id": direction,
                        "scenario_id": f"m{base}",
                        "seed": 7,
                        "intervention_type": "load_mass",
                        "base_value": float(base),
                        "modified_value": float(modified),
                    }
                )
        source = {"routes": routes, "scenes": scenes, "pair_requests": pair_requests}
        reference = {
            name: {
                "train_pair_ids": ["left_240_360", "right_240_360"],
                "val_pair_ids": [],
                "test_pair_ids": ["left_360_480", "right_360_480"],
            }
            for name in ("main", "left_to_right", "right_to_left")
        }

        generated = build_split_manifest(source, reference, seed=1)

        self.assertGreater(
            generated["audit"]["reference_manifest"]["main"][
                "cross_split_physical_condition_count"
            ],
            0,
        )
        for protocol in ("main", "left_to_right", "right_to_left"):
            audit = generated["audit"]["generated_manifest"][protocol]
            self.assertEqual(audit["cross_split_physical_condition_count"], 0)
            self.assertEqual(audit["missing_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()

