from __future__ import annotations

import unittest

from safety_probing.adapters.ltx_continuation import validate_prefix_geometry


class LTXPrefixGeometryTest(unittest.TestCase):
    def test_maps_dataset_future_to_internal_sequence(self) -> None:
        geometry = validate_prefix_geometry(
            prefix_frames=57,
            source_total_frames=120,
            internal_total_frames=121,
        )
        self.assertEqual(geometry.generated_future_frames, 64)
        self.assertEqual(geometry.evaluated_future_frames, 63)
        self.assertEqual(geometry.discarded_tail_frames, 1)

    def test_rejects_non_vae_aligned_prefix(self) -> None:
        with self.assertRaises(ValueError):
            validate_prefix_geometry(
                prefix_frames=60,
                source_total_frames=120,
                internal_total_frames=121,
            )


if __name__ == "__main__":
    unittest.main()
