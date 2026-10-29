from __future__ import annotations

import unittest

import torch

from safety_probing.ltx_real_latents import raw_rgb_video_tensor
from safety_probing.ltx_rollout_cache import extract_future_feature
from safety_probing.ltx_safety_query import (
    aggregate_rollout_predictions,
    build_ltx_safety_query_decoder,
)


class LTXRealLatentTest(unittest.TestCase):
    def test_raw_rgb_tensor_normalizes_and_repeats_tail(self) -> None:
        raw = bytes(
            [
                0,
                127,
                255,
                255,
                127,
                0,
            ]
        )
        video = raw_rgb_video_tensor(
            raw,
            source_frames=2,
            height=1,
            width=1,
            repeat_tail_frames=1,
            preprocess_device="cpu",
        )
        self.assertEqual(tuple(video.shape), (1, 3, 3, 1, 1))
        self.assertTrue(torch.equal(video[:, :, 1], video[:, :, 2]))
        self.assertAlmostEqual(float(video[0, 0, 0, 0, 0]), -1.0)
        self.assertAlmostEqual(float(video[0, 2, 0, 0, 0]), 1.0)


class LTXRolloutCacheTest(unittest.TestCase):
    def test_extracts_future_and_moves_channels_last(self) -> None:
        latent = torch.arange(1 * 2 * 4 * 2 * 3).reshape(1, 2, 4, 2, 3)
        feature = extract_future_feature(
            {"latent": latent, "prefix_latent_frames": 2},
            prefix_latent_frames=2,
        )
        self.assertEqual(tuple(feature.shape), (2, 2, 3, 2))
        self.assertEqual(float(feature[0, 0, 0, 0]), float(latent[0, 0, 2, 0, 0]))
        self.assertEqual(float(feature[1, 1, 2, 1]), float(latent[0, 1, 3, 1, 2]))

    def test_rejects_changed_prefix_geometry(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "prefix latent geometry"):
            extract_future_feature(
                {
                    "latent": torch.zeros(1, 2, 4, 2, 3),
                    "prefix_latent_frames": 1,
                },
                prefix_latent_frames=2,
            )


class LTXSafetyQueryTest(unittest.TestCase):
    def test_decoder_emits_future_physical_groups(self) -> None:
        model = build_ltx_safety_query_decoder(
            feature_dim=8,
            memory_temporal_tokens=3,
            spatial_height=2,
            spatial_width=2,
            target_temporal_tokens=4,
            state_dim=5,
            contact_count=3,
            summary_dim=4,
            model_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        output = model(torch.randn(2, 3, 2, 2, 8))
        self.assertEqual(tuple(output["static"].shape), (2, 2))
        self.assertEqual(tuple(output["future_state"].shape), (2, 4, 5))
        self.assertEqual(tuple(output["future_contact_logits"].shape), (2, 4, 3))
        self.assertEqual(tuple(output["future_contact_xy"].shape), (2, 4, 3, 2))
        self.assertEqual(tuple(output["future_summary"].shape), (2, 4))
        self.assertEqual(tuple(output["future_tip_logits"].shape), (2,))

    def test_rollout_ensemble_averages_probabilities(self) -> None:
        first = {
            "future_tip_logits": torch.logit(torch.tensor([0.2, 0.8])),
            "static": torch.tensor([[1.0, 3.0]]),
        }
        second = {
            "future_tip_logits": torch.logit(torch.tensor([0.4, 0.6])),
            "static": torch.tensor([[3.0, 5.0]]),
        }
        result = aggregate_rollout_predictions([first, second])
        self.assertTrue(
            torch.allclose(
                torch.sigmoid(result["future_tip_logits"]),
                torch.tensor([0.3, 0.7]),
            )
        )
        self.assertTrue(
            torch.equal(result["static"], torch.tensor([[2.0, 4.0]]))
        )


if __name__ == "__main__":
    unittest.main()
