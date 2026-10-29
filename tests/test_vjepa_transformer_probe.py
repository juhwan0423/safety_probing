from __future__ import annotations

import unittest

import torch

from safety_probing.vjepa_transformer_probe import (
    build_mean_pool_mlp_probe,
    build_temporal_transformer_probe,
    fit_token_channel_normalization,
)


class VJEPATransformerProbeTest(unittest.TestCase):
    def test_probe_preserves_batch_shape(self) -> None:
        model = build_temporal_transformer_probe(
            input_dim=8,
            temporal_tokens=4,
            model_dim=8,
            layers=2,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        model.eval()
        output = model(torch.randn(3, 4, 8))
        self.assertEqual(tuple(output.shape), (3,))

    def test_probe_rejects_changed_temporal_contract(self) -> None:
        model = build_temporal_transformer_probe(
            input_dim=8,
            temporal_tokens=4,
            model_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        with self.assertRaisesRegex(ValueError, "Expected 4 temporal tokens"):
            model(torch.randn(2, 3, 8))

    def test_position_embedding_makes_order_observable(self) -> None:
        torch.manual_seed(7)
        model = build_temporal_transformer_probe(
            input_dim=8,
            temporal_tokens=4,
            model_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
        model.eval()
        features = torch.randn(2, 4, 8)
        chronological = model(features)
        reversed_order = model(features.flip(dims=(1,)))
        self.assertFalse(torch.allclose(chronological, reversed_order))

    def test_normalization_is_fit_from_train_indices_only(self) -> None:
        features = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[1000.0, 2000.0], [3000.0, 4000.0]],
            ]
        )
        mean, std = fit_token_channel_normalization(
            features, torch.tensor([0, 1])
        )
        normalized_train = (features[:2] - mean) / std
        self.assertTrue(
            torch.allclose(normalized_train.mean(dim=(0, 1)), torch.zeros(2))
        )
        self.assertTrue(
            torch.allclose(
                normalized_train.std(dim=(0, 1), unbiased=False),
                torch.ones(2),
            )
        )

    def test_mean_pool_mlp_is_order_invariant(self) -> None:
        torch.manual_seed(11)
        model = build_mean_pool_mlp_probe(
            input_dim=8,
            temporal_tokens=4,
            hidden_dim=8,
            dropout=0.0,
        )
        model.eval()
        features = torch.randn(3, 4, 8)
        chronological = model(features)
        reversed_order = model(features.flip(dims=(1,)))
        self.assertTrue(torch.allclose(chronological, reversed_order))


if __name__ == "__main__":
    unittest.main()
