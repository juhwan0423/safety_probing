from __future__ import annotations

import unittest

from safety_probing.adapters.vjepa import (
    future_patch_index_ranges,
    latent_candidate_energy,
    uniform_frame_indices,
)
from safety_probing.vjepa_native import (
    select_gold_pair,
    summarize_candidate_retrieval,
)


class VJEPANativeContractTest(unittest.TestCase):
    def test_sampling_preserves_observation_future_boundary(self) -> None:
        context = uniform_frame_indices(0, 64, 20)
        future = uniform_frame_indices(65, 119, 20)
        self.assertEqual((context[0], context[-1]), (0, 64))
        self.assertEqual((future[0], future[-1]), (65, 119))
        self.assertEqual(len(set(context + future)), 40)

    def test_future_mask_starts_after_all_context_tubelets(self) -> None:
        context, future = future_patch_index_ranges(
            context_frame_count=20,
            future_frame_count=20,
            resolution=256,
        )
        self.assertEqual(len(context), 2560)
        self.assertEqual(len(future), 2560)
        self.assertEqual(context.stop, future.start)

    def test_selects_first_decisive_gold_pair_in_requested_split(self) -> None:
        pairs = [
            {
                "pair_id": "b",
                "quality_tier": "gold",
                "run_A_safety_outcome": "observed_tip",
                "run_B_safety_outcome": "confirmed_recovered",
                "splits": {"main": "val"},
            },
            {
                "pair_id": "a",
                "quality_tier": "gold",
                "run_A_safety_outcome": "confirmed_recovered",
                "run_B_safety_outcome": "observed_tip",
                "splits": {"main": "val"},
            },
        ]
        outcomes = {
            "a": {"A": "confirmed_recovered", "B": "observed_tip"},
            "b": {"A": "observed_tip", "B": "confirmed_recovered"},
        }
        runs = [
            {
                "pair_id": pair_id,
                "run_label": label,
                "safety_outcome": outcomes[pair_id][label],
            }
            for pair_id in ("a", "b")
            for label in ("A", "B")
        ]
        pair, selected_runs = select_gold_pair(pairs, runs)
        self.assertEqual(pair["pair_id"], "a")
        self.assertEqual(set(selected_runs), {"A", "B"})

    def test_retrieval_reports_positive_correct_candidate_margin(self) -> None:
        summary = summarize_candidate_retrieval(
            [[0.2, 0.4], [0.5, 0.1]],
            {"A": "observed_tip", "B": "confirmed_recovered"},
        )
        self.assertEqual(summary["identity_accuracy"], 1.0)
        self.assertEqual(summary["outcome_accuracy"], 1.0)
        self.assertAlmostEqual(
            summary["queries"][0]["correct_vs_alternative_margin"], 0.2
        )

    def test_energy_accepts_query_conditioned_candidate_targets(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        predicted = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        targets = torch.tensor(
            [
                [[[1.0, 0.0]], [[0.0, 1.0]]],
                [[[1.0, 0.0]], [[0.0, 1.0]]],
            ]
        )
        energy = latent_candidate_energy(predicted, targets)
        self.assertLess(
            energy["mean_token_cosine_distance"][0][0],
            energy["mean_token_cosine_distance"][0][1],
        )
        self.assertLess(
            energy["mean_token_cosine_distance"][1][1],
            energy["mean_token_cosine_distance"][1][0],
        )


if __name__ == "__main__":
    unittest.main()
