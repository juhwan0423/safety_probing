from __future__ import annotations

import unittest

from safety_probing.vjepa_linear_probe import (
    average_precision,
    binary_auroc,
    empirical_prevalence_scores,
    pairwise_accuracy,
    select_balanced_threshold,
    spearman_correlation,
)


class VJEPALinearProbeMetricsTest(unittest.TestCase):
    def test_binary_ranking_metrics_are_perfect_for_separated_scores(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(binary_auroc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)

    def test_threshold_is_selected_from_validation_only_values(self) -> None:
        threshold = select_balanced_threshold(
            [0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9]
        )
        self.assertEqual(threshold, 0.5)

    def test_pairwise_accuracy_gives_half_credit_to_ties(self) -> None:
        report = pairwise_accuracy(
            [0.4, 0.4, 0.8],
            [
                {"tip_index": 1, "stable_index": 0},
                {"tip_index": 2, "stable_index": 0},
            ],
        )
        self.assertEqual(report["tie_count"], 1)
        self.assertEqual(report["accuracy"], 0.75)

    def test_spearman_handles_monotonic_relationship(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([3.0, 1.0, 2.0], [30.0, 10.0, 20.0]),
            1.0,
        )

    def test_prevalence_control_uses_only_train_groups(self) -> None:
        records = [
            {"split": "train", "binary_tip_label": 1, "speed_label": "fast"},
            {"split": "train", "binary_tip_label": 0, "speed_label": "slow"},
            {"split": "test", "binary_tip_label": 1, "speed_label": "fast"},
            {"split": "test", "binary_tip_label": 0, "speed_label": "new"},
        ]
        scores, seen, specification = empirical_prevalence_scores(
            records, ("speed_label",)
        )
        self.assertAlmostEqual(float(scores[2]), 2 / 3)
        self.assertAlmostEqual(float(scores[3]), 0.5)
        self.assertEqual(seen.tolist(), [True, True, True, False])
        self.assertEqual(specification["train_group_count"], 2)


if __name__ == "__main__":
    unittest.main()
