from __future__ import annotations

import unittest

from safety_probing.config import LabelingConfig
from safety_probing.labeling import (
    CONFIRMED_RECOVERED,
    OBSERVED_TIP,
    RIGHT_CENSORED,
    derive_safety_label,
    modernized_quality,
)


class SafetyLabelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LabelingConfig.from_dict({})

    def test_posthoc_support_loss_upgrades_source_stable(self) -> None:
        result = derive_safety_label(
            {"tip_over": False, "failure_time": None},
            {
                "posthoc_roll_tip_time_s": None,
                "posthoc_support_loss_time_s": 4.2,
                "last_sim_time_s": 5.0,
                "terminal_abs_roll_deg": 12.0,
                "terminal_abs_roll_rate_deg_s": 1.0,
                "longest_support_loss_streak_steps": 40,
                "terminal_support_loss_streak_steps": 40,
            },
            self.config,
        )
        self.assertEqual(result["safety_outcome"], OBSERVED_TIP)
        self.assertEqual(result["binary_tip_label"], 1)
        self.assertTrue(result["upgraded_from_source"])

    def test_unresolved_terminal_motion_is_right_censored(self) -> None:
        result = derive_safety_label(
            {"tip_over": False, "failure_time": None},
            {
                "posthoc_roll_tip_time_s": None,
                "posthoc_support_loss_time_s": None,
                "last_sim_time_s": 5.0,
                "terminal_abs_roll_deg": 3.0,
                "terminal_abs_roll_rate_deg_s": 4.0,
                "longest_support_loss_streak_steps": 0,
                "terminal_support_loss_streak_steps": 0,
            },
            self.config,
        )
        self.assertEqual(result["safety_outcome"], RIGHT_CENSORED)
        self.assertIsNone(result["binary_tip_label"])
        self.assertFalse(result["future_prediction_eligible"])

    def test_settled_non_event_is_confirmed_recovered(self) -> None:
        result = derive_safety_label(
            {"tip_over": False, "failure_time": None},
            {
                "posthoc_roll_tip_time_s": None,
                "posthoc_support_loss_time_s": None,
                "last_sim_time_s": 5.0,
                "terminal_abs_roll_deg": 4.0,
                "terminal_abs_roll_rate_deg_s": -3.0,
                "longest_support_loss_streak_steps": 0,
                "terminal_support_loss_streak_steps": 0,
            },
            self.config,
        )
        self.assertEqual(result["safety_outcome"], CONFIRMED_RECOVERED)
        self.assertEqual(result["binary_tip_label"], 0)

    def test_censored_pair_is_not_binary_eligible(self) -> None:
        quality = modernized_quality(
            outcome_a=RIGHT_CENSORED,
            outcome_b=OBSERVED_TIP,
            same_action=True,
            max_roll_deg_delta=30.0,
            max_pitch_deg_delta=0.0,
        )
        self.assertEqual(quality["tier"], "censored")
        self.assertFalse(quality["accepted"])


if __name__ == "__main__":
    unittest.main()
