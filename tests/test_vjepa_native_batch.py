from __future__ import annotations

import unittest

from safety_probing.vjepa_native_batch import summarize_batch_records


class VJEPANativeBatchTest(unittest.TestCase):
    def test_summary_ignores_failed_pairs_in_retrieval_metric(self) -> None:
        complete = {
            "status": "complete",
            "split": "val",
            "primary_result": {
                "all_queries_correct": False,
                "queries": [
                    {
                        "correct_outcome": True,
                        "correct_vs_alternative_margin": 0.2,
                    },
                    {
                        "correct_outcome": False,
                        "correct_vs_alternative_margin": -0.1,
                    },
                ],
            },
        }
        failed = {"status": "failed", "split": "val"}

        summary = summarize_batch_records(
            [complete, failed], requested_pair_count=2
        )

        self.assertEqual(summary["completed_pair_count"], 1)
        self.assertEqual(summary["status_counts"], {"complete": 1, "failed": 1})
        self.assertEqual(summary["primary_query_count"], 2)
        self.assertEqual(summary["primary_outcome_retrieval_accuracy"], 0.5)
        self.assertEqual(
            summary["by_split"]["val"]["outcome_retrieval_accuracy"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
