"""
tests/test_evaluation_metrics.py

Regression tests for evaluation/metrics/ranking.py and
evaluation/evaluator.py::OfflineEvaluator.evaluate_batch, covering the two
correctness bugs found in review:

  1. precision_at_k divided by k instead of the number of recommendations
     actually returned (silently deflated whenever fewer than k items are
     recommended -- the common case for this system's candidate slates).
  2. evaluate_batch averaged every metric (including optional ones like
     pool_*_ratio) by total_users, instead of by the number of users who
     actually contributed that metric.

Run:
    python -m pytest tests/test_evaluation_metrics.py -v
"""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.evaluator import OfflineEvaluator
from evaluation.metrics.ranking import (
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK(unittest.TestCase):

    def test_fewer_than_k_recommendations_divides_by_actual_count(self):
        # Only 3 items ever recommended, k=10 -- must divide by 3, not 10.
        recommended = ["a", "b", "c"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(precision_at_k(recommended, relevant, k=10), 2 / 3)

    def test_exactly_k_recommendations(self):
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        self.assertAlmostEqual(precision_at_k(recommended, relevant, k=5), 2 / 5)

    def test_more_than_k_recommendations_truncates_to_k(self):
        recommended = ["a", "b", "c", "d", "e", "f"]
        relevant = {"a", "f"}   # "f" is beyond k=3, must not count
        self.assertAlmostEqual(precision_at_k(recommended, relevant, k=3), 1 / 3)

    def test_empty_recommendation_list_returns_zero(self):
        self.assertEqual(precision_at_k([], {"a"}, k=10), 0.0)

    def test_no_hits_returns_zero(self):
        self.assertEqual(precision_at_k(["a", "b"], {"z"}, k=5), 0.0)

    def test_all_hits_returns_one(self):
        recommended = ["a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(precision_at_k(recommended, relevant, k=10), 1.0)

    def test_fewer_than_k_no_longer_equals_half_of_true_value(self):
        """Direct regression check for the exact bug reported: with 5
        recommendations, 2 hits, and k=10, precision must equal precision@5
        (0.4), not be silently halved to 0.2 by dividing by k=10."""
        recommended = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        p5 = precision_at_k(recommended, relevant, k=5)
        p10 = precision_at_k(recommended, relevant, k=10)
        self.assertAlmostEqual(p5, 0.4)
        self.assertAlmostEqual(p10, 0.4)
        self.assertEqual(p5, p10)


class TestRecallAtK(unittest.TestCase):

    def test_partial_recall(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "x", "y"}
        self.assertAlmostEqual(recall_at_k(recommended, relevant, k=3), 1 / 3)

    def test_full_recall(self):
        recommended = ["a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(recall_at_k(recommended, relevant, k=2), 1.0)

    def test_empty_relevant_set_returns_zero(self):
        self.assertEqual(recall_at_k(["a", "b"], set(), k=5), 0.0)

    def test_no_hits_returns_zero(self):
        self.assertEqual(recall_at_k(["a"], {"z"}, k=5), 0.0)


class TestMRR(unittest.TestCase):

    def test_first_item_is_relevant(self):
        self.assertEqual(mrr(["a", "b", "c"], {"a"}), 1.0)

    def test_third_item_is_first_relevant(self):
        self.assertAlmostEqual(mrr(["a", "b", "c"], {"c"}), 1 / 3)

    def test_no_relevant_item_present_returns_zero(self):
        self.assertEqual(mrr(["a", "b"], {"z"}), 0.0)

    def test_empty_relevant_set_returns_zero(self):
        self.assertEqual(mrr(["a", "b"], set()), 0.0)

    def test_empty_recommended_list_returns_zero(self):
        self.assertEqual(mrr([], {"a"}), 0.0)


class TestMapAtK(unittest.TestCase):

    def test_all_relevant_at_top(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        self.assertAlmostEqual(map_at_k(recommended, relevant, k=3), 1.0)

    def test_empty_relevant_set_returns_zero(self):
        self.assertEqual(map_at_k(["a", "b"], set(), k=5), 0.0)

    def test_no_hits_returns_zero(self):
        self.assertEqual(map_at_k(["a", "b"], {"z"}, k=5), 0.0)

    def test_partial_hits_interleaved(self):
        # hit at rank 1 (precision 1/1) and rank 3 (precision 2/3);
        # denominator is min(len(relevant), k) = 2
        recommended = ["a", "x", "b"]
        relevant = {"a", "b"}
        expected = (1 / 1 + 2 / 3) / 2
        self.assertAlmostEqual(map_at_k(recommended, relevant, k=3), expected)


class TestNdcgAtK(unittest.TestCase):

    def test_perfect_ranking_gives_one(self):
        recommended = ["a", "b", "c"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, k=3), 1.0)

    def test_no_hits_returns_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], {"z"}, k=5), 0.0)

    def test_empty_relevant_set_returns_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], set(), k=5), 0.0)

    def test_worse_than_ideal_ranking_is_between_zero_and_one(self):
        recommended = ["x", "a", "b"]   # relevant items pushed back a rank
        relevant = {"a", "b"}
        score = ndcg_at_k(recommended, relevant, k=3)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_graded_relevance_dict_perfect_ranking_gives_one(self):
        # Ternary-style grades (0/1/2), best-graded item first.
        recommended = ["a", "b", "c"]
        relevant = {"a": 2.0, "b": 1.0}
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, k=3), 1.0)

    def test_graded_relevance_penalizes_solved_item_ranked_below_attempted(self):
        # "b" (attempted, grade 1) ranked ahead of "a" (solved, grade 2) --
        # binary NDCG couldn't distinguish this from the ideal ordering,
        # since both are just "relevant"; graded NDCG must score it < 1.0.
        recommended = ["b", "a", "c"]
        relevant = {"a": 2.0, "b": 1.0}
        score = ndcg_at_k(recommended, relevant, k=3)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_graded_relevance_matches_binary_when_grades_are_all_one(self):
        recommended = ["a", "x", "b"]
        binary = {"a", "b"}
        graded = {"a": 1.0, "b": 1.0}
        self.assertAlmostEqual(
            ndcg_at_k(recommended, binary, k=3),
            ndcg_at_k(recommended, graded, k=3),
        )

    def test_graded_relevance_zero_grade_items_do_not_count_as_hits(self):
        recommended = ["a", "b"]
        relevant = {"a": 0.0, "b": 2.0}
        # only "b" (grade 2) should contribute -- "a" has grade 0
        expected_dcg = (2 ** 2 - 1) / np.log2(2 + 1)
        expected_idcg = (2 ** 2 - 1) / np.log2(1 + 1)
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, k=2), expected_dcg / expected_idcg)

    def test_empty_relevant_dict_returns_zero(self):
        self.assertEqual(ndcg_at_k(["a", "b"], {}, k=5), 0.0)


class TestEvaluateBatchAveraging(unittest.TestCase):
    """Regression coverage for the evaluate_batch fix: optional per-user
    metrics (pool_*_ratio) must be averaged only over the users that
    actually contributed them, not over every user in the batch."""

    def test_pool_ratio_averaged_only_over_users_with_item_pools(self):
        evaluator = OfflineEvaluator(k_values=[5])
        data = [
            {"recommended": ["a", "b", "c"], "relevant": ["a"],
             "item_pools": {"a": "X", "b": "X", "c": "X"}},   # 100% pool X
            {"recommended": ["d", "e", "f"], "relevant": ["d"]},   # no item_pools at all
        ]
        summary = evaluator.evaluate_batch(data)
        # Must be 1.0 (the single contributing user was 100% pool X), not
        # 0.5 (which is what dividing by total_users=2 would incorrectly give).
        self.assertAlmostEqual(summary["pool_X_ratio"], 1.0)

    def test_metric_present_for_all_users_still_averages_over_total_users(self):
        evaluator = OfflineEvaluator(k_values=[5])
        data = [
            {"recommended": ["a"], "relevant": ["a"]},   # mrr = 1.0
            {"recommended": ["z"], "relevant": ["a"]},   # mrr = 0.0
        ]
        summary = evaluator.evaluate_batch(data)
        self.assertAlmostEqual(summary["mrr"], 0.5)

    def test_two_different_pools_each_averaged_independently(self):
        evaluator = OfflineEvaluator(k_values=[5])
        data = [
            {"recommended": ["a"], "relevant": ["a"], "item_pools": {"a": "X"}},
            {"recommended": ["b"], "relevant": ["b"], "item_pools": {"b": "Y"}},
            {"recommended": ["c"], "relevant": ["c"]},   # no item_pools
        ]
        summary = evaluator.evaluate_batch(data)
        # Each pool ratio was only ever contributed by exactly one user, at 1.0.
        self.assertAlmostEqual(summary["pool_X_ratio"], 1.0)
        self.assertAlmostEqual(summary["pool_Y_ratio"], 1.0)

    def test_empty_user_data_returns_empty_summary(self):
        evaluator = OfflineEvaluator(k_values=[5])
        self.assertEqual(evaluator.evaluate_batch([]), {})

    def test_users_with_empty_relevant_set_do_not_crash_and_are_included(self):
        evaluator = OfflineEvaluator(k_values=[5])
        data = [
            {"recommended": ["a", "b"], "relevant": []},
            {"recommended": ["a", "b"], "relevant": ["a"]},
        ]
        summary = evaluator.evaluate_batch(data)
        self.assertIn("recall@5", summary)
        self.assertIn("precision@5", summary)


class TestEvaluateUserGradedRelevance(unittest.TestCase):
    """graded_relevance is an optional, additive parameter: omitting it
    must reproduce the exact old binary-only behaviour; supplying it must
    change ONLY the ndcg@k values, never precision/recall/map/mrr."""

    def setUp(self):
        self.evaluator = OfflineEvaluator(k_values=[5])

    def test_omitting_graded_relevance_matches_old_binary_behaviour(self):
        recommended = ["b", "a", "c"]
        relevant = {"a", "b"}
        without = self.evaluator.evaluate_user(recommended, relevant)
        with_none = self.evaluator.evaluate_user(recommended, relevant, graded_relevance=None)
        self.assertEqual(without, with_none)

    def test_graded_relevance_only_changes_ndcg(self):
        recommended = ["b", "a", "c"]
        relevant = {"a", "b"}
        graded = {"a": 2.0, "b": 1.0}

        binary_result = self.evaluator.evaluate_user(recommended, relevant)
        graded_result = self.evaluator.evaluate_user(recommended, relevant, graded_relevance=graded)

        for key in ("mrr", "precision@5", "recall@5", "map@5"):
            self.assertEqual(binary_result[key], graded_result[key], f"{key} must be unaffected by graded_relevance")
        self.assertNotEqual(binary_result["ndcg@5"], graded_result["ndcg@5"])

    def test_evaluate_batch_threads_graded_relevance_through(self):
        data = [
            {"recommended": ["b", "a"], "relevant": ["a", "b"], "graded_relevance": {"a": 2.0, "b": 1.0}},
        ]
        summary = self.evaluator.evaluate_batch(data)
        # "b" (grade 1) ranked ahead of "a" (grade 2) -- not the ideal
        # graded ordering, so ndcg@5 must be < 1.0 even though every item
        # is "relevant" under the binary set (which alone would call this
        # a perfect ranking).
        self.assertLess(summary["ndcg@5"], 1.0)

    def test_evaluate_batch_without_graded_relevance_still_works(self):
        data = [{"recommended": ["a", "b"], "relevant": ["a", "b"]}]
        summary = self.evaluator.evaluate_batch(data)
        self.assertAlmostEqual(summary["ndcg@5"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
