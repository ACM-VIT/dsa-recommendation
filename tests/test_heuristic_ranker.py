"""
tests/test_heuristic_ranker.py

Tests the hand-tuned weighted heuristic ranker: zpd_fit (Gaussian centered
on this candidate's own avg_mastery, ported from the formerly dead-code
ranking.py), pool agreement saturation, urgency boost, similarity clamp,
variety scoring, weight correctness, and top_k behaviour.

Run:
    python -m pytest tests/test_heuristic_ranker.py -v
"""

from __future__ import annotations

import unittest

from pipeline.recommender.services.heuristic_ranker import (
    HeuristicRanker, RankedCandidate, rank_top_k,
    zpd_fit_score, calculate_variety_score,
    ZPD_TARGET_DELTA, ZPD_SIGMA, POOL_AGREEMENT_SATURATION,
    WEIGHT_ZPD_FIT, WEIGHT_POOL_AGREE, WEIGHT_URGENCY,
    WEIGHT_SIMILARITY, WEIGHT_VARIETY,
)


def _row(pid, avg_mastery=0.5, difficulty_score=0.6, pool_count=1,
         max_urgency=0.0, best_pool_score=0.0, topic_tags=None):
    return {
        "problem_id": pid,
        "pool_sources": ["course_path"] * pool_count,
        "pool_count": pool_count,
        "topic_tags": topic_tags if topic_tags is not None else ["arrays"],
        "difficulty_score": difficulty_score,
        "avg_mastery": avg_mastery,
        "max_urgency": max_urgency,
        "predicted_success": 0.68,
        "best_pool_score": best_pool_score,
    }


class TestWeightsSumToOne(unittest.TestCase):

    def test_default_weights_sum_to_one(self):
        total = (WEIGHT_ZPD_FIT + WEIGHT_POOL_AGREE + WEIGHT_URGENCY
                + WEIGHT_SIMILARITY + WEIGHT_VARIETY)
        self.assertAlmostEqual(total, 1.0, places=6)


class TestZpdFitScore(unittest.TestCase):

    def test_peak_at_mastery_plus_target_delta(self):
        avg_mastery = 0.5
        ideal = avg_mastery + ZPD_TARGET_DELTA
        self.assertAlmostEqual(zpd_fit_score(avg_mastery, ideal), 1.0, places=6)

    def test_symmetric_falloff_around_ideal(self):
        avg_mastery = 0.5
        ideal = avg_mastery + ZPD_TARGET_DELTA
        below = zpd_fit_score(avg_mastery, ideal - 0.1)
        above = zpd_fit_score(avg_mastery, ideal + 0.1)
        self.assertAlmostEqual(below, above, places=6)

    def test_far_from_ideal_scores_low(self):
        score = zpd_fit_score(avg_mastery=0.1, difficulty_score=0.95)
        self.assertLess(score, 0.1)

    def test_ideal_shifts_with_avg_mastery(self):
        # a candidate at difficulty 0.9 should fit a high-mastery user
        # better than a low-mastery user
        high_mastery_fit = zpd_fit_score(avg_mastery=0.8, difficulty_score=0.9)
        low_mastery_fit = zpd_fit_score(avg_mastery=0.1, difficulty_score=0.9)
        self.assertGreater(high_mastery_fit, low_mastery_fit)

    def test_missing_difficulty_returns_neutral(self):
        self.assertEqual(zpd_fit_score(0.5, None), 0.5)

    def test_missing_avg_mastery_returns_neutral(self):
        self.assertEqual(zpd_fit_score(None, 0.5), 0.5)


class TestVarietyScore(unittest.TestCase):

    def test_no_recent_topics_is_neutral(self):
        self.assertEqual(calculate_variety_score(["arrays"], []), 1.0)
        self.assertEqual(calculate_variety_score(["arrays"], None), 1.0)

    def test_topic_seen_recently_lowers_score(self):
        score = calculate_variety_score(["arrays"], ["graphs", "arrays"])
        self.assertLess(score, 1.0)

    def test_topic_not_in_recent_window_is_neutral(self):
        score = calculate_variety_score(["dp"], ["graphs", "arrays"])
        self.assertEqual(score, 1.0)

    def test_all_topics_recent_scores_zero(self):
        score = calculate_variety_score(["arrays", "graphs"], ["arrays", "graphs"])
        self.assertEqual(score, 0.0)

    def test_window_limits_how_far_back_counts(self):
        recent = ["arrays"] + ["filler"] * 20   # arrays is outside a small window
        score = calculate_variety_score(["arrays"], recent, window=5)
        self.assertEqual(score, 1.0)

    def test_never_negative(self):
        score = calculate_variety_score(["arrays"], ["arrays", "arrays", "arrays"])
        self.assertGreaterEqual(score, 0.0)


class TestPoolAgreement(unittest.TestCase):

    def test_single_pool_partial_credit(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", pool_count=1))
        self.assertAlmostEqual(rc.pool_agreement, 1 / POOL_AGREEMENT_SATURATION, places=4)

    def test_saturation_point_gets_full_credit(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", pool_count=POOL_AGREEMENT_SATURATION))
        self.assertAlmostEqual(rc.pool_agreement, 1.0, places=4)

    def test_beyond_saturation_stays_capped_at_one(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", pool_count=7))
        self.assertAlmostEqual(rc.pool_agreement, 1.0, places=4)

    def test_more_pools_scores_higher_than_fewer(self):
        ranker = HeuristicRanker()
        rc_one = ranker.score_one(_row("p1", pool_count=1))
        rc_three = ranker.score_one(_row("p2", pool_count=3))
        self.assertGreater(rc_three.score, rc_one.score)


class TestUrgencyBoost(unittest.TestCase):

    def test_higher_urgency_scores_higher(self):
        ranker = HeuristicRanker()
        rc_low = ranker.score_one(_row("p1", max_urgency=0.1))
        rc_high = ranker.score_one(_row("p2", max_urgency=0.9))
        self.assertGreater(rc_high.score, rc_low.score)

    def test_missing_urgency_defaults_to_zero(self):
        ranker = HeuristicRanker()
        row = _row("p1")
        row["max_urgency"] = None
        rc = ranker.score_one(row)
        self.assertEqual(rc.urgency_boost, 0.0)


class TestSimilarityClamp(unittest.TestCase):

    def test_similarity_clamped_above_one(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", best_pool_score=1.5))
        self.assertLessEqual(rc.similarity_score, 1.0)

    def test_similarity_clamped_below_zero(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1", best_pool_score=-0.3))
        self.assertGreaterEqual(rc.similarity_score, 0.0)

    def test_missing_similarity_defaults_to_zero(self):
        ranker = HeuristicRanker()
        row = _row("p1")
        row["best_pool_score"] = None
        rc = ranker.score_one(row)
        self.assertEqual(rc.similarity_score, 0.0)


class TestVarietyInRanker(unittest.TestCase):

    def test_recent_topic_candidate_ranks_below_fresh_topic_candidate(self):
        ranker = HeuristicRanker()
        rows = [
            _row("fresh", topic_tags=["dp"]),
            _row("stale", topic_tags=["arrays"]),
        ]
        ranked = ranker.rank(rows, recent_topics=["arrays", "arrays"])
        self.assertEqual(ranked[0].problem_id, "fresh")

    def test_no_recent_topics_leaves_variety_neutral(self):
        ranker = HeuristicRanker()
        rc = ranker.score_one(_row("p1"), recent_topics=None)
        self.assertEqual(rc.variety_score, 1.0)


class TestRankingOrder(unittest.TestCase):

    def test_rank_sorts_best_first(self):
        ranker = HeuristicRanker()
        rows = [
            _row("low", avg_mastery=0.1, difficulty_score=0.95, pool_count=1),
            _row("high", avg_mastery=0.5, difficulty_score=0.6, pool_count=3, max_urgency=0.5),
        ]
        ranked = ranker.rank(rows)
        self.assertEqual(ranked[0].problem_id, "high")
        self.assertEqual(ranked[1].problem_id, "low")

    def test_top_k_limits_result_size(self):
        rows = [_row(f"p{i}") for i in range(20)]
        result = rank_top_k(rows, k=10)
        self.assertEqual(len(result), 10)

    def test_top_k_smaller_input_returns_all(self):
        rows = [_row(f"p{i}") for i in range(3)]
        result = rank_top_k(rows, k=10)
        self.assertEqual(len(result), 3)

    def test_empty_input_returns_empty(self):
        result = rank_top_k([], k=10)
        self.assertEqual(result, [])

    def test_top_k_dicts_have_rank_score(self):
        rows = [_row("p1")]
        result = rank_top_k(rows, k=10)
        self.assertIn("rank_score", result[0])
        self.assertIn("rank_components", result[0])

    def test_original_row_fields_preserved_in_output(self):
        rows = [_row("p1")]
        result = rank_top_k(rows, k=10)
        self.assertEqual(result[0]["problem_id"], "p1")
        self.assertIn("topic_tags", result[0])


class TestCustomWeights(unittest.TestCase):

    def test_custom_weights_change_ranking(self):
        """
        With urgency weighted to dominate everything else, a high-urgency/
        poor-zpd-fit candidate should outrank a perfect-fit/zero-urgency
        one -- proves weights are actually used, not hardcoded internally.
        """
        ranker = HeuristicRanker(
            weight_zpd_fit=0.05, weight_pool_agree=0.05,
            weight_urgency=0.80, weight_similarity=0.05, weight_variety=0.05,
        )
        rows = [
            _row("perfect_fit", avg_mastery=0.5, difficulty_score=0.6, max_urgency=0.0),
            _row("urgent", avg_mastery=0.1, difficulty_score=0.95, max_urgency=0.95),
        ]
        ranked = ranker.rank(rows)
        self.assertEqual(ranked[0].problem_id, "urgent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
