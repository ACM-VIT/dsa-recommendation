"""
tests/test_evaluation_data_loader.py

Tests evaluation/data_loader.py: query-group boundary detection over a
DataFrame, pool attribution per row, and build_user_data()'s construction
of OfflineEvaluator-shaped input (recommended ordering, binary relevant
set, graded_relevance dict, item_pools) from a validation-parquet-shaped
DataFrame.

Run:
    python -m pytest tests/test_evaluation_data_loader.py -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from evaluation.data_loader import build_user_data, iter_query_groups


def _row(query_id, candidate_id, label, score, pools=()):
    row = {
        "query_id": query_id, "candidate_id": candidate_id, "label": float(label),
        "score": float(score),
        "from_pool_A": "A" in pools, "from_pool_B_C": "B_C" in pools,
        "from_pool_D": "D" in pools, "from_pool_E": "E" in pools,
        "from_pool_F": "F" in pools, "from_pool_G": "G" in pools,
        "from_pool_vector": "vector" in pools,
    }
    return row


class TestIterQueryGroups(unittest.TestCase):

    def test_splits_contiguous_blocks_by_query_id(self):
        df = pd.DataFrame([
            _row("q1", "a", 1, 0.9), _row("q1", "b", 0, 0.5),
            _row("q2", "c", 2, 0.8), _row("q2", "d", 0, 0.4), _row("q2", "e", 1, 0.3),
        ])
        groups = list(iter_query_groups(df))
        self.assertEqual([qid for qid, _ in groups], ["q1", "q2"])
        self.assertEqual(len(groups[0][1]), 2)
        self.assertEqual(len(groups[1][1]), 3)

    def test_empty_dataframe_yields_nothing(self):
        df = pd.DataFrame(columns=["query_id", "candidate_id", "label", "score"])
        self.assertEqual(list(iter_query_groups(df)), [])

    def test_single_group(self):
        df = pd.DataFrame([_row("q1", "a", 1, 0.9), _row("q1", "b", 0, 0.1)])
        groups = list(iter_query_groups(df))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0], "q1")


class TestBuildUserData(unittest.TestCase):

    def test_recommended_is_ordered_by_score_column_descending(self):
        df = pd.DataFrame([
            _row("q1", "a", 1, 0.2), _row("q1", "b", 0, 0.9), _row("q1", "c", 2, 0.5),
        ])
        user_data = build_user_data(df, score_column="score")
        self.assertEqual(user_data[0]["recommended"], ["b", "c", "a"])

    def test_one_entry_per_query_group(self):
        df = pd.DataFrame([
            _row("q1", "a", 1, 0.9), _row("q1", "b", 0, 0.1),
            _row("q2", "c", 2, 0.5),
        ])
        user_data = build_user_data(df, score_column="score")
        self.assertEqual(len(user_data), 2)
        self.assertEqual({e["query_id"] for e in user_data}, {"q1", "q2"})

    def test_relevant_set_is_binary_label_greater_than_zero(self):
        df = pd.DataFrame([
            _row("q1", "a", 0, 0.9), _row("q1", "b", 1, 0.5), _row("q1", "c", 2, 0.1),
        ])
        user_data = build_user_data(df, score_column="score")
        self.assertEqual(user_data[0]["relevant"], {"b", "c"})

    def test_graded_relevance_keeps_the_actual_grade_for_positive_labels_only(self):
        df = pd.DataFrame([
            _row("q1", "a", 0, 0.9), _row("q1", "b", 1, 0.5), _row("q1", "c", 2, 0.1),
        ])
        user_data = build_user_data(df, score_column="score")
        self.assertEqual(user_data[0]["graded_relevance"], {"b": 1.0, "c": 2.0})
        self.assertNotIn("a", user_data[0]["graded_relevance"])

    def test_item_pools_picks_the_true_pool_column_per_row(self):
        df = pd.DataFrame([
            _row("q1", "a", 1, 0.9, pools=("A",)),
            _row("q1", "b", 0, 0.5, pools=("vector",)),
        ])
        user_data = build_user_data(df, score_column="score")
        self.assertEqual(user_data[0]["item_pools"], {"a": "A", "b": "vector"})

    def test_item_with_no_pool_column_true_is_unknown(self):
        df = pd.DataFrame([_row("q1", "a", 1, 0.9, pools=())])
        user_data = build_user_data(df, score_column="score")
        self.assertEqual(user_data[0]["item_pools"]["a"], "unknown")

    def test_heuristic_and_lightgbm_orderings_share_identical_ground_truth(self):
        """The whole point of scoring from the same df with two different
        score columns: query groups, relevant sets, and graded_relevance
        must be identical between the two -- only `recommended` order may differ."""
        df = pd.DataFrame([
            _row("q1", "a", 1, 0.9), _row("q1", "b", 0, 0.5), _row("q1", "c", 2, 0.1),
        ])
        df["lgbm_score"] = [0.1, 0.9, 0.5]   # deliberately different ordering

        heuristic_data = build_user_data(df, score_column="score")
        lgbm_data = build_user_data(df, score_column="lgbm_score")

        self.assertEqual(heuristic_data[0]["relevant"], lgbm_data[0]["relevant"])
        self.assertEqual(heuristic_data[0]["graded_relevance"], lgbm_data[0]["graded_relevance"])
        self.assertEqual(set(heuristic_data[0]["recommended"]), set(lgbm_data[0]["recommended"]))
        self.assertNotEqual(heuristic_data[0]["recommended"], lgbm_data[0]["recommended"])

    def test_deterministic_given_same_input(self):
        df = pd.DataFrame([
            _row("q1", "a", 1, 0.9), _row("q1", "b", 0, 0.5), _row("q1", "c", 2, 0.5),
        ])
        r1 = build_user_data(df, score_column="score")
        r2 = build_user_data(df, score_column="score")
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
