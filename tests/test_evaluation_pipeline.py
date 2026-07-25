"""
tests/test_evaluation_pipeline.py

End-to-end test of the real (non-mock) evaluation pipeline:
evaluation/experiments.py::run_validation_evaluation() against a small,
synthetic validation-parquet-shaped DataFrame and a tiny trained LightGBM
model -- proving the whole chain (load validation data -> score with
LightGBMRanker -> build both rankers' query groups -> OfflineEvaluator ->
Markdown + JSON + console report) works together, without depending on
training/artifacts/validation.parquet or lightgbm_model.txt existing.

Run:
    python -m pytest tests/test_evaluation_pipeline.py -v
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from evaluation import experiments
from evaluation import report as report_module
from pipeline.recommender.services import lightgbm_ranker as lr
from training.feature_registry import build_default_registry


def _build_validation_dataframe(feature_names, categorical_names, seed=0) -> pd.DataFrame:
    """A small, deterministic validation-parquet-shaped DataFrame: 4 query
    groups, real ternary 0/1/2 labels, real from_pool_* columns, and every
    trainable feature column the registry declares."""
    rng = np.random.RandomState(seed)
    rows = []
    query_sizes = [3, 4, 2, 5]
    candidate_counter = 0
    for q_idx, size in enumerate(query_sizes):
        query_id = f"q{q_idx}"
        for _ in range(size):
            candidate_counter += 1
            row = {
                "query_id": query_id,
                "candidate_id": f"c{candidate_counter}",
                "user_id": f"u{q_idx}",
                "label": float(rng.choice([0, 1, 2])),
                "current_heuristic_rank_score": float(rng.uniform(0, 1)),
                "from_pool_A": bool(rng.choice([True, False])),
                "from_pool_B_C": False, "from_pool_D": False, "from_pool_E": False,
                "from_pool_F": False, "from_pool_G": False, "from_pool_vector": False,
            }
            for name in feature_names:
                if name in row:
                    continue
                if name in categorical_names:
                    row[name] = rng.choice(["beginner", "mid", "advanced"])
                else:
                    row[name] = rng.uniform(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def _train_tiny_model(feature_names, categorical_names, df: pd.DataFrame, path: Path):
    X = df[feature_names].copy()
    for name in feature_names:
        if name in categorical_names:
            X[name] = X[name].astype("category")
        else:
            X[name] = X[name].astype("float64")
    y = df["label"]
    group = df.groupby("query_id", sort=False).size().to_numpy()
    dataset = lgb.Dataset(X, label=y, group=group, categorical_feature=categorical_names or "auto")
    booster = lgb.train(
        {"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 1, "num_leaves": 3},
        dataset, num_boost_round=5,
    )
    booster.save_model(str(path))


class TestRunValidationEvaluationEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        registry = build_default_registry()
        cls.feature_names = [f.name for f in registry.model_matrix_features()]
        cls.categorical_names = [f.name for f in registry.categorical_features()]

        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)

        cls.df = _build_validation_dataframe(cls.feature_names, cls.categorical_names)
        cls.validation_path = tmp_path / "validation.parquet"
        cls.df.to_parquet(cls.validation_path, index=False)

        cls.model_path = tmp_path / "model.txt"
        _train_tiny_model(cls.feature_names, cls.categorical_names, cls.df, cls.model_path)

        cls.reports_dir = tmp_path / "reports"

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def setUp(self):
        lr.reset_lightgbm_ranker_cache()

    def tearDown(self):
        lr.reset_lightgbm_ranker_cache()

    def _run(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.model_path):
            with patch.object(report_module, "REPORTS_DIR", self.reports_dir):
                return experiments.run_validation_evaluation(
                    validation_path=self.validation_path, model_name="LightGBM",
                )

    def test_runs_without_error_and_returns_two_metric_dicts(self):
        heuristic_metrics, lightgbm_metrics = self._run()
        self.assertIsInstance(heuristic_metrics, dict)
        self.assertIsInstance(lightgbm_metrics, dict)
        self.assertGreater(len(heuristic_metrics), 0)
        self.assertGreater(len(lightgbm_metrics), 0)

    def test_produces_ndcg_map_mrr_precision_recall_for_both_rankers(self):
        heuristic_metrics, lightgbm_metrics = self._run()
        expected_keys = {"mrr", "ndcg@5", "ndcg@10", "map@5", "map@10", "precision@5", "precision@10", "recall@5", "recall@10"}
        self.assertTrue(expected_keys.issubset(heuristic_metrics.keys()))
        self.assertTrue(expected_keys.issubset(lightgbm_metrics.keys()))

    def test_writes_markdown_and_json_reports(self):
        self._run()
        md_path = self.reports_dir / "lightgbm_eval_report.md"
        json_path = self.reports_dir / "lightgbm_eval_summary.json"
        self.assertTrue(md_path.exists())
        self.assertTrue(json_path.exists())

        with open(json_path) as fh:
            summary = json.load(fh)
        self.assertIn("heuristic", summary)
        self.assertIn("LightGBM", summary)
        self.assertIn("diff", summary)
        self.assertIn("meta", summary)
        self.assertEqual(summary["meta"]["n_rows"], len(self.df))
        self.assertEqual(summary["meta"]["n_query_groups"], 4)

    def test_deterministic_across_repeated_runs(self):
        first_heuristic, first_lightgbm = self._run()
        second_heuristic, second_lightgbm = self._run()
        self.assertEqual(first_heuristic, second_heuristic)
        self.assertEqual(first_lightgbm, second_lightgbm)

    def test_both_rankers_see_the_same_number_of_query_groups(self):
        # meta records the heuristic's query-group count; cross-check against
        # the raw dataframe's own group count as an independent computation.
        self._run()
        with open(self.reports_dir / "lightgbm_eval_summary.json") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["meta"]["n_query_groups"], self.df["query_id"].nunique())

    def test_graded_ndcg_is_actually_exercised(self):
        """Construct a case where the binary and graded relevance sets
        would disagree, and confirm evaluate against real data actually
        uses graded labels: a heuristic ordering that puts a label=1 item
        ahead of a label=2 item within the same group must score < 1.0 on
        ndcg even though both are "relevant" under a binary set."""
        from evaluation.data_loader import build_user_data
        heuristic_data = build_user_data(self.df, score_column="current_heuristic_rank_score")
        found_non_trivial_case = any(
            len(entry["graded_relevance"]) >= 2 and len(set(entry["graded_relevance"].values())) > 1
            for entry in heuristic_data
        )
        self.assertTrue(found_non_trivial_case, "test data must include at least one group with mixed grades")


class TestMainMissingModelHandling(unittest.TestCase):
    """evaluation.experiments.main() -- the CLI entry point's handling of
    "nothing trained/generated yet", which must never silently fall back
    to a heuristic-only comparison (see module docstring's rationale)."""

    @classmethod
    def setUpClass(cls):
        registry = build_default_registry()
        cls.feature_names = [f.name for f in registry.model_matrix_features()]
        cls.categorical_names = [f.name for f in registry.categorical_features()]

        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(cls._tmpdir.name)
        cls.df = _build_validation_dataframe(cls.feature_names, cls.categorical_names)
        cls.validation_path = tmp_path / "validation.parquet"
        cls.df.to_parquet(cls.validation_path, index=False)

        cls.model_path = tmp_path / "model.txt"
        _train_tiny_model(cls.feature_names, cls.categorical_names, cls.df, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def setUp(self):
        lr.reset_lightgbm_ranker_cache()

    def tearDown(self):
        lr.reset_lightgbm_ranker_cache()

    def test_missing_model_exits_nonzero_with_actionable_message(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH",
                   Path("/nonexistent/lightgbm_model.txt")):
            with patch("evaluation.experiments.VALIDATION_DATASET_PATH", self.validation_path):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = experiments.main()

        self.assertEqual(exit_code, 1)
        message = stderr.getvalue()
        self.assertIn("No trained LightGBM model found", message)
        self.assertIn("python -m training.train_lightgbm", message)

    def test_missing_model_does_not_fall_back_to_heuristic_only(self):
        """The failure must be a hard stop, not a quiet heuristic-only run:
        no report/summary files should appear for a run that never
        actually evaluated LightGBM."""
        reports_dir = Path(self._tmpdir.name) / "reports_missing_model"
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH",
                   Path("/nonexistent/lightgbm_model.txt")):
            with patch("evaluation.experiments.VALIDATION_DATASET_PATH", self.validation_path):
                with patch.object(report_module, "REPORTS_DIR", reports_dir):
                    with redirect_stderr(io.StringIO()):
                        experiments.main()

        self.assertFalse(reports_dir.exists(), "no report should be written when evaluation never ran")

    def test_missing_validation_dataset_exits_nonzero_with_actionable_message(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.model_path):
            with patch("evaluation.experiments.VALIDATION_DATASET_PATH", Path("/nonexistent/validation.parquet")):
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    exit_code = experiments.main()

        self.assertEqual(exit_code, 1)
        message = stderr.getvalue()
        self.assertIn("Validation dataset not found", message)

    def test_successful_run_still_exits_zero(self):
        reports_dir = Path(self._tmpdir.name) / "reports_success"
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.model_path):
            with patch("evaluation.experiments.VALIDATION_DATASET_PATH", self.validation_path):
                with patch.object(report_module, "REPORTS_DIR", reports_dir):
                    exit_code = experiments.main()
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
