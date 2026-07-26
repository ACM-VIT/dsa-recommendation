"""
tests/test_lightgbm_ranker.py

Tests pipeline/recommender/services/lightgbm_ranker.py: model loading
(success + failure modes), feature-schema validation against
FeatureRegistry, live scoring/reranking (score_candidates/rerank) and
offline/batch scoring (score_dataframe), config-driven ranker switching
(RANKER=heuristic/lightgbm/hybrid), and automatic fallback to the
heuristic ranker on any failure.

Uses a tiny LightGBM model trained ad hoc in-memory (matching the real
registry's feature names/order) rather than depending on
training/artifacts/lightgbm_model.txt existing -- these tests must pass on
a fresh checkout before any training has run.

Run:
    python -m pytest tests/test_lightgbm_ranker.py -v
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.recommender.models.user_graph import ConceptEdge, EdgeType, UserGraph, UserNode
from pipeline.recommender.services.adaptive_difficulty import DifficultyPlan
from pipeline.recommender.services.candidate_filtering import MergedCandidate
from pipeline.recommender.services.heuristic_ranker import HeuristicRanker
from training.feature_registry import build_default_registry

from pipeline.recommender.services import lightgbm_ranker as lr


def _train_tiny_model(feature_names, categorical_names, path: Path, n_rows: int = 40, seed: int = 0):
    """A structurally-valid LightGBM model whose feature_name() matches
    the given list exactly -- objective/accuracy are irrelevant here, only
    the schema (names + order) that load_model() validates."""
    rng = np.random.RandomState(seed)
    data = {}
    for name in feature_names:
        if name in categorical_names:
            data[name] = pd.Categorical(rng.choice(["beginner", "mid", "advanced"], size=n_rows))
        else:
            data[name] = rng.uniform(0, 1, size=n_rows)
    X = pd.DataFrame(data)[feature_names]
    y = rng.uniform(0, 2, size=n_rows)

    dataset = lgb.Dataset(X, label=y, categorical_feature=categorical_names or "auto")
    booster = lgb.train(
        {"objective": "regression", "verbose": -1, "min_data_in_leaf": 1, "num_leaves": 3},
        dataset, num_boost_round=3,
    )
    booster.save_model(str(path))


def _make_graph(user_id="u1") -> UserGraph:
    graph = UserGraph(user=UserNode(user_id=user_id))
    graph.add_concept_edge(ConceptEdge(
        concept_slug="array", edge_type=EdgeType.LEARNING, mastery_score=0.5, urgency=0.2,
    ))
    graph.add_concept_edge(ConceptEdge(
        concept_slug="graph", edge_type=EdgeType.LEARNING, mastery_score=0.3, urgency=0.6,
    ))
    return graph


def _make_candidates() -> list[MergedCandidate]:
    return [
        MergedCandidate("p1", ["A"], best_score=0.5, topic_tags=["array"],
                         difficulty_score=0.4, predicted_success=0.68),
        MergedCandidate("p2", ["D"], best_score=0.3, topic_tags=["graph"],
                         difficulty_score=0.6, predicted_success=0.5),
        MergedCandidate("p3", ["A", "G"], best_score=0.7, topic_tags=["array", "graph"],
                         difficulty_score=0.3, predicted_success=0.75),
    ]


class LightGBMRankerTestBase(unittest.TestCase):
    """Builds one shared valid tiny model file for the whole module."""

    @classmethod
    def setUpClass(cls):
        registry = build_default_registry()
        cls.feature_names = [f.name for f in registry.model_matrix_features()]
        cls.categorical_names = [f.name for f in registry.categorical_features()]

        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.valid_model_path = Path(cls._tmpdir.name) / "valid_model.txt"
        _train_tiny_model(cls.feature_names, cls.categorical_names, cls.valid_model_path)

        # A model trained on the WRONG feature order/names, for the
        # feature-mismatch failure path.
        cls.mismatched_model_path = Path(cls._tmpdir.name) / "mismatched_model.txt"
        _train_tiny_model(list(reversed(cls.feature_names)), cls.categorical_names, cls.mismatched_model_path)

        # A valid sidecar metadata file for the valid model, for
        # get_model_info() tests.
        cls.metadata_path = Path(cls._tmpdir.name) / "model_metadata.json"
        cls.metadata_content = {
            "model_version": "lgbm-test-version",
            "training_timestamp": "20260101T000000Z",
            "git_commit_hash": "deadbeef",
            "lightgbm_version": "4.7.0",
            "feature_registry_version": 1,
            "feature_count": len(cls.feature_names),
            "dataset_version": "abc123",
            "training_rows": 100, "validation_rows": 25,
            "query_groups_train": 20, "query_groups_validation": 5,
            "random_seed": 42, "best_iteration": 17,
            "training_parameters": {}, "training_metrics": {},
            "model_file_name": "valid_model.txt",
        }
        with open(cls.metadata_path, "w", encoding="utf-8") as fh:
            json.dump(cls.metadata_content, fh)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def setUp(self):
        lr.reset_lightgbm_ranker_cache()

    def tearDown(self):
        lr.reset_lightgbm_ranker_cache()

    def _fake_feature_frame(self, n_rows: int = 10) -> pd.DataFrame:
        """An already-feature-extracted DataFrame (validation.parquet's
        shape) for score_dataframe()'s offline/batch path -- distinct from
        _make_candidates()/_make_graph() above, which build live objects
        for the score_candidates()/rerank() path."""
        rng = np.random.RandomState(1)
        data = {}
        for name in self.feature_names:
            if name in self.categorical_names:
                data[name] = rng.choice(["beginner", "mid", "advanced"], size=n_rows)
            else:
                data[name] = rng.uniform(0, 1, size=n_rows)
        return pd.DataFrame(data)


class TestLoadModel(LightGBMRankerTestBase):

    def test_load_model_succeeds_with_matching_schema(self):
        ranker = lr.LightGBMRanker(model_path=self.valid_model_path)
        self.assertFalse(ranker.is_loaded)
        ranker.load_model()
        self.assertTrue(ranker.is_loaded)
        self.assertEqual(list(ranker._booster.feature_name()), self.feature_names)

    def test_load_model_raises_on_missing_file(self):
        ranker = lr.LightGBMRanker(model_path=Path("/nonexistent/does_not_exist.txt"))
        with self.assertRaises(lr.LightGBMRankerError):
            ranker.load_model()
        self.assertFalse(ranker.is_loaded)

    def test_load_model_raises_on_feature_mismatch(self):
        ranker = lr.LightGBMRanker(model_path=self.mismatched_model_path)
        with self.assertRaises(lr.LightGBMRankerError) as ctx:
            ranker.load_model()
        self.assertIn("feature order", str(ctx.exception))
        self.assertFalse(ranker.is_loaded)


class TestScoreDataframe(LightGBMRankerTestBase):
    """score_dataframe() -- the offline/batch scoring path evaluation/
    uses to score an already-feature-extracted DataFrame (e.g.
    validation.parquet) directly, without reconstructing UserGraph/
    MergedCandidate objects. Shares _prepare_feature_matrix() with
    score_candidates(); TestScoreAndRerank below covers the live path."""

    def setUp(self):
        super().setUp()
        self.ranker = lr.LightGBMRanker(model_path=self.valid_model_path)

    def test_raises_before_load(self):
        df = self._fake_feature_frame()
        with self.assertRaises(lr.LightGBMRankerError):
            self.ranker.score_dataframe(df)

    def test_returns_one_score_per_row_no_nan(self):
        self.ranker.load_model()
        df = self._fake_feature_frame(n_rows=15)
        scores = self.ranker.score_dataframe(df)
        self.assertEqual(len(scores), 15)
        self.assertFalse(np.isnan(scores).any())

    def test_raises_on_missing_feature_column(self):
        self.ranker.load_model()
        df = self._fake_feature_frame().drop(columns=[self.feature_names[0]])
        with self.assertRaises(lr.LightGBMRankerError):
            self.ranker.score_dataframe(df)

    def test_ignores_extra_columns_not_in_registry(self):
        self.ranker.load_model()
        df = self._fake_feature_frame()
        df["query_id"] = "q1"
        df["candidate_id"] = [f"c{i}" for i in range(len(df))]
        scores = self.ranker.score_dataframe(df)
        self.assertEqual(len(scores), len(df))

    def test_deterministic_across_repeated_calls(self):
        self.ranker.load_model()
        df = self._fake_feature_frame(n_rows=20)
        s1 = self.ranker.score_dataframe(df)
        s2 = self.ranker.score_dataframe(df)
        np.testing.assert_array_equal(s1, s2)


class TestScoreAndRerank(LightGBMRankerTestBase):

    def setUp(self):
        super().setUp()
        self.ranker = lr.LightGBMRanker(model_path=self.valid_model_path)
        self.graph = _make_graph()
        self.plan = DifficultyPlan(avg_mastery=0.4, level="mid")
        self.candidates = _make_candidates()

    def test_score_candidates_raises_before_load(self):
        with self.assertRaises(lr.LightGBMRankerError):
            self.ranker.score_candidates(self.candidates, self.graph, self.plan)

    def test_score_candidates_returns_one_score_per_candidate_no_nan(self):
        self.ranker.load_model()
        scored = self.ranker.score_candidates(self.candidates, self.graph, self.plan)
        self.assertEqual(len(scored), len(self.candidates))
        self.assertEqual({sc.candidate.problem_id for sc in scored},
                          {c.problem_id for c in self.candidates})
        for sc in scored:
            self.assertFalse(pd.isna(sc.score))

    def test_score_candidates_empty_list_returns_empty(self):
        self.ranker.load_model()
        self.assertEqual(self.ranker.score_candidates([], self.graph, self.plan), [])

    def test_rerank_preserves_candidate_objects_sorted_by_score(self):
        self.ranker.load_model()
        reranked = self.ranker.rerank(self.candidates, self.graph, self.plan)
        self.assertEqual({c.problem_id for c in reranked}, {c.problem_id for c in self.candidates})
        scores = [sc.score for sc in self.ranker.score_candidates(reranked, self.graph, self.plan)]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestCatalogMetadataReachesFeatureExtraction(LightGBMRankerTestBase):
    """Proves score_candidates()'s catalog_metadata_by_problem_id parameter
    actually reaches extract_features() per-candidate (keyed correctly by
    problem_id, defaulting to None/{} when absent) -- previously nothing
    in recommend.py ever populated this parameter, so it was always {},
    and company_tag_count/frequency/rating/asked_by_faang were always NaN
    at inference time even though the plumbing already existed."""

    def setUp(self):
        super().setUp()
        self.ranker = lr.LightGBMRanker(model_path=self.valid_model_path)
        self.ranker.load_model()
        self.graph = _make_graph()
        self.plan = DifficultyPlan(avg_mastery=0.4, level="mid")
        self.candidates = _make_candidates()

    def test_catalog_metadata_passed_per_candidate_by_problem_id(self):
        catalog_metadata_by_problem_id = {
            "p1": {"companies": ["Google", "Amazon"], "frequency": 0.9,
                   "rating": 0.95, "asked_by_faang": True},
        }
        with patch(
            "pipeline.recommender.services.lightgbm_ranker.extract_features",
            wraps=lr.extract_features,
        ) as spy:
            self.ranker.score_candidates(
                self.candidates, self.graph, self.plan,
                catalog_metadata_by_problem_id=catalog_metadata_by_problem_id,
            )

        seen_by_problem_id = {
            call.args[2].problem_id: call.kwargs["catalog_metadata"]
            for call in spy.call_args_list
        }
        self.assertEqual(seen_by_problem_id["p1"], catalog_metadata_by_problem_id["p1"])
        # p2/p3 have no entry -- must be passed through as None, not fabricated.
        self.assertIsNone(seen_by_problem_id["p2"])
        self.assertIsNone(seen_by_problem_id["p3"])

    def test_company_tag_count_populated_in_resulting_feature_matrix_when_catalog_metadata_present(self):
        catalog_metadata_by_problem_id = {
            "p1": {"companies": ["Google", "Amazon"], "frequency": 0.9,
                   "rating": 0.95, "asked_by_faang": True},
        }
        captured_dfs = []
        real_prepare = self.ranker._prepare_feature_matrix

        def _capture(df):
            captured_dfs.append(df.copy())
            return real_prepare(df)

        with patch.object(self.ranker, "_prepare_feature_matrix", side_effect=_capture):
            self.ranker.score_candidates(
                self.candidates, self.graph, self.plan,
                catalog_metadata_by_problem_id=catalog_metadata_by_problem_id,
            )

        df = captured_dfs[0]
        p1_row = df[df["candidate_id"] == "p1"].iloc[0] if "candidate_id" in df.columns else None
        if p1_row is None:
            # candidate_id may not be a raw column; fall back to positional
            # match since _make_candidates()'s first candidate is p1.
            p1_row = df.iloc[0]
        self.assertEqual(p1_row["company_tag_count"], 2.0)
        self.assertEqual(p1_row["frequency"], 0.9)
        self.assertEqual(p1_row["rating"], 0.95)
        self.assertTrue(p1_row["asked_by_faang"])

    def test_no_catalog_metadata_leaves_columns_unfabricated(self):
        with patch(
            "pipeline.recommender.services.lightgbm_ranker.extract_features",
            wraps=lr.extract_features,
        ) as spy:
            self.ranker.score_candidates(self.candidates, self.graph, self.plan)

        for call in spy.call_args_list:
            self.assertIsNone(call.kwargs["catalog_metadata"])


class TestModelMetadataIntegration(LightGBMRankerTestBase):
    """get_model_info() -- metadata loaded alongside the model, and the
    missing-metadata fallback (model still loads/serves fine without it)."""

    def test_get_model_info_returns_none_before_load(self):
        ranker = lr.LightGBMRanker(model_path=self.valid_model_path, metadata_path=self.metadata_path)
        self.assertIsNone(ranker.get_model_info())

    def test_get_model_info_returns_metadata_after_successful_load(self):
        ranker = lr.LightGBMRanker(model_path=self.valid_model_path, metadata_path=self.metadata_path)
        ranker.load_model()
        info = ranker.get_model_info()
        self.assertEqual(info, self.metadata_content)

    def test_missing_metadata_file_does_not_prevent_model_load(self):
        ranker = lr.LightGBMRanker(
            model_path=self.valid_model_path,
            metadata_path=Path(self._tmpdir.name) / "does_not_exist.json",
        )
        ranker.load_model()   # must not raise
        self.assertTrue(ranker.is_loaded)
        self.assertIsNone(ranker.get_model_info())

    def test_missing_metadata_file_logs_a_warning(self):
        ranker = lr.LightGBMRanker(
            model_path=self.valid_model_path,
            metadata_path=Path(self._tmpdir.name) / "does_not_exist.json",
        )
        with self.assertLogs("pipeline.recommender.services.lightgbm_ranker", level="WARNING") as cm:
            ranker.load_model()
        self.assertTrue(any("no metadata found" in msg for msg in cm.output))

    def test_version_consistency_between_metadata_and_registry(self):
        """The metadata's feature_registry_version/feature_count must
        agree with the registry actually in use -- a stale/mismatched
        metadata file for a DIFFERENT registry version would be a real
        deployment hazard to catch, even though load_model() itself only
        validates feature NAMES/ORDER (not the metadata's version numbers)."""
        ranker = lr.LightGBMRanker(model_path=self.valid_model_path, metadata_path=self.metadata_path)
        ranker.load_model()
        info = ranker.get_model_info()
        registry = build_default_registry()
        self.assertEqual(info["feature_registry_version"], registry.to_dict()["version"])
        self.assertEqual(info["feature_count"], len(self.feature_names))


class TestSingletonLoadsOnce(LightGBMRankerTestBase):

    def test_singleton_load_model_called_at_most_once_on_success(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.valid_model_path):
            with patch("lightgbm.Booster", side_effect=lgb.Booster) as booster_spy:
                first = lr.get_lightgbm_ranker()
                second = lr.get_lightgbm_ranker()
                self.assertIs(first, second)
                self.assertTrue(first.is_loaded)
                self.assertEqual(booster_spy.call_count, 1, "must not re-read the model file on every call")

    def test_singleton_does_not_retry_after_a_failed_load(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH",
                   Path("/nonexistent/missing.txt")):
            with patch("lightgbm.Booster", side_effect=Exception("disk read")) as booster_spy:
                with self.assertRaises(lr.LightGBMRankerError):
                    lr.get_lightgbm_ranker()
                with self.assertRaises(lr.LightGBMRankerError):
                    lr.get_lightgbm_ranker()
                self.assertEqual(booster_spy.call_count, 1, "must not retry disk IO on every call")


class TestRankerModeConfig(unittest.TestCase):

    def test_default_mode_is_heuristic(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RANKER", None)
            self.assertEqual(lr.get_ranker_mode(), "heuristic")

    def test_explicit_modes_are_respected(self):
        for mode in ("heuristic", "lightgbm", "hybrid"):
            with patch.dict(os.environ, {"RANKER": mode}):
                self.assertEqual(lr.get_ranker_mode(), mode)

    def test_unknown_mode_falls_back_to_heuristic(self):
        with patch.dict(os.environ, {"RANKER": "not_a_real_mode"}):
            self.assertEqual(lr.get_ranker_mode(), "heuristic")

    def test_mode_is_case_insensitive(self):
        with patch.dict(os.environ, {"RANKER": "LightGBM"}):
            self.assertEqual(lr.get_ranker_mode(), "lightgbm")


class TestRankCandidates(LightGBMRankerTestBase):

    def setUp(self):
        super().setUp()
        self.graph = _make_graph()
        self.plan = DifficultyPlan(avg_mastery=0.4, level="mid")
        self.candidates = _make_candidates()
        self.filtering_rows = [
            {"problem_id": c.problem_id, "pool_sources": c.pool_sources, "pool_count": c.pool_count,
             "topic_tags": c.topic_tags, "difficulty_score": c.difficulty_score,
             "avg_mastery": 0.4, "max_urgency": 0.2, "predicted_success": c.predicted_success,
             "best_pool_score": c.best_score}
            for c in self.candidates
        ]

    def test_heuristic_mode_matches_heuristic_ranker_directly(self):
        with patch.dict(os.environ, {"RANKER": "heuristic"}):
            result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
        expected = HeuristicRanker().top_k(self.filtering_rows, k=10)
        self.assertEqual([r["problem_id"] for r in result], [r["problem_id"] for r in expected])

    def test_lightgbm_mode_returns_all_candidates_sorted_by_rank_score(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.valid_model_path):
            with patch.dict(os.environ, {"RANKER": "lightgbm"}):
                result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
        self.assertEqual({r["problem_id"] for r in result}, {c.problem_id for c in self.candidates})
        scores = [r["rank_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_hybrid_mode_returns_all_candidates_with_blended_score(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.valid_model_path):
            with patch.dict(os.environ, {"RANKER": "hybrid"}):
                result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
        self.assertEqual({r["problem_id"] for r in result}, {c.problem_id for c in self.candidates})
        for r in result:
            self.assertGreaterEqual(r["rank_score"], 0.0)
            self.assertLessEqual(r["rank_score"], 1.0)

    def test_respects_k(self):
        with patch.dict(os.environ, {"RANKER": "heuristic"}):
            result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=2)
        self.assertEqual(len(result), 2)


class TestFallbackBehaviour(LightGBMRankerTestBase):

    def setUp(self):
        super().setUp()
        self.graph = _make_graph()
        self.plan = DifficultyPlan(avg_mastery=0.4, level="mid")
        self.candidates = _make_candidates()
        self.filtering_rows = [
            {"problem_id": c.problem_id, "pool_sources": c.pool_sources, "pool_count": c.pool_count,
             "topic_tags": c.topic_tags, "difficulty_score": c.difficulty_score,
             "avg_mastery": 0.4, "max_urgency": 0.2, "predicted_success": c.predicted_success,
             "best_pool_score": c.best_score}
            for c in self.candidates
        ]

    def test_missing_model_falls_back_to_heuristic_and_logs(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH",
                   Path("/nonexistent/missing.txt")):
            with patch.dict(os.environ, {"RANKER": "lightgbm"}):
                with self.assertLogs("pipeline.recommender.services.lightgbm_ranker", level="WARNING") as cm:
                    result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
        expected = HeuristicRanker().top_k(self.filtering_rows, k=10)
        self.assertEqual([r["problem_id"] for r in result], [r["problem_id"] for r in expected])
        self.assertTrue(any("falling back to heuristic" in msg for msg in cm.output))

    def test_feature_mismatch_falls_back_to_heuristic(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.mismatched_model_path):
            with patch.dict(os.environ, {"RANKER": "lightgbm"}):
                result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
        expected = HeuristicRanker().top_k(self.filtering_rows, k=10)
        self.assertEqual([r["problem_id"] for r in result], [r["problem_id"] for r in expected])

    def test_inference_failure_falls_back_to_heuristic(self):
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH", self.valid_model_path):
            with patch.object(lr.LightGBMRanker, "score_candidates",
                              side_effect=lr.LightGBMRankerError("boom")):
                with patch.dict(os.environ, {"RANKER": "lightgbm"}):
                    result = lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
        expected = HeuristicRanker().top_k(self.filtering_rows, k=10)
        self.assertEqual([r["problem_id"] for r in result], [r["problem_id"] for r in expected])

    def test_recommender_never_raises_when_model_unavailable(self):
        """The whole point of the fallback: a missing/broken model must
        never surface as an exception to the caller."""
        with patch("pipeline.recommender.services.lightgbm_ranker.LIGHTGBM_MODEL_PATH",
                   Path("/nonexistent/missing.txt")):
            with patch.dict(os.environ, {"RANKER": "hybrid"}):
                try:
                    lr.rank_candidates(self.filtering_rows, self.candidates, self.graph, self.plan, k=10)
                except Exception as exc:   # noqa: BLE001 -- explicitly asserting NO exception of any kind
                    self.fail(f"rank_candidates raised {exc!r} instead of falling back")


if __name__ == "__main__":
    unittest.main(verbosity=2)
