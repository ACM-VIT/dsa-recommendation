"""
tests/test_pools.py

Tests all four candidate pools without a real Qdrant. A fake client returns
tagged/difficulty-scored problems so each pool's selection logic can be
checked.

Run:
    python -m pytest tests/test_pools.py -v
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone, timedelta

from pipeline.recommender.models.user_graph import (
    UserGraph, UserNode, ConceptEdge, ConceptConceptEdge, EdgeType,
)
from pipeline.recommender.pools.base_pool import STARTER_CONCEPTS
from pipeline.recommender.pools.pools import (
    CoursePathPool, VectorPool, DifficultyPool, UrgencyPool,
    build_pools, POOL_CLASSES,
)

NOW = time.time()


def _iso(days):
    return (datetime.fromtimestamp(NOW, tz=timezone.utc) + timedelta(days=days)).isoformat()


# --------------------------------------------------------------------------
# Fake Qdrant
# --------------------------------------------------------------------------

class _Pt:
    def __init__(self, pid, tags, diff, score=0.9):
        self.id = pid
        self.payload = {"problem_id": pid, "topic_tags": tags, "difficulty_score": diff}
        self.score = score
        self.vector = None


class FakeQdrant:
    """
    Minimal stand-in. `problems` is a list of _Pt.
    scroll() filters by topic_tags MatchAny and difficulty Range.
    query_points() returns everything sorted by score (ANN stand-in).
    """
    def __init__(self, problems):
        self.problems = problems

    def scroll(self, collection_name, scroll_filter=None, limit=10,
               with_payload=True, with_vectors=False):
        want_tags = None
        diff_range = None
        if scroll_filter is not None:
            for cond in scroll_filter.must:
                key = getattr(cond, "key", None)
                if key == "topic_tags" and getattr(cond, "match", None) is not None:
                    want_tags = set(cond.match.any)
                if key == "difficulty_score" and getattr(cond, "range", None) is not None:
                    diff_range = cond.range
        out = []
        for p in self.problems:
            if want_tags is not None and not (set(p.payload["topic_tags"]) & want_tags):
                continue
            if diff_range is not None:
                d = p.payload["difficulty_score"]
                if diff_range.gte is not None and d < diff_range.gte:
                    continue
                if diff_range.lte is not None and d > diff_range.lte:
                    continue
            out.append(p)
            if len(out) >= limit:
                break
        return out, None

    def query_points(self, collection_name, query, limit=10,
                     with_payload=True, with_vectors=False):
        class R:
            points = sorted(self.problems, key=lambda p: p.score, reverse=True)[:limit]
        return R()


def _problems():
    return [
        _Pt("p_arrays_easy",  ["arrays"],   0.2),
        _Pt("p_arrays_med",   ["arrays"],   0.5),
        _Pt("p_arrays_hard",  ["arrays"],   0.8),
        _Pt("p_graphs_easy",  ["graphs"],   0.2),
        _Pt("p_graphs_hard",  ["graphs"],   0.85),
        _Pt("p_dp_med",       ["dp"],       0.5),
        _Pt("p_trees_easy",   ["trees"],    0.25),
        _Pt("p_solved",       ["arrays"],   0.3),
    ]


class _StubState:
    """Minimal UserStateVector stand-in for pool tests."""
    def __init__(self, qv, graph):
        self._qv = qv
        self.solved_ids = set(graph.solved_ids)
        self.is_cold_start = qv is None
    def to_query_vector(self):
        return self._qv


def _graph(concepts=None, solved=None, cc=None):
    g = UserGraph(user=UserNode(user_id="u1"))
    for c in (concepts or []):
        g.add_concept_edge(c)
    for pid in (solved or []):
        g.solved_ids.add(pid)
    for e in (cc or []):
        g.add_cc_edge(e)
    return g


def _concept(slug, mastery=0.5, urgency=0.0, severity=0.0,
             edge_type=EdgeType.LEARNING, next_review_date=None):
    return ConceptEdge(slug, edge_type, mastery_score=mastery,
                       urgency=urgency, severity=severity,
                       next_review_date=next_review_date)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class TestCoursePath(unittest.TestCase):
    def test_returns_in_progress_concepts(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=[])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("arrays" in c.problem_id for c in cands))

    def test_excludes_solved(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=["p_solved"])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertNotIn("p_solved", [c.problem_id for c in cands])

    def test_explore_mode_surfaces_unseen_concept_from_mastered(self):
        """Old NoveltyPool behavior, now the 'explore' target set of CoursePathPool."""
        g = _graph(
            [_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED)],
            solved=[],
            cc=[ConceptConceptEdge("arrays", "trees", EdgeType.PREREQ, 1.0)],
        )
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("trees" in c.problem_id for c in cands))

    def test_unlock_backlog_biases_quota_away_from_explore(self):
        """
        Duolingo-style: with several unlock targets still pending (>3), the
        quota split should heavily favor unlock over explore -- NOT a flat
        50/50 -- so the user works through the curriculum backlog before
        novelty exploration gets a meaningful share. Give both target sets
        abundant candidates so the split itself (not data scarcity) is what
        limits how many of each come back.
        """
        unlock_topics = ["arrays", "graphs", "trees", "dp", "strings"]
        problems = []
        for t in unlock_topics:
            for i in range(5):
                problems.append(_Pt(f"{t}_{i}", [t], 0.5))
        for i in range(25):
            problems.append(_Pt(f"greedy_{i}", ["greedy"], 0.5))

        concepts = [_concept(t, mastery=0.5) for t in unlock_topics]  # in-progress, unlock targets
        concepts.append(_concept("hash_map", mastery=0.8, edge_type=EdgeType.MASTERED))
        g = _graph(
            concepts, solved=[],
            # COOCCURS, not PREREQ -- PREREQ from a mastered concept would
            # also make "greedy" an unlock target, defeating the point of
            # this test (isolating pure explore-only targets).
            cc=[ConceptConceptEdge("hash_map", "greedy", EdgeType.COOCCURS, 1.0)],
        )
        pool = CoursePathPool(qdrant=FakeQdrant(problems))
        cands = pool.generate(g, _StubState(None, g), n=10)

        unlock_count = sum(1 for c in cands if set(c.topic_tags) & set(unlock_topics))
        explore_count = sum(1 for c in cands if "greedy" in c.topic_tags)
        self.assertGreater(unlock_count, explore_count)
        self.assertGreaterEqual(unlock_count, 7,
                                "unlock share should dominate a large backlog, not split ~50/50")

    def test_empty_when_no_unlock_and_no_explore_targets(self):
        """Mastered concept with no cc_edges at all: nothing in progress, nothing
        to unlock, nothing to explore -- and STARTER_CONCEPTS fallback only
        fires for a genuinely EMPTY graph (see TestColdStartFallback), not
        this case where the user has a concept but it's a dead end."""
        g = _graph([_concept("arrays", mastery=0.8, edge_type=EdgeType.MASTERED)], solved=[])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertEqual(cands, [])


class TestVectorPool(unittest.TestCase):
    def test_uses_ann_when_vector_present(self):
        g = _graph([_concept("arrays", mastery=0.6)], solved=[])
        pool = VectorPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState([1.0] * 1920, g), n=5)
        self.assertGreater(len(cands), 0)

    def test_cold_start_falls_back_to_cooccurrence_graph(self):
        """Old TransferPool's graph-only fallback, now VectorPool's own
        cold-start branch: no vector, but a COOCCURS edge exists."""
        g = _graph(
            [_concept("arrays", mastery=0.6)],
            solved=[],
            cc=[ConceptConceptEdge("arrays", "graphs", EdgeType.COOCCURS, 0.7)],
        )
        pool = VectorPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=5)
        self.assertTrue(any("graphs" in c.problem_id for c in cands))

    def test_cold_start_returns_empty_with_no_vector_and_no_cooccurrence(self):
        g = _graph([_concept("arrays", mastery=0.6)], solved=[])
        pool = VectorPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=5)
        self.assertEqual(cands, [])


class TestDifficultyPool(unittest.TestCase):
    def test_targets_weak_concepts(self):
        g = _graph([
            _concept("graphs", mastery=0.3, severity=0.8, edge_type=EdgeType.WEAK),
        ], solved=[])
        pool = DifficultyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("graphs" in c.problem_id for c in cands))

    def test_weak_concepts_draw_easier_difficulty_only(self):
        g = _graph([
            _concept("arrays", mastery=0.3, severity=0.7, edge_type=EdgeType.WEAK),
        ], solved=[])
        pool = DifficultyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        # hard arrays problem (0.8) should be excluded by the easy-med band
        self.assertNotIn("p_arrays_hard", [c.problem_id for c in cands])

    def test_targets_partial_mastery_stretch_concepts(self):
        g = _graph([_concept("dp", mastery=0.5)], solved=[])
        pool = DifficultyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("dp" in c.problem_id for c in cands))

    def test_stretch_concepts_exclude_easy_band(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=[])
        pool = DifficultyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertNotIn("p_arrays_easy", [c.problem_id for c in cands])

    def test_combines_weak_and_stretch_targets(self):
        g = _graph([
            _concept("graphs", mastery=0.3, severity=0.8, edge_type=EdgeType.WEAK),  # weak
            _concept("dp", mastery=0.5),                                              # stretch
        ], solved=[])
        pool = DifficultyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        tags = {t for c in cands for t in c.topic_tags}
        self.assertIn("graphs", tags)
        self.assertIn("dp", tags)

    def test_empty_when_no_weak_and_no_stretch(self):
        # A fully mastered concept isn't "weak", but IS a valid stretch
        # fallback target (old StretchPool behavior: "if nothing partial,
        # stretch on mastered concepts instead") -- so this only returns
        # empty when the user has NO concepts at all, not just none in the
        # partial-mastery band.
        g = _graph([], solved=[])
        pool = DifficultyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertEqual(cands, [])


class TestUrgencyPool(unittest.TestCase):
    def test_overdue_concept_included(self):
        g = _graph([
            _concept("arrays", mastery=0.6, next_review_date=_iso(-3)),
        ], solved=[])
        pool = UrgencyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("arrays" in c.problem_id for c in cands))

    def test_urgent_concept_included(self):
        g = _graph([
            _concept("graphs", mastery=0.6, urgency=0.8),
        ], solved=[])
        pool = UrgencyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertTrue(any("graphs" in c.problem_id for c in cands))

    def test_future_review_excluded(self):
        g = _graph([
            _concept("arrays", mastery=0.6, next_review_date=_iso(+5)),
        ], solved=[])
        pool = UrgencyPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=10)
        self.assertEqual(cands, [])


class TestRegistry(unittest.TestCase):
    def test_build_pools_has_all_four(self):
        pools = build_pools(qdrant=FakeQdrant(_problems()))
        self.assertEqual(set(pools.keys()),
                         {"difficulty", "vector", "course_path", "urgency"})

    def test_all_pools_exclude_solved(self):
        g = _graph([
            _concept("arrays", mastery=0.5),
            _concept("graphs", mastery=0.3, severity=0.8, edge_type=EdgeType.WEAK),
        ], solved=["p_solved"])
        pools = build_pools(qdrant=FakeQdrant(_problems()))
        state = _StubState([1.0] * 1920, g)
        for name, pool in pools.items():
            cands = pool.generate(g, state, n=10)
            self.assertNotIn("p_solved", [c.problem_id for c in cands],
                             msg=f"pool {name} leaked a solved problem")

    def test_candidates_carry_pool_name(self):
        g = _graph([_concept("arrays", mastery=0.5)], solved=[])
        pool = CoursePathPool(qdrant=FakeQdrant(_problems()))
        cands = pool.generate(g, _StubState(None, g), n=5)
        for c in cands:
            self.assertEqual(c.pool, "course_path")


class TestColdStartFallback(unittest.TestCase):
    """
    Regression test for a real bug: CoursePathPool's (and formerly
    NoveltyPool's) cold-start branch used to read graph.concept_edges /
    mastered concepts to build their fallback list -- which is exactly
    what's EMPTY for a genuinely brand-new user, so it silently returned
    zero candidates for every new signup. Fallback is STARTER_CONCEPTS
    (see base_pool.py), so the fixture problems here MUST be tagged with
    the actual plural STARTER_CONCEPTS slugs ("arrays"/"strings"), not an
    arbitrary singular guess -- a previous version of this fixture used
    singular tags ("array"/"string") that didn't match STARTER_CONCEPTS at
    all, silently making these tests exercise nothing.
    """

    def test_starter_concepts_match_actual_taxonomy_slugs(self):
        self.assertEqual(STARTER_CONCEPTS, ["arrays", "strings", "hash_map", "sorting"])

    def test_course_path_returns_candidates_for_totally_cold_user(self):
        problems = [
            _Pt("arrays_0",  ["arrays"],  0.1),
            _Pt("strings_0", ["strings"], 0.15),
            _Pt("math_0",    ["math"],    0.1),
        ]
        pool = CoursePathPool(qdrant=FakeQdrant(problems))
        graph = _graph()   # zero concepts, zero cc_edges, zero solved
        cands = pool.generate(graph, _StubState(None, graph), n=10)
        self.assertGreater(len(cands), 0,
                           "CoursePathPool returned nothing for a cold-start "
                           "user -- starter concept fallback is broken")
        tags = {t for c in cands for t in c.topic_tags}
        self.assertTrue(tags & {"arrays", "strings"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
