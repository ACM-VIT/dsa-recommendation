"""
Candidate filtering layer.

Sits between the 4 candidate pools and the ranker. Per the architecture:

    pool difficulty, vector, course_path, urgency
        --(each pool's raw candidates, already difficulty-banded)-->
    per-pool filter: removes solved, deprioritised, locked, missing prereqs
        -->
    CANDIDATE FILTERING:
        - each pool sends {problem_id: pool_name} into a local object
        - concatenated after removing garbage recommendations into a
          common object {[pid1, pid2, ... pidn]: pool_name}
        - GLOBAL FILTERING: removes duplicated questions, further
          processes the question pool
        -->
    filter to 55-80% predicted success (optimal ~68%, the ZPD band)
        -->
    ranker / scoring engine

This module owns everything up to and including the ZPD band filter. It does
NOT rank or score candidates for final ordering -- that is the ranker's job.
It DOES need a predicted-success estimate to apply the ZPD filter; a simple
BKT-mastery-based estimator is provided as a default, since BKT mastery
scores are already present on every ConceptEdge in the UserGraph (same
store this whole recommender reads from). If a real trained
success-probability model exists later, pass it in via
`success_estimator=` and this layer will use that instead.

Every mastery/urgency number used here comes directly from the same
UserGraph the BKT/HLR update path populates (via UserGraphService) -- no
separate data source.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from pipeline.recommender.models.user_graph import UserGraph, EdgeType
from pipeline.recommender.pools.base_pool import Candidate
from pipeline.recommender.telemetry import MASTERY_THRESHOLD

log = logging.getLogger(__name__)


# Locked / eligibility ------------------------------------------------------

# ZPD band from the architecture diagram: candidates with predicted success
# probability outside [ZPD_LO, ZPD_HI] are filtered out. ZPD_OPTIMAL is the
# sweet spot the diagram calls out (~68%) -- not a hard cutoff, just the
# center of the band, exposed for the ranker to use as a tie-breaker if it
# wants to prefer candidates closer to it.
ZPD_LO      = 0.55
ZPD_HI      = 0.80
ZPD_OPTIMAL = 0.68


# ---------------------------------------------------------------------------
# Prerequisite hard gate -- ported from the formerly dead-code ranking.py
# (rank_candidates was never imported/called anywhere in the live path; its
# ZPD-fit and variety scoring moved to heuristic_ranker.py, and this
# prerequisite check moves here). This is deliberately a HARD FILTER, not a
# ranking signal -- "what can appear at all" and "how what appears gets
# ordered" stay separate. It checks the backend's own `topic_prerequisite`
# Postgres table, which is a SEPARATE source of truth from the offline
# concept graph's cc_edges PREREQ edges that UserGraph.is_locked() (used in
# _filter_pool below) already checks. Both run: is_locked() keeps pools from
# generating obviously-locked candidates in the first place (efficiency,
# uses the graph pools already have in hand); this table check is the
# authoritative final gate against the backend's canonical prerequisite
# data, run once here regardless of which pool a candidate came from.
# ---------------------------------------------------------------------------

_PREREQ_TABLE_CACHE: Optional[dict] = None


def _load_prerequisite_table() -> dict:
    """{topic_id: [prerequisite_topic_id, ...]} from Postgres topic_prerequisite."""
    from database.postgres.db import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT topic_id, prerequisite_id FROM topic_prerequisite")
            rows = cur.fetchall()
        prereqs: dict = {}
        for topic_id, prereq_id in rows:
            prereqs.setdefault(topic_id, []).append(prereq_id)
        return prereqs
    finally:
        conn.close()


def _get_prerequisite_table() -> dict:
    """
    Only cache successful loads -- a transient DB failure (or no
    DATABASE_URL configured at all, e.g. in tests) returns {} uncached, so
    the gate degrades to a no-op rather than either crashing recommendation
    generation or permanently disabling itself for the life of the process.
    """
    global _PREREQ_TABLE_CACHE
    if _PREREQ_TABLE_CACHE is None:
        try:
            _PREREQ_TABLE_CACHE = _load_prerequisite_table()
        except Exception as exc:
            log.warning("Prerequisite table unavailable, gate disabled for this call: %s", exc)
            return {}
    return _PREREQ_TABLE_CACHE


@dataclass
class MergedCandidate:
    """One problem, deduplicated across every pool that proposed it."""
    problem_id:        str
    pool_sources:       list          # e.g. ["course_path", "vector"] -- every pool that proposed it
    best_score:         float         # max pool-local score across sources
    topic_tags:         list
    difficulty_score:   Optional[float]
    predicted_success:  Optional[float] = None   # filled by apply_zpd_filter

    @property
    def pool_count(self) -> int:
        return len(self.pool_sources)


@dataclass
class FilterReport:
    """What the filtering layer removed and why -- for debugging / logging."""
    input_count:          int = 0
    removed_solved:       int = 0
    removed_deprioritised: int = 0
    removed_locked:       int = 0
    removed_prereq_gate:  int = 0
    removed_duplicates:   int = 0     # candidates merged into an existing entry
    removed_zpd:          int = 0
    output_count:         int = 0

    def to_dict(self) -> dict:
        return {
            "input_count":           self.input_count,
            "removed_solved":        self.removed_solved,
            "removed_deprioritised": self.removed_deprioritised,
            "removed_locked":        self.removed_locked,
            "removed_prereq_gate":   self.removed_prereq_gate,
            "removed_duplicates":    self.removed_duplicates,
            "removed_zpd":           self.removed_zpd,
            "output_count":          self.output_count,
        }


class CandidateFilteringLayer:
    """
    Usage:
        layer = CandidateFilteringLayer(graph)
        merged, report = layer.run({
            "course_path": pool_course_path_candidates,
            "difficulty": pool_difficulty_candidates,
            ...
        })
        ranker_input = layer.to_ranker_input(merged)
    """

    def __init__(self, graph: UserGraph,
                 success_estimator: Optional[Callable[["MergedCandidate", UserGraph], float]] = None):
        self.graph = graph
        self._success_estimator = success_estimator or self._default_success_estimator
        self._prereq_index = self._build_prereq_index(graph)
        self._prereq_table = _get_prerequisite_table()

    # ------------------------------------------------------------------ public

    def run(self, pool_candidates: dict) -> tuple:
        """
        Full pipeline: per-pool filter -> merge/dedup -> ZPD band filter.
        Returns (list[MergedCandidate], FilterReport).
        """
        report = FilterReport()
        filtered_by_pool = {}
        for pool_name, candidates in pool_candidates.items():
            report.input_count += len(candidates)
            filtered_by_pool[pool_name] = self._filter_pool(candidates, report)

        merged, dup_count = self._merge(filtered_by_pool)
        report.removed_duplicates = dup_count

        merged = self._apply_zpd_filter(merged, report)
        report.output_count = len(merged)
        return merged, report

    # ------------------------------------------------------------- per-pool

    def _filter_pool(self, candidates: list, report: FilterReport) -> list:
        """Remove solved, deprioritised, recently-shown, and locked
        candidates from one pool's raw list -- defense in depth alongside
        each pool's own BasePool._exclude_ids() pre-filter."""
        out = []
        for c in candidates:
            if c.problem_id in self.graph.solved_ids:
                report.removed_solved += 1
                continue
            if c.problem_id in self.graph.deprioritised_ids:
                report.removed_deprioritised += 1
                continue
            if self.graph.recently_exposed(c.problem_id):
                report.removed_deprioritised += 1
                continue
            if self._is_locked(c):
                report.removed_locked += 1
                continue
            if self._fails_prerequisite_gate(c):
                report.removed_prereq_gate += 1
                continue
            out.append(c)
        return out

    def _fails_prerequisite_gate(self, c: Candidate) -> bool:
        """
        Authoritative hard gate against the backend's topic_prerequisite
        table (see module docstring for why this exists alongside
        _is_locked). No-ops gracefully if the table couldn't be loaded.
        """
        if not self._prereq_table or not c.topic_tags:
            return False
        for topic in c.topic_tags:
            for prereq in self._prereq_table.get(topic, []):
                edge = self.graph.concept_edges.get(prereq)
                mastery = edge.mastery_score if edge else 0.0
                if mastery < MASTERY_THRESHOLD:
                    return True
        return False

    def _is_locked(self, c: Candidate) -> bool:
        """
        A candidate is locked if ANY of its topic_tags requires a prerequisite
        concept the user hasn't mastered yet.
        """
        if not c.topic_tags:
            return False
        mastered = set(self.graph.mastered_concepts())
        for tag in c.topic_tags:
            required = self._prereq_index.get(tag, [])
            for prereq_slug in required:
                if prereq_slug not in mastered:
                    return True
        return False

    def _build_prereq_index(self, graph: UserGraph) -> dict:
        """
        graph.cc_edges is keyed by source concept, with PREREQ edges pointing
        to the concept it unlocks (source is prerequisite OF target). Build
        the reverse index here: target -> [required prerequisite slugs],
        which is what _is_locked needs to check a candidate's own tags.
        """
        reverse = {}
        for src, edges in graph.cc_edges.items():
            for e in edges:
                if e.edge_type == EdgeType.PREREQ:
                    reverse.setdefault(e.target_slug, []).append(src)
        return reverse

    # ------------------------------------------------------------- merge

    def _merge(self, filtered_by_pool: dict) -> tuple:
        """
        Concatenate every pool's filtered candidates into one deduplicated
        object. A problem recommended by multiple pools keeps all pool names
        in pool_sources and the best (max) score across them -- this is the
        "removing garbage recommendations into a common object" step from
        the diagram.
        """
        merged: dict = {}
        dup_count = 0
        for pool_name, candidates in filtered_by_pool.items():
            for c in candidates:
                if c.problem_id in merged:
                    dup_count += 1
                    existing = merged[c.problem_id]
                    if pool_name not in existing.pool_sources:
                        existing.pool_sources.append(pool_name)
                    existing.best_score = max(existing.best_score, c.score)
                    if not existing.topic_tags and c.topic_tags:
                        existing.topic_tags = c.topic_tags
                    if existing.difficulty_score is None:
                        existing.difficulty_score = c.difficulty_score
                else:
                    merged[c.problem_id] = MergedCandidate(
                        problem_id=c.problem_id,
                        pool_sources=[pool_name],
                        best_score=c.score,
                        topic_tags=list(c.topic_tags or []),
                        difficulty_score=c.difficulty_score,
                    )
        return list(merged.values()), dup_count

    # ------------------------------------------------------------- ZPD filter

    def _default_success_estimator(self, mc: MergedCandidate, graph: UserGraph) -> float:
        """
        Fallback predicted-success estimate when no trained model is supplied.
        Combines average CURRENT proficiency across the candidate's topic
        tags (BKT mastery decayed by HLR retention -- see
        UserGraph.effective_proficiency) with a difficulty-delta penalty --
        higher proficiency and lower difficulty both raise predicted
        success. Using proficiency rather than raw historical mastery here
        means a candidate on a topic the user has forgotten predicts LOWER
        success, matching their real current ability rather than their
        peak-ever performance.

        Returns a value in (0, 1). Cold-start candidates (no mastery data on
        any of their tags) default to 0.68 -- the diagram's own optimal ZPD
        point -- so they aren't unfairly filtered out before enough signal
        exists to judge them.
        """
        if not mc.topic_tags:
            return ZPD_OPTIMAL

        masteries = [
            graph.effective_proficiency(t)
            for t in mc.topic_tags
            if t in graph.concept_edges
        ]
        if not masteries:
            return ZPD_OPTIMAL

        avg_mastery = sum(masteries) / len(masteries)
        difficulty = mc.difficulty_score if mc.difficulty_score is not None else 0.5

        # Sigmoid centered so that mastery == difficulty gives ~0.5, and
        # higher mastery relative to difficulty pushes success upward.
        delta = (avg_mastery - difficulty) * 6.0
        success = 1.0 / (1.0 + math.exp(-delta))
        return round(success, 4)

    def _apply_zpd_filter(self, merged: list, report: FilterReport) -> list:
        """
        Keep only candidates whose predicted success probability lands in
        [ZPD_LO, ZPD_HI] -- the productive-struggle zone from the diagram.
        Below it: too hard, likely frustration. Above it: too easy, no growth.
        """
        out = []
        for mc in merged:
            mc.predicted_success = self._success_estimator(mc, self.graph)
            if ZPD_LO <= mc.predicted_success <= ZPD_HI:
                out.append(mc)
            else:
                report.removed_zpd += 1
        return out

    # ------------------------------------------------------------- ranker bridge

    def to_ranker_input(self, merged: list) -> list:
        """
        Flatten MergedCandidate objects into plain dicts shaped for a
        downstream ranker (Shraddha's LightGBM feature vector). Every field
        here is traceable back to the same UserGraph her BKT/HLR stores feed --
        this is the actual integration point with her recommendation engine,
        not a separate data path.
        """
        rows = []
        for mc in merged:
            # effective_proficiency (mastery decayed by HLR retention), not
            # raw mastery_score -- so the ranker's zpd_fit signal targets
            # the user's CURRENT real ability, not their historical peak.
            masteries = [
                self.graph.effective_proficiency(t)
                for t in mc.topic_tags if t in self.graph.concept_edges
            ]
            urgencies = [
                self.graph.concept_edges[t].urgency
                for t in mc.topic_tags if t in self.graph.concept_edges
            ]
            rows.append({
                "problem_id":        mc.problem_id,
                "pool_sources":      mc.pool_sources,
                "pool_count":        mc.pool_count,
                "topic_tags":        mc.topic_tags,
                "difficulty_score":  mc.difficulty_score,
                "avg_mastery":       round(sum(masteries) / len(masteries), 4) if masteries else None,
                "max_urgency":       round(max(urgencies), 4) if urgencies else None,
                "predicted_success": mc.predicted_success,
                "best_pool_score":   mc.best_score,
            })
        return rows
