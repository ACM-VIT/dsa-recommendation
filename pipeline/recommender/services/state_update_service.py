"""
State update service.

This is the "STATE UPDATE LAYER": the thing that runs immediately after a
user solves/attempts a question, updating everything downstream needs to
reflect it before the next recommendation is served.

Flow:

    submission event (problem_id, verdict, hints, etc.)
        --> current mastery/HLR state read FROM the UserGraph itself
            (UserGraph is now the single canonical learner-state store --
            see module docstring below for why)
        --> BKT update (bkt.py)                -> new mastery_score per topic
        --> HLR update (hlr.py)                 -> new urgency/half_life per topic
        --> UserGraph mutation:
              - a new ProblemEdge is added for this problem_id
              - each affected ConceptEdge's mastery/urgency/half_life/
                confidence fields are updated from the BKT/HLR output
        --> Redis + Neo4j write-through (UserGraphService.persist)
        --> Postgres write-through (best-effort, synchronous) so the
            legacy read-only GET /mastery and GET /urgency endpoints keep
            working without themselves touching Redis/Neo4j/UserGraph
        --> UserStateVector rebuilt (UserStateBuilder.build)

Canonical store: UserGraph (Redis -> Neo4j) is the ONE source of truth for
recommendation. Previously there were three-plus divergent stores: a
stateless request-body round-trip, an in-memory dict local to this service,
and Postgres tables -- with no guarantee any two agreed. The in-memory dict
is gone; "current state" for a submission is now read directly off the
UserGraph (which this same service just wrote last time), and Postgres is
demoted to a synchronously-updated read projection kept only so the two
read-only legacy endpoints don't need to change.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.recommender.models.user_graph import (
    UserGraph, ProblemEdge, ConceptEdge, EdgeType,
)
from pipeline.recommender.models.user_state import UserStateBuilder, UserStateVector
from pipeline.recommender.services.user_graph_service import UserGraphService
from pipeline.recommender.bkt import process_submission as bkt_process_submission
from pipeline.recommender.hlr import process_hlr
from pipeline.recommender.telemetry import MASTERY_THRESHOLD

log = logging.getLogger(__name__)


@dataclass
class StateUpdateResult:
    """What changed as a result of processing one submission."""
    user_id:            str
    problem_id:          str
    updated_topics:      list            # topic slugs touched by this submission
    newly_mastered:      list            # topics that just crossed MASTERY_THRESHOLD
    graph:               UserGraph
    state:               Optional[UserStateVector]
    processed_at:        float
    # Full per-topic breakdown -- kept on the result so callers (e.g.
    # submission_controller.handle_update) can build a detailed response
    # without re-running BKT/HLR a second time.
    updated_mastery:      dict = field(default_factory=dict)
    updated_hlr:          dict = field(default_factory=dict)
    bkt_results:          list = field(default_factory=list)
    hlr_results:          list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "user_id":       self.user_id,
            "problem_id":    self.problem_id,
            "updated_topics": self.updated_topics,
            "newly_mastered": self.newly_mastered,
            "is_cold_start":  self.state.is_cold_start if self.state else None,
            "vector_dim":     int(self.state.vector.shape[0]) if self.state and self.state.vector is not None else None,
            "processed_at":   self.processed_at,
        }


class StateUpdateService:
    """
    Usage (called by the backend right after a submission is graded):

        svc = StateUpdateService(graph_service, qdrant)
        result = svc.process_submission(user_id, submission_dict)

    submission_dict matches what bkt.py/hlr.py already expect:
        {problemId, verdict, hintsUsed, submissionCount, normalisedScore,
         testCasesPassed, totalTestCases, timestamp}
    """

    def __init__(self, graph_service: UserGraphService, qdrant=None,
                 postgres_write_through: bool = True):
        self.graph_service = graph_service
        self.state_builder = UserStateBuilder(qdrant_client=qdrant)
        self.postgres_write_through = postgres_write_through

    def process_submission(self, user_id: str, submission: dict,
                           rebuild_vector: bool = True) -> StateUpdateResult:
        """
        Full state update pipeline for one submission. Returns the updated
        graph (and, unless rebuild_vector=False, the updated 1920-d vector)
        so the caller can serve a recommendation immediately without a
        second round trip.
        """
        now = submission.get("timestamp", time.time())

        # 0. Fetch (or cold-start create) the graph -- this IS the current
        # state; there's no separate in-memory store to consult anymore.
        graph = self._get_or_create_graph(user_id)
        current_mastery = self._mastery_from_graph(graph)
        current_hlr = self._hlr_from_graph(graph)

        # 1. BKT update
        updated_mastery, newly_mastered, bkt_results = bkt_process_submission(
            submission, current_mastery)

        # 2. HLR update
        updated_hlr, hlr_results = process_hlr(submission, current_hlr)

        # 3. Graph mutation: add the new ProblemEdge, update every affected
        # ConceptEdge from BKT+HLR output.
        self._apply_problem_edge(graph, submission, now)
        updated_topics = self._apply_concept_updates(
            graph, bkt_results, hlr_results, updated_mastery, updated_hlr)

        # 4. Write-through the mutated graph to BOTH tiers. invalidate()
        # first clears any stale Redis entry (defensive: if persist() then
        # only partially succeeds, we haven't left old data behind);
        # persist() writes the fresh graph to Redis AND Neo4j.
        self.graph_service.invalidate(user_id)
        self.graph_service.persist(user_id, graph)

        # 5. Best-effort Postgres write-through, purely so the legacy
        # GET /mastery / GET /urgency endpoints (which read Postgres
        # directly, not UserGraph) keep returning fresh data. UserGraph
        # remains authoritative regardless of whether this succeeds.
        if self.postgres_write_through:
            self._write_through_postgres(user_id, updated_mastery, updated_hlr)

        # 6. Recompute the 1920-d vector so the caller can serve a
        # recommendation immediately with the fresh state.
        state = None
        if rebuild_vector:
            state = self.state_builder.build(graph)

        return StateUpdateResult(
            user_id=user_id,
            problem_id=str(submission.get("problemId", "")),
            updated_topics=updated_topics,
            newly_mastered=newly_mastered,
            graph=graph,
            state=state,
            processed_at=now,
            updated_mastery=updated_mastery,
            updated_hlr=updated_hlr,
            bkt_results=bkt_results,
            hlr_results=hlr_results,
        )

    # ------------------------------------------------------------- helpers

    def _get_or_create_graph(self, user_id: str) -> UserGraph:
        """
        Fetch the existing graph, or create a fresh new-user graph if this
        is the very first submission for a user with no prior state.
        get() already handles this gracefully for a user with a User row
        but no telemetry yet; this only falls back to an explicit fresh
        graph if even the User row lookup fails (e.g. the submission
        arrives before the User row write has propagated).
        """
        try:
            return self.graph_service.get(user_id)
        except ValueError:
            return self.graph_service.new_user_graph(user_id)

    @staticmethod
    def _mastery_from_graph(graph: UserGraph) -> dict:
        """Current BKT mastery per topic, read straight off the canonical graph."""
        return {slug: edge.mastery_score for slug, edge in graph.concept_edges.items()}

    @staticmethod
    def _hlr_from_graph(graph: UserGraph) -> dict:
        """
        Current HLR state per topic, reconstructed from ConceptEdge fields.
        hlr.py::process_hlr only reads `half_life` and `last_review` off
        each entry (everything else it returns is freshly computed), so
        that's all this needs to reconstruct -- no separate HLR store.
        """
        out = {}
        for slug, edge in graph.concept_edges.items():
            if edge.last_attempted is None:
                continue
            out[slug] = {
                "half_life": edge.half_life,
                "last_review": datetime.fromtimestamp(
                    edge.last_attempted, tz=timezone.utc).isoformat(),
            }
        return out

    def _apply_problem_edge(self, graph: UserGraph, submission: dict, now: float) -> None:
        """Add a new ProblemEdge node for this submission."""
        problem_id = str(submission.get("problemId", ""))
        if not problem_id:
            return
        verdict = submission.get("verdict", "")
        edge_type = EdgeType.SOLVED if verdict == "OK" else EdgeType.ATTEMPTED
        normalised = submission.get("normalisedScore", 0.0)
        graph.add_problem_edge(ProblemEdge(
            problem_id=problem_id,
            edge_type=edge_type,
            normalised_score=float(normalised),
            timestamp=now,
        ))

    def _apply_concept_updates(self, graph: UserGraph, bkt_results: list,
                               hlr_results: list, updated_mastery: dict,
                               updated_hlr: dict) -> list:
        """
        Update every ConceptEdge touched by this submission with fresh
        mastery (BKT) and urgency/half_life (HLR) values. Returns the list
        of topic slugs that were touched.
        """
        touched = set()
        hlr_by_topic = {r["topic"]: r for r in hlr_results}

        for r in bkt_results:
            topic = r["topic"]
            touched.add(topic)
            mastery = r["new_p_l"]

            hlr_state = updated_hlr.get(topic, {})
            urgency = hlr_state.get("p_recall")
            urgency = (1.0 - urgency) if urgency is not None else 0.0
            half_life = hlr_state.get("half_life", 1.0)

            edge_type = (EdgeType.MASTERED if mastery >= MASTERY_THRESHOLD
                        else EdgeType.LEARNING)

            existing = graph.concept_edges.get(topic)
            confidence = existing.confidence if existing else 0.66

            # Authoritative overwrite -- the BKT/HLR output computed just now
            # IS the new truth for this topic. add_concept_edge's max-merge
            # would incorrectly refuse to let mastery decrease after a poor
            # submission; see update_concept_state's docstring for why.
            graph.update_concept_state(
                topic,
                edge_type=edge_type,
                mastery_score=mastery,
                confidence=confidence,
                urgency=urgency,
                half_life=half_life,
                last_attempted=time.time(),
            )

        return list(touched)

    def _write_through_postgres(self, user_id: str, updated_mastery: dict,
                                 updated_hlr: dict) -> None:
        """
        Best-effort sync write to the legacy Postgres tables. Never allowed
        to fail the submission -- UserGraph is already durably updated by
        this point, so a Postgres hiccup here only means the two read-only
        legacy endpoints see stale data until the next successful submission.
        """
        try:
            from database.postgres.db import update_user_mastery, save_user_hlr
            if updated_mastery:
                update_user_mastery(user_id, updated_mastery)
            if updated_hlr:
                save_user_hlr(user_id, updated_hlr)
        except Exception as exc:
            log.warning(
                "Postgres write-through failed for user %s (UserGraph is "
                "already up to date -- only legacy read endpoints affected): %s",
                user_id, exc,
            )


def process_submission_and_get_vector(
    user_id: str, submission: dict,
    graph_service: UserGraphService, qdrant=None,
) -> StateUpdateResult:
    """Convenience one-shot wrapper."""
    return StateUpdateService(graph_service, qdrant=qdrant).process_submission(
        user_id, submission)
