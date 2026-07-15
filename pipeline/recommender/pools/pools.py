"""
The four candidate pools. Each subclasses BasePool and implements generate().

generate(graph, state, n, mix) takes TWO quotas from the pipeline:
  - n    : how many candidates this pool should return (from the adaptive
           difficulty controller's per-pool WEIGHT, converted to a count by
           the pool generation orchestrator)
  - mix  : this pool's easy/medium/hard percentage split (from the adaptive
           difficulty controller's per-pool MIX), so the pool draws the
           right proportion of each difficulty band instead of one fixed band

Previously there were 7 pools (A/B_C/D/E/F/G/vector). On inspection, those 7
boiled down to 3 retrieval axes (structural graph traversal, semantic
similarity, mastery-band filtering) plus one distinct recency axis, not 7
independent concerns:

  DifficultyPool  - absorbs the old WeaknessPool (D) and StretchPool (F).
                    Mastery-band filtering was never a separate retrieval
                    axis -- both were the same "problems by concept, at a
                    difficulty band" query with different mastery-score
                    cutoffs. Handled here as two internal target sets (weak
                    / stretch), each restricted to its own allowed bands via
                    _draw_with_mix's allowed_bands override, combined into
                    one candidate list.
  VectorPool      - absorbs the old TransferPool (B_C) and VectorPool
                    (vector). Both were ANN over the user vector; the only
                    difference was TransferPool's graph-cooccurrence
                    fallback when no vector exists, which is now VectorPool's
                    own cold-start branch.
  CoursePathPool  - absorbs the old CoursePathPool (A) and NoveltyPool (G).
                    Both walk cc_edges from the user's current concepts;
                    they differ only in which edge type/target-selection
                    rule they use (PREREQ-unlocked vs. COOCCURS-reachable-
                    unseen). Handled as two internal target sets (unlock /
                    explore), combined into one candidate list. Shares
                    STARTER_CONCEPTS cold-start fallback via
                    BasePool._starter_concept_fallback (previously
                    duplicated across the two separate pool classes).
  UrgencyPool     - renamed SpacedReviewPool (E), logic unchanged. Kept as
                    its own top-level pool because recency/forgetting is a
                    genuinely distinct axis from mastery-band or structural
                    traversal -- collapsing it into another pool would blur
                    a signal worth keeping explicit.
"""

from __future__ import annotations

from pipeline.recommender.models.user_graph import UserGraph, EdgeType
from pipeline.recommender.models.user_state import UserStateVector
from pipeline.recommender.pools.base_pool import (
    BasePool, Candidate, EASY_BAND, MED_BAND, HARD_BAND, STARTER_CONCEPTS,
)

# CoursePathPool's unlock-vs-explore quota split (Duolingo-style curriculum
# priority -- see CoursePathPool.generate()).
_UNLOCK_BACKLOG_THRESHOLD = 3      # "several" unlock targets still pending
_UNLOCK_SHARE_WITH_BACKLOG = 0.85  # mostly curriculum-unlock while backlog is large
_UNLOCK_SHARE_SMALL_BACKLOG = 0.60 # still curriculum-leaning once backlog is small


class DifficultyPool(BasePool):
    """
    Difficulty pool.
    Draws candidates by concept mastery-band: weak concepts (mastery < 0.4,
    restricted to easy/medium so recovery stays gentle) and partial-mastery
    "stretch" concepts (0.4 <= mastery < 0.75, restricted to medium/hard so
    growth stays growth). Both sets are queried in the same generate() call
    and combined -- if only one set has targets, it gets the full quota.
    """
    name = "difficulty"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    _WEAK_BANDS = ("easy", "medium")
    _STRETCH_BANDS = ("medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)

        weak = set(graph.weak_concepts())
        low_mastery = [s for s, e in graph.concept_edges.items() if e.mastery_score < 0.4]
        weak_targets = list(weak | set(low_mastery))

        stretch_targets = [s for s, e in graph.concept_edges.items()
                           if 0.4 <= e.mastery_score < 0.75]
        if not stretch_targets:
            # if nothing partial, stretch on mastered concepts instead
            stretch_targets = list(graph.mastered_concepts())

        if not weak_targets and not stretch_targets:
            return []

        if weak_targets and stretch_targets:
            weak_n = (n + 1) // 2
            stretch_n = n - weak_n
        elif weak_targets:
            weak_n, stretch_n = n, 0
        else:
            weak_n, stretch_n = 0, n

        out = []
        local_exclude = set(exclude)
        if weak_n > 0 and weak_targets:
            got = self._draw_with_mix(weak_targets, weak_n, local_exclude, mix,
                                      graph=graph, allowed_bands=self._WEAK_BANDS)
            out.extend(got)
            local_exclude |= {c.problem_id for c in got}
        if stretch_n > 0 and stretch_targets:
            got = self._draw_with_mix(stretch_targets, stretch_n, local_exclude, mix,
                                      graph=graph, allowed_bands=self._STRETCH_BANDS)
            out.extend(got)

        return out[:n]


class VectorPool(BasePool):
    """
    Vector pool - semantic similarity.
    Primary: ANN over the user state vector on the 1920-d full collection
    (Question + Solution + RGCN shared embedding space) -- surfaces
    structurally similar problems regardless of topic label, and covers
    both "near transfer" (same pattern) and "far transfer" (analogous
    pattern) since both live in the same similarity search.
    Fallback (no vector yet, e.g. cold start): concepts reachable via
    co-occurrence edges from what the user already knows, drawn by concept
    + difficulty mix instead of ANN.
    """
    name = "vector"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        qv = state.to_query_vector() if state is not None else None
        if qv is not None:
            return self._ann(qv, n, exclude, graph=graph, mix=mix)

        cooccur = []
        for edges in graph.cc_edges.values():
            for e in edges:
                if e.edge_type == EdgeType.COOCCURS:
                    cooccur.append(e.target_slug)
        return self._draw_with_mix(cooccur, n, exclude, mix, graph=graph)


class CoursePathPool(BasePool):
    """
    Course path pool - structural graph traversal.
    Unlock targets: concepts currently in progress (not yet mastered) plus
    concepts unlocked by mastered prerequisites -- the curriculum's next
    step. Explore targets: concepts the user hasn't touched yet but are
    reachable from what they've mastered via any concept-concept edge --
    gentle novelty. Both target sets are queried in the same generate()
    call and combined.
    """
    name = "course_path"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        mastered = set(graph.mastered_concepts())
        seen = set(graph.concept_edges.keys())

        # unlock targets: in-progress concepts + prereq-unlocked concepts
        in_progress = [s for s in graph.concept_edges if s not in mastered]
        unlocked = []
        for src, edges in graph.cc_edges.items():
            for e in edges:
                if e.edge_type == EdgeType.PREREQ and src in mastered:
                    unlocked.append(e.target_slug)
        unlock_targets = list(dict.fromkeys(in_progress + unlocked))

        # explore targets: unseen concepts reachable from mastered ones
        novel = []
        for src, edges in graph.cc_edges.items():
            if src not in mastered:
                continue
            for e in edges:
                if e.target_slug not in seen:
                    novel.append(e.target_slug)
        explore_targets = list(dict.fromkeys(novel))

        if not unlock_targets and not explore_targets:
            # Genuinely cold start: nothing in progress, nothing unlocked,
            # nothing mastered to explore from. Fall back to starter topics
            # via _draw_with_mix so the actual data's difficulty bands
            # decide what comes back, rather than a single fixed-band query.
            fallback = self._starter_concept_fallback(seen)
            if not fallback:
                return []
            return self._draw_with_mix(fallback, n, exclude, mix, graph=graph)

        if unlock_targets and explore_targets:
            # Duolingo-style: exhaust the prerequisite/curriculum-unlock
            # backlog before leaning into novelty exploration. A user with
            # several concepts still in-progress or newly unlockable should
            # mostly see curriculum-unlock candidates; explore only gets a
            # meaningful share once that backlog is small -- "recommend
            # prerequisite topics before advanced ones," not a fixed 50/50
            # split regardless of how much curriculum is left.
            unlock_share = _UNLOCK_SHARE_WITH_BACKLOG if len(unlock_targets) > _UNLOCK_BACKLOG_THRESHOLD \
                else _UNLOCK_SHARE_SMALL_BACKLOG
            unlock_n = round(n * unlock_share)
            explore_n = n - unlock_n
        elif unlock_targets:
            unlock_n, explore_n = n, 0
        else:
            unlock_n, explore_n = 0, n

        out = []
        local_exclude = set(exclude)
        if unlock_n > 0 and unlock_targets:
            got = self._draw_with_mix(unlock_targets, unlock_n, local_exclude, mix, graph=graph)
            out.extend(got)
            local_exclude |= {c.problem_id for c in got}
        if explore_n > 0 and explore_targets:
            got = self._draw_with_mix(explore_targets, explore_n, local_exclude, mix, graph=graph)
            out.extend(got)

        return out[:n]


class UrgencyPool(BasePool):
    """
    Urgency pool - HLR / spaced review (renamed from SpacedReviewPool,
    logic unchanged).
    Concepts that are overdue (SM-2 next_review_date in the past) or high
    HLR urgency (user is forgetting). Draws at the difficulty the user
    learned them, following the controller's full mix.
    """
    name = "urgency"
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def generate(self, graph, state, n=20, mix=None):
        exclude = self._exclude_ids(graph)
        urgent = set(graph.urgent_concepts())

        import time
        from datetime import datetime, timezone
        now = time.time()
        overdue = set()
        for s, e in graph.concept_edges.items():
            if not e.next_review_date:
                continue
            try:
                due = datetime.fromisoformat(e.next_review_date)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                if due.timestamp() <= now:
                    overdue.add(s)
            except (ValueError, TypeError):
                continue

        target = list(urgent | overdue)
        if not target:
            return []
        return self._draw_with_mix(target, n, exclude, mix, graph=graph)


# registry so the pool generation layer can build them by name
POOL_CLASSES = {
    "difficulty":  DifficultyPool,
    "vector":      VectorPool,
    "course_path": CoursePathPool,
    "urgency":     UrgencyPool,
}


def build_pools(qdrant=None, collection="problems_full") -> dict:
    """Instantiate every pool, keyed by name."""
    return {name: cls(qdrant=qdrant, collection=collection)
            for name, cls in POOL_CLASSES.items()}
