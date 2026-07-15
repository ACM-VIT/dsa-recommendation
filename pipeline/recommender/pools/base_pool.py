"""
Candidate pools.

Each pool produces a list of candidate problem_ids for one strategy. They all
share the same interface so the pool generation layer can call them uniformly
and hand their output to candidate filtering.

A pool NEVER makes the final recommendation. It only proposes candidates.
Filtering (remove solved / locked / out-of-range / missing prereqs) and
ranking happen downstream.

Pools:
  A       course path         - next problems in curriculum order
  B_C     near / far transfer  - same or analogous pattern to what user solved
  D       weakness recovery    - target weak / low-mastery concepts
  E       spaced review        - overdue SM-2 / high-urgency HLR concepts
  F       stretch              - slightly above current ability
  G       novelty              - unseen concepts reachable from mastered ones
  vector  semantic similarity  - ANN over the user state vector

Every pool takes:
  - graph  : UserGraph (BKT / HLR / SM-2 / severity per concept)
  - state  : UserStateVector (1920-d user embedding + concept lists)
  - qdrant : a client with .query_points / .scroll (may be None for graph-only pools)
  - n      : how many candidates to return (the controller's weight decides this)

and returns:
  list[Candidate]  where Candidate = (problem_id, pool_name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pipeline.recommender.models.user_graph import UserGraph
from pipeline.recommender.models.user_state import UserStateVector


# Shared difficulty bands on the 0-1 difficulty_score. Every pool draws from
# these three bands; what changes per pool is WHICH bands it's allowed to use
# and in what proportion -- that proportion now comes from the adaptive
# difficulty controller's per-pool mix (easy/medium/hard percentages), not a
# single fixed band per pool as before.
EASY_BAND = (0.0, 0.34)
MED_BAND  = (0.34, 0.66)
HARD_BAND = (0.66, 1.0)

_BAND_MAP = {"easy": EASY_BAND, "medium": MED_BAND, "hard": HARD_BAND}

# Foundational topics for a genuinely cold-start user: someone with ZERO
# concept_edges and ZERO cc_edges (cc_edges are only loaded for concepts the
# user has already touched, so a brand new user has none either). Without a
# fallback, any pool whose targets are derived purely from concept_edges/
# cc_edges has nothing to work with for exactly the case it needs to handle.
#
# Matches KNode's difficulty_tier=1 seeded topics ("1=Arrays/Strings" per
# the schema) -- the topics every learner starts with regardless of
# background. Adjust this list if the seeded topic taxonomy changes.
#
# Lives here (not in pools.py) because it's shared cold-start infra used by
# more than one pool's fallback branch -- see BasePool._starter_concept_fallback.
STARTER_CONCEPTS = ["arrays", "strings", "hash_map", "sorting"]


@dataclass
class Candidate:
    problem_id:      str
    pool:            str
    score:           float = 0.0    # pool-local relevance, filled where meaningful
    topic_tags:      list  = None   # populated from Qdrant payload when available
    difficulty_score: float = None  # populated from Qdrant payload when available

    def __post_init__(self):
        if self.topic_tags is None:
            self.topic_tags = []


class BasePool:
    """Shared helpers for all pools."""

    name = "base"

    # A pool's natural difficulty lean, used to restrict which of the
    # controller's easy/medium/hard mix percentages apply to it. E.g. pool D
    # (weakness recovery) never draws hard problems even if the controller's
    # global mix includes a hard percentage -- that percentage gets
    # renormalised across just ("easy", "medium") for this pool.
    # Subclasses override this; default is all three bands (no restriction).
    ALLOWED_BANDS = ("easy", "medium", "hard")

    def __init__(self, qdrant=None, collection: str = "problems_full"):
        self.qdrant = qdrant
        self.collection = collection

    def generate(self, graph: UserGraph, state: UserStateVector,
                 n: int = 20, mix: Optional[dict] = None) -> list[Candidate]:
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------

    def _exclude_ids(self, graph: UserGraph) -> set:
        """
        Problems we never want to propose: already solved, deprioritised
        (skipped 3+ times), or shown to the user again too recently.

        The recently-exposed check was a real gap -- UserGraph.exposed_ids/
        recently_exposed() already existed for this exact purpose (a
        problem recommended once or twice but not yet solved/skipped-3x
        had NO protection against being immediately re-recommended on the
        very next call), just never wired in here. Uses recently_exposed()'s
        own default 7-day window.
        """
        recently_shown = {
            pid for pid in graph.exposed_ids if graph.recently_exposed(pid)
        }
        return set(graph.solved_ids) | set(graph.deprioritised_ids) | recently_shown

    def _self_filter_locked(self, candidates: list, graph: Optional[UserGraph]) -> list:
        """
        Drop candidates locked by an unmastered prerequisite. Delegates to
        graph.is_locked() (single source of truth, shared with the global
        candidate filtering layer) rather than each pool re-deriving its own
        prereq index. Every pool applies this before returning -- "each pool
        filters its own questions" for missing prereqs, not just the global
        filtering layer downstream.
        """
        if graph is None:
            return candidates
        return [c for c in candidates if not graph.is_locked(c.topic_tags)]

    def _self_filter_difficulty_relevance(self, candidates: list,
                                          graph: Optional[UserGraph],
                                          max_delta: float = 0.5) -> list:
        """
        Coarse "not wildly irrelevant" safety net for ANN-based results
        (VectorPool), which select purely on vector similarity and have
        NO difficulty band restriction on the query itself --
        unlike the concept-based pools, which already only ever fetch within
        their controller-assigned band via _draw_with_mix. Drops candidates
        whose difficulty is far from the user's overall ability estimate
        (average mastery across all known concepts).

        max_delta is intentionally generous (0.5) -- this is a safety net
        against genuinely mismatched difficulty slipping through pure
        similarity search, not the primary difficulty targeting mechanism
        (that's ALLOWED_BANDS+mix for band-restricted pools, and the ZPD
        band filter downstream for every pool).

        Cold-start users (no concept_edges yet) have no ability estimate to
        compare against, so nothing is filtered here for them -- there is
        nothing "too hard/easy" to judge yet.
        """
        if graph is None:
            return candidates
        edges = list(graph.concept_edges.values())
        if not edges:
            return candidates
        ability = sum(e.mastery_score for e in edges) / len(edges)
        out = []
        for c in candidates:
            if c.difficulty_score is None:
                out.append(c)   # can't judge missing data, don't penalise it
                continue
            if abs(c.difficulty_score - ability) <= max_delta:
                out.append(c)
        return out

    def _apply_mix_to_candidates(self, candidates: list, n: int,
                                 mix: Optional[dict]) -> list[Candidate]:
        """
        Post-hoc difficulty-mix quota over an ALREADY-fetched candidate
        list -- used by _ann(), which retrieves purely by vector
        similarity in one round-trip and can't re-query Qdrant per
        difficulty band the way _draw_with_mix's concept-based path does.
        Buckets candidates into easy/medium/hard, then takes a quota from
        each bucket in the SAME proportions as the adaptive difficulty
        controller's mix, preserving each bucket's original similarity
        order.

        Without this, _ann() only had _self_filter_difficulty_relevance's
        loose +/-0.5 safety net -- VectorPool's results didn't respect the
        controller's easy/medium/hard proportions the way DifficultyPool/
        CoursePathPool's _draw_with_mix-based results do, so the adaptive
        difficulty curve was silently only 3/4-enforced across the pools.

        If mix is None, returns the first n candidates unchanged (matches
        _draw_with_mix's identical no-mix fallback). If a band's bucket
        can't fill its quota, the shortfall spills into whatever's left
        in the other buckets (in band order) rather than shrinking the
        result below n.
        """
        if mix is None:
            return candidates[:n]

        buckets = {"easy": [], "medium": [], "hard": []}
        for c in candidates:
            if c.difficulty_score is None:
                buckets["medium"].append(c)   # can't judge missing data -- treat as medium
            elif c.difficulty_score < EASY_BAND[1]:
                buckets["easy"].append(c)
            elif c.difficulty_score < MED_BAND[1]:
                buckets["medium"].append(c)
            else:
                buckets["hard"].append(c)

        bands = ("easy", "medium", "hard")
        sub = {b: max(0.0, mix.get(b, 0.0)) for b in bands}
        total = sum(sub.values())
        sub = ({b: 1.0 / len(bands) for b in bands} if total <= 0
               else {b: v / total for b, v in sub.items()})

        counts = {}
        remaining = n
        for i, b in enumerate(bands):
            if i == len(bands) - 1:
                counts[b] = remaining
            else:
                take = min(round(n * sub[b]), remaining)
                counts[b] = take
                remaining -= take

        selected = []
        for b in bands:
            take = min(counts[b], len(buckets[b]))
            selected.extend(buckets[b][:take])
            buckets[b] = buckets[b][take:]

        shortfall = n - len(selected)
        if shortfall > 0:
            leftovers = buckets["easy"] + buckets["medium"] + buckets["hard"]
            selected.extend(leftovers[:shortfall])

        return selected[:n]

    def _draw_with_mix(self, concept_slugs, n, exclude,
                       mix: Optional[dict] = None,
                       graph: Optional[UserGraph] = None,
                       allowed_bands: Optional[tuple] = None) -> list[Candidate]:
        """
        Split n candidates across easy/medium/hard according to `mix` (the
        adaptive difficulty controller's per-pool {"easy":.., "medium":..,
        "hard":..} percentages), restricted to `allowed_bands` (defaults to
        this pool's ALLOWED_BANDS) and renormalised across just those bands.

        `allowed_bands` lets a single pool draw different sub-target-sets
        under different band restrictions in the same generate() call --
        e.g. DifficultyPool draws its "weak" targets restricted to
        (easy, medium) and its "stretch" targets restricted to (medium,
        hard) in two separate calls, without needing two pool classes.

        If mix is None (caller didn't wire the controller's plan through),
        falls back to a single call across the full allowed range --
        preserves old behaviour for any caller not yet passing mix.

        Results are self-filtered for locked prereqs before returning
        (concept-based pools already only fetch within the correct
        difficulty band, so no separate difficulty-relevance pass is
        needed here -- that's specific to the ANN path in _ann()).
        """
        if not concept_slugs:
            return []

        bands = tuple(allowed_bands) if allowed_bands is not None else self.ALLOWED_BANDS

        if mix is None:
            lo = _BAND_MAP[bands[0]][0]
            hi = _BAND_MAP[bands[-1]][1]
            out = self._problems_by_concept(concept_slugs, n, exclude, difficulty=(lo, hi))
            return self._self_filter_locked(out, graph)

        # renormalise the controller's mix over just the allowed bands
        sub = {b: max(0.0, mix.get(b, 0.0)) for b in bands}
        total = sum(sub.values())
        if total <= 0:
            sub = {b: 1.0 / len(bands) for b in bands}
        else:
            sub = {b: v / total for b, v in sub.items()}

        # allocate integer counts per band; give the last band any rounding
        # remainder so the total always sums to exactly n
        counts = {}
        remaining = n
        bands = list(bands)
        for i, b in enumerate(bands):
            if i == len(bands) - 1:
                counts[b] = remaining
            else:
                c = round(n * sub[b])
                c = min(c, remaining)
                counts[b] = c
                remaining -= c

        out = []
        local_exclude = set(exclude)
        for b in bands:
            cnt = counts[b]
            if cnt <= 0:
                continue
            got = self._problems_by_concept(concept_slugs, cnt, local_exclude,
                                            difficulty=_BAND_MAP[b])
            out.extend(got)
            local_exclude |= {c.problem_id for c in got}
        out = self._self_filter_locked(out, graph)
        return out[:n]

    def _starter_concept_fallback(self, seen: set) -> list:
        """
        STARTER_CONCEPTS minus anything already in the user's graph -- for
        genuinely cold-start users (zero concept_edges, zero cc_edges) whose
        pool-specific target derivation has nothing to work with. Shared by
        CoursePathPool's unlock/explore modes so both fall back to the same
        starter set instead of duplicating this list-and-filter twice.
        """
        return [c for c in STARTER_CONCEPTS if c not in seen]

    def _problems_by_concept(self, concept_slugs, n, exclude,
                             difficulty=None) -> list[Candidate]:
        """
        Pull problems tagged with any of the given concepts from Qdrant,
        optionally filtered to a difficulty band. Uses scroll (metadata only,
        no vector needed). difficulty is (lo, hi) on difficulty_score or None.
        """
        if not self.qdrant or not concept_slugs:
            return []
        from qdrant_client.models import (
            Filter, FieldCondition, MatchAny, Range,
        )
        must = [FieldCondition(key="topic_tags", match=MatchAny(any=list(concept_slugs)))]
        if difficulty is not None:
            lo, hi = difficulty
            must.append(FieldCondition(
                key="difficulty_score", range=Range(gte=lo, lte=hi)))
        try:
            points, _ = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(must=must),
                limit=n * 3, with_payload=True, with_vectors=False,
            )
        except Exception:
            return []
        out = []
        for p in points:
            pl = p.payload or {}
            pid = str(pl.get("problem_id", p.id))
            if pid in exclude:
                continue
            out.append(Candidate(
                pid, self.name,
                topic_tags=pl.get("topic_tags") or [],
                difficulty_score=pl.get("difficulty_score"),
            ))
            if len(out) >= n:
                break
        return out

    def _ann(self, query_vec, n, exclude, graph: Optional[UserGraph] = None,
             mix: Optional[dict] = None) -> list[Candidate]:
        """
        ANN search over the user/query vector on the full collection.

        Unlike _problems_by_concept, this path selects purely on vector
        similarity with no difficulty band restriction on the query itself
        -- so results are self-filtered here for both locked prereqs AND
        difficulty relevance (_self_filter_difficulty_relevance), which the
        concept-based pools don't need since they already only fetch within
        their assigned band. When `mix` is supplied, the controller's
        easy/medium/hard proportions are ALSO applied post-hoc
        (_apply_mix_to_candidates) so this pool's results respect the same
        adaptive difficulty curve as the concept-based pools, not just the
        loose safety-net filter. Over-fetches (n*5 instead of n*3) so the
        mix quota has enough candidates per band to actually fill from.
        """
        if not self.qdrant or query_vec is None:
            return []
        try:
            hits = self.qdrant.query_points(
                collection_name=self.collection,
                query=query_vec,
                limit=max(n * 5, 30), with_payload=True, with_vectors=False,
            ).points
        except Exception:
            return []
        out = []
        for h in hits:
            pl = h.payload or {}
            pid = str(pl.get("problem_id", h.id))
            if pid in exclude:
                continue
            out.append(Candidate(
                pid, self.name, score=float(getattr(h, "score", 0.0)),
                topic_tags=pl.get("topic_tags") or [],
                difficulty_score=pl.get("difficulty_score"),
            ))
        out = self._self_filter_locked(out, graph)
        out = self._self_filter_difficulty_relevance(out, graph)
        out = self._apply_mix_to_candidates(out, n, mix)
        return out