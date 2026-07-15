"""
Adaptive difficulty controller.

Sits between pool generation and candidate filtering. It does NOT generate
questions. It reads the user's current state (BKT mastery, HLR urgency and
half-life, confidence, SM-2 review timing, concept-gap severity — all already
carried on the UserGraph built from Shraddha's stores and Postgres) and
produces, for every pool:

    - a weight  : how much this pool should contribute to the final slate
    - a mix     : this pool's own easy / medium / hard percentage split

The pool generation layer (built separately) uses the weight to decide how
many candidates to draw from each pool, and the mix to decide the difficulty
spread within that pool's draw.

No ELO is used — difficulty preference is derived entirely from the user's
average BKT mastery and how many concepts are weak / urgent / overdue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pipeline.recommender.models.user_graph import UserGraph, EdgeType


# Pools this controller assigns weights to. Matches the pool generation layer.
# Previously 7 letters (A/B_C/D/E/F/G/vector) -- collapsed to 4 pools that
# absorb the old ones as internal target sets (see pools.py's module
# docstring for the mapping): course_path=A+G, vector=B_C+vector,
# difficulty=D+F, urgency=E (unchanged).
POOLS = ["difficulty", "vector", "course_path", "urgency"]

# A user is treated as "beginner-leaning" below this average mastery,
# "advanced-leaning" above the upper bound. Between them is the mid band.
LOW_MASTERY  = 0.35
HIGH_MASTERY = 0.65

# When the user has many overdue / urgent concepts, review-type pools
# (E spaced-repetition, D weakness) get boosted relative to growth pools.
URGENCY_BOOST_THRESHOLD = 0.6
SEVERITY_WEAK_THRESHOLD = 0.6

# Difficulty mix presets by user level. Each is (easy, medium, hard) and sums
# to 1.0. The controller interpolates and adjusts these per pool.
# (No separate MIX_MID constant: _base_mix linearly interpolates between
# these two endpoints across the mid band, which already passes through
# the natural midpoint -- a separate unused constant there was dead code.)
MIX_BEGINNER = (0.60, 0.30, 0.10)
MIX_ADVANCED = (0.15, 0.40, 0.45)

# Distinct from MIX_BEGINNER: a user's very first-ever recommendations
# (is_cold_start -- zero concept_edges or zero solved problems, not merely
# "low mastery") get an EXTRA gentle mix, not just the general beginner
# mix. A user who has solved nothing yet has zero evidence they can even
# handle a medium problem -- MIX_BEGINNER's 30% medium / 10% hard is too
# much to lead with. Once they've solved even one problem, avg_mastery
# (no longer exactly 0) and _base_mix's normal interpolation take over.
MIX_COLD_START = (0.85, 0.15, 0.0)


@dataclass
class PoolDirective:
    """The controller's instruction for a single pool."""
    pool:   str
    weight: float                       # normalised, all pools sum to 1.0
    mix:    dict                         # {"easy": .., "medium": .., "hard": ..}


@dataclass
class DifficultyPlan:
    """Full output: one directive per pool, plus the signals it was built from."""
    directives:   dict = field(default_factory=dict)   # pool -> PoolDirective
    avg_mastery:  float = 0.0
    level:        str   = "mid"                         # beginner / mid / advanced
    n_weak:       int   = 0
    n_urgent:     int   = 0
    n_overdue:    int   = 0
    is_cold_start: bool = False

    def weight_of(self, pool: str) -> float:
        d = self.directives.get(pool)
        return d.weight if d else 0.0

    def mix_of(self, pool: str) -> dict:
        d = self.directives.get(pool)
        return d.mix if d else {"easy": 0.34, "medium": 0.33, "hard": 0.33}

    def to_dict(self) -> dict:
        return {
            "level":         self.level,
            "avg_mastery":   round(self.avg_mastery, 4),
            "n_weak":        self.n_weak,
            "n_urgent":      self.n_urgent,
            "n_overdue":     self.n_overdue,
            "is_cold_start": self.is_cold_start,
            "pools": {
                p: {"weight": round(d.weight, 4), "mix": d.mix}
                for p, d in self.directives.items()
            },
        }


class AdaptiveDifficultyController:
    """Turns a UserGraph into a per-pool difficulty plan."""

    def __init__(self, now: float | None = None):
        self._now = now if now is not None else time.time()

    # ------------------------------------------------------------------ public

    def build_plan(self, graph: UserGraph) -> DifficultyPlan:
        avg_mastery = self._avg_mastery(graph)
        n_weak      = len(graph.weak_concepts(SEVERITY_WEAK_THRESHOLD))
        n_urgent    = len(graph.urgent_concepts(URGENCY_BOOST_THRESHOLD))
        n_overdue   = self._count_overdue(graph)
        is_cold     = len(graph.concept_edges) == 0 or not graph.solved_ids

        level = self._level(avg_mastery)

        weights = self._pool_weights(
            level, n_weak, n_urgent, n_overdue, is_cold
        )
        # Cold start gets its own extra-gentle mix (see MIX_COLD_START),
        # not just the general beginner interpolation -- a user's very
        # first recommendations should require extremely low difficulty.
        base_mix = MIX_COLD_START if is_cold else self._base_mix(avg_mastery)

        directives = {}
        for pool in POOLS:
            mix = self._pool_mix(pool, base_mix, level)
            directives[pool] = PoolDirective(
                pool=pool,
                weight=weights[pool],
                mix=mix,
            )

        return DifficultyPlan(
            directives=directives,
            avg_mastery=avg_mastery,
            level=level,
            n_weak=n_weak,
            n_urgent=n_urgent,
            n_overdue=n_overdue,
            is_cold_start=is_cold,
        )

    # ------------------------------------------------------------- signals

    def _avg_mastery(self, graph: UserGraph) -> float:
        """
        Average CURRENT proficiency (BKT mastery decayed by HLR retention,
        see UserGraph.effective_proficiency), not raw historical mastery --
        so the easy/medium/hard mix eases back for a user returning after a
        break instead of assuming they're still at peak historical skill.
        This is the Duolingo-style "skills fade if you don't practice"
        property driving the difficulty curve.
        """
        slugs = list(graph.concept_edges.keys())
        if not slugs:
            return 0.0
        return sum(graph.effective_proficiency(s) for s in slugs) / len(slugs)

    def _count_overdue(self, graph: UserGraph) -> int:
        """Concepts whose SM-2 next_review_date is in the past."""
        overdue = 0
        for e in graph.concept_edges.values():
            if not e.next_review_date:
                continue
            try:
                due = datetime.fromisoformat(e.next_review_date)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if due.timestamp() <= self._now:
                overdue += 1
        return overdue

    def _level(self, avg_mastery: float) -> str:
        if avg_mastery < LOW_MASTERY:
            return "beginner"
        if avg_mastery > HIGH_MASTERY:
            return "advanced"
        return "mid"

    # ------------------------------------------------------------- weights

    def _pool_weights(self, level, n_weak, n_urgent, n_overdue, is_cold) -> dict:
        """
        Raw pool scores, then normalised to sum to 1.0.

        Weights are the old 7-letter raw scores summed into their new pool
        (course_path=A+G, vector=B_C+vector, difficulty=D+F, urgency=E) --
        preserves the original per-level tuning philosophy exactly, just
        collapsed onto 4 pools instead of 7.

        Cold start: lean on course_path (was A+G) since there is little
        behavioural signal to target weakness or review.
        Otherwise: base weights by level, then boost urgency when the user
        has overdue/urgent concepts, and boost difficulty (which now also
        covers stretch) when the user has many weak concepts.
        """
        if is_cold:
            raw = {
                "course_path": 0.35 + 0.20,   # A + G
                "vector":      0.15 + 0.05,   # B_C + vector
                "difficulty":  0.05 + 0.10,   # D + F
                "urgency":     0.05,          # E
            }
            return self._normalise(raw)

        # base weights per level (old A/B_C/D/E/F/G/vector raw scores, summed
        # into their new pool)
        if level == "beginner":
            raw = {
                "course_path": 0.25 + 0.10,   # A + G
                "vector":      0.20 + 0.10,   # B_C + vector
                "difficulty":  0.20 + 0.05,   # D + F
                "urgency":     0.10,          # E
            }
        elif level == "advanced":
            raw = {
                "course_path": 0.05 + 0.15,   # A + G
                "vector":      0.15 + 0.20,   # B_C + vector
                "difficulty":  0.10 + 0.25,   # D + F
                "urgency":     0.10,          # E
            }
        else:  # mid
            raw = {
                "course_path": 0.15 + 0.10,   # A + G
                "vector":      0.20 + 0.15,   # B_C + vector
                "difficulty":  0.15 + 0.15,   # D + F
                "urgency":     0.10,          # E
            }

        # review pressure: overdue reviews and urgent (forgetting) concepts
        # push urgency up
        if n_overdue > 0 or n_urgent > 0:
            pressure = min(0.20, 0.03 * (n_overdue + n_urgent))
            raw["urgency"] += pressure

        # weakness pressure: many weak concepts push difficulty up (more
        # total capacity for DifficultyPool, which allocates roughly half
        # its quota to weak targets internally)
        if n_weak > 0:
            raw["difficulty"] += min(0.15, 0.03 * n_weak)

        return self._normalise(raw)

    def _normalise(self, raw: dict) -> dict:
        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------- mixes

    def _base_mix(self, avg_mastery: float) -> tuple:
        """Interpolate a global easy/med/hard mix from average mastery."""
        if avg_mastery < LOW_MASTERY:
            return MIX_BEGINNER
        if avg_mastery > HIGH_MASTERY:
            return MIX_ADVANCED
        # linear interpolate between MID endpoints across the mid band
        span = HIGH_MASTERY - LOW_MASTERY
        t = (avg_mastery - LOW_MASTERY) / span if span > 0 else 0.5
        lo, hi = MIX_BEGINNER, MIX_ADVANCED
        return tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))

    def _pool_mix(self, pool: str, base_mix: tuple, level: str) -> dict:
        """
        Adjust the base mix per pool. Each pool has a natural difficulty lean:
          difficulty   -> follows base at the TOP level. It internally splits
                          into weak (was D, easier) and stretch (was F,
                          harder) target sets, each restricted to its own
                          allowed bands via _draw_with_mix's allowed_bands
                          override (see pools.py) -- so the easy-vs-hard
                          lean is already enforced per sub-target-set
                          regardless of what this top-level mix says, and no
                          top-level shift is needed (old D and F wanted
                          opposite shifts, which no longer makes sense to
                          apply to a single pool-level mix).
          vector       -> follows base (old B_C's small easy-shift only
                          applied to its graph-cooccurrence fallback path,
                          not the primary ANN path, so dropping it is a
                          negligible, cold-start-only behavior change).
          course_path  -> slightly easier (old G/novelty wanted a full easy
                          shift for "introduce new concepts gently"; old A/
                          course-path wanted no shift. Compromise: half the
                          old shift amount, since both unlock and explore
                          target sets now share one top-level mix with no
                          per-target-set band restriction like difficulty
                          has).
          urgency      -> follows base (review at the difficulty the user
                          learned the concept at, unchanged from old E).
        """
        easy, med, hard = base_mix

        if pool == "course_path":
            easy, med, hard = self._shift(easy, med, hard, toward="easy", amount=0.05)

        total = easy + med + hard
        return {
            "easy":   round(easy / total, 4),
            "medium": round(med  / total, 4),
            "hard":   round(hard / total, 4),
        }

    def _shift(self, easy, med, hard, toward, amount=0.10):
        """Move probability mass toward easy or hard, clamped at 0."""
        if toward == "easy":
            hard = max(0.0, hard - amount)
            easy = easy + amount
        elif toward == "hard":
            easy = max(0.0, easy - amount)
            hard = hard + amount
        return easy, med, hard


def build_difficulty_plan(graph: UserGraph, now: float | None = None) -> DifficultyPlan:
    """Convenience wrapper."""
    return AdaptiveDifficultyController(now=now).build_plan(graph)
