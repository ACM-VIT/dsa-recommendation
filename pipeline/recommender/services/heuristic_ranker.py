"""
Heuristic ranker.

A hand-tuned weighted formula standing in for LightGBM/LambdaMART until
there's enough real RecommendationLog data (was_attempted / was_skipped)
to train one. This is not a placeholder to be embarrassed about -- rule-based
scoring is how most recommendation systems actually start; it becomes the
labeled-data source a trained model eventually replaces.

This is the ONLY ranker in the online path. pipeline/recommender/ranking.py
used to exist alongside this one with a more elaborate (but never wired-in)
scoring formula -- it was dead code, never imported by recommend.py or any
controller. Its two genuinely useful ideas that this ranker didn't already
have have been ported in below (zpd_fit's Gaussian and variety_score); it
has since been deleted.

Scoring combines five signals, all already present on every candidate after
CandidateFilteringLayer.to_ranker_input() (plus recent_topics, passed in
separately -- see rank()):

  zpd_fit           how well this problem's difficulty fits the user's Zone
                    of Proximal Development: a Gaussian centered at
                    (avg_mastery + ZPD_TARGET_DELTA), not just proximity to
                    a single fixed optimal point -- so the ideal difficulty
                    shifts with the user's actual mastery on this
                    candidate's topics, not a global constant. PRIMARY
                    signal -- a candidate at exactly the productive-struggle
                    point for THIS user should usually win.
  pool_agreement    how many independent pools proposed this same problem.
                    3 pools agreeing is a much stronger signal than 1.
  urgency_boost     HLR forgetting-curve urgency -- a nearly-forgotten
                    concept's review problem gets pushed up the list.
  similarity_score  the originating pool's own local relevance score
                    (meaningful for ANN pools like vector, 0 by default
                    for concept-based pools that don't compute one).
  variety_score     penalizes topics the user has been seeing a lot
                    recently, so the slate doesn't repeat the same 1-2
                    topics over and over. Neutral (1.0, no penalty) when
                    recent_topics isn't provided.

Weights are named constants, intentionally simple to retune by hand as real
usage data starts suggesting better values -- no training required to adjust.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Zone of Proximal Development: the ideal difficulty for a user with
# average mastery M (on this candidate's topics) is M + ZPD_TARGET_DELTA --
# "somewhat harder than what you already know" is the productive-struggle
# sweet spot. ZPD_SIGMA controls how sharply the score falls off away from
# that ideal (Gaussian width -- smaller = stricter difficulty matching).
# Ported from the (formerly dead-code) ranking.py::zpd_fit_score, which is
# more principled than a flat proximity-to-fixed-point score since the
# ideal difficulty now tracks each candidate's own avg_mastery.
ZPD_TARGET_DELTA = 0.10
ZPD_SIGMA = 0.20

# Pool agreement saturates here -- 3+ pools agreeing gives the same full
# credit as more; beyond 3 there are only 4 pools total, so it stops being
# a meaningfully stronger signal past this point.
POOL_AGREEMENT_SATURATION = 3

# How many of the user's most-recently-touched topics count as "recent"
# for the variety penalty.
VARIETY_WINDOW = 10

# Hand-tuned weights, sum to 1.0. Adjust these directly as real usage data
# comes in -- no retraining needed, just edit the numbers. zpd_fit stays
# dominant; pool_agreement and urgency each gave up 0.05 to make room for
# variety_score, which was previously entirely absent from the live ranking
# path (DiversityMixer only hard-caps AFTER ranking, doesn't feed back into
# score order).
WEIGHT_ZPD_FIT    = 0.40
WEIGHT_POOL_AGREE = 0.20
WEIGHT_URGENCY    = 0.15
WEIGHT_SIMILARITY = 0.10
WEIGHT_VARIETY    = 0.15

assert abs((WEIGHT_ZPD_FIT + WEIGHT_POOL_AGREE + WEIGHT_URGENCY
           + WEIGHT_SIMILARITY + WEIGHT_VARIETY) - 1.0) < 1e-9


def zpd_fit_score(avg_mastery: float, difficulty_score) -> float:
    """
    Score how well a problem's difficulty fits this user's ZPD, for this
    candidate's own avg_mastery (not a single global optimal point).

        score = exp(-0.5 * ((difficulty - ideal) / ZPD_SIGMA)^2)
        ideal = min(1.0, avg_mastery + ZPD_TARGET_DELTA)

    Returns 0.5 (neutral) when difficulty_score is unknown -- don't
    penalise missing data.
    """
    if difficulty_score is None or avg_mastery is None:
        return 0.5
    ideal = min(1.0, avg_mastery + ZPD_TARGET_DELTA)
    return math.exp(-0.5 * ((difficulty_score - ideal) / ZPD_SIGMA) ** 2)


def calculate_variety_score(topic_tags, recent_topics, window: int = VARIETY_WINDOW) -> float:
    """
    1.0 (no penalty) when none of this candidate's topics appear in the
    user's most recent `window` touched topics; drops toward 0.0 as more
    of its topics have been seen recently. `recent_topics` empty/None ->
    neutral 1.0 (nothing to compare against, e.g. cold start or caller
    didn't provide it).
    """
    if not recent_topics or not topic_tags:
        return 1.0
    recent_window = recent_topics[-window:]
    recent_count = sum(1 for t in topic_tags if t in recent_window)
    return round(max(0.0, 1.0 - (recent_count / max(1, len(topic_tags)))), 4)


@dataclass
class RankedCandidate:
    """One ranker_input row plus its computed score and sub-scores (for debugging/explanation)."""
    row:               dict
    score:             float
    zpd_fit:           float
    pool_agreement:    float
    urgency_boost:     float
    similarity_score:  float
    variety_score:     float

    @property
    def problem_id(self) -> str:
        return self.row.get("problem_id")

    def to_dict(self) -> dict:
        return {
            **self.row,
            "rank_score": round(self.score, 4),
            "rank_components": {
                "zpd_fit":        round(self.zpd_fit, 4),
                "pool_agreement": round(self.pool_agreement, 4),
                "urgency":        round(self.urgency_boost, 4),
                "similarity":     round(self.similarity_score, 4),
                "variety":        round(self.variety_score, 4),
            },
        }


class HeuristicRanker:
    """
    Usage:
        ranker = HeuristicRanker()
        top10 = ranker.top_k(ranker_input_rows, k=10, recent_topics=recent)

    ranker_input_rows is exactly what CandidateFilteringLayer.to_ranker_input()
    returns -- a list of dicts with problem_id, pool_sources, pool_count,
    topic_tags, difficulty_score, avg_mastery, max_urgency,
    predicted_success, best_pool_score.

    recent_topics is an optional list of the user's most-recently-touched
    topic slugs (most recent last), used for the variety penalty. Omit it
    (or pass None/[]) to disable variety scoring -- every candidate then
    gets a neutral 1.0 for that component.
    """

    def __init__(self,
                 weight_zpd_fit: float = WEIGHT_ZPD_FIT,
                 weight_pool_agree: float = WEIGHT_POOL_AGREE,
                 weight_urgency: float = WEIGHT_URGENCY,
                 weight_similarity: float = WEIGHT_SIMILARITY,
                 weight_variety: float = WEIGHT_VARIETY):
        self.weight_zpd_fit = weight_zpd_fit
        self.weight_pool_agree = weight_pool_agree
        self.weight_urgency = weight_urgency
        self.weight_similarity = weight_similarity
        self.weight_variety = weight_variety

    def score_one(self, row: dict, recent_topics: list = None) -> RankedCandidate:
        zpd = zpd_fit_score(row.get("avg_mastery"), row.get("difficulty_score"))

        pool_count = row.get("pool_count") or 1
        pool_agreement = min(1.0, pool_count / POOL_AGREEMENT_SATURATION)

        urgency = row.get("max_urgency")
        urgency_boost = urgency if urgency is not None else 0.0

        similarity = row.get("best_pool_score")
        similarity_score = similarity if similarity is not None else 0.0
        similarity_score = max(0.0, min(1.0, similarity_score))   # clamp defensively

        variety_score = calculate_variety_score(row.get("topic_tags"), recent_topics)

        score = (
            self.weight_zpd_fit    * zpd +
            self.weight_pool_agree * pool_agreement +
            self.weight_urgency    * urgency_boost +
            self.weight_similarity * similarity_score +
            self.weight_variety    * variety_score
        )

        return RankedCandidate(
            row=row, score=score,
            zpd_fit=zpd, pool_agreement=pool_agreement,
            urgency_boost=urgency_boost, similarity_score=similarity_score,
            variety_score=variety_score,
        )

    def rank(self, rows: list, recent_topics: list = None) -> list:
        """Score every row, return all of them sorted best-first."""
        ranked = [self.score_one(r, recent_topics=recent_topics) for r in rows]
        ranked.sort(key=lambda rc: rc.score, reverse=True)
        return ranked

    def top_k(self, rows: list, k: int = 10, recent_topics: list = None) -> list:
        """Convenience: rank then take the top k as plain dicts (rank_score attached)."""
        ranked = self.rank(rows, recent_topics=recent_topics)
        return [rc.to_dict() for rc in ranked[:k]]


def rank_top_k(rows: list, k: int = 10, recent_topics: list = None) -> list:
    """One-shot convenience wrapper with default weights."""
    return HeuristicRanker().top_k(rows, k=k, recent_topics=recent_topics)
