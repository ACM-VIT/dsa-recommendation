import json
import os
import sys
from collections import defaultdict

from pipeline.recommender.telemetry import (
    MASTERY_THRESHOLD,
    compute_telemetry_signal_from_submission,
)

# Load problem->topic mapping. Absolute path so this works regardless of the
# working directory the process is launched from. Falls back to an empty
# mapping (with a warning) instead of crashing at import time if the file is
# missing, so an unrelated import chain doesn't take down the whole app.
#
# Diagnostic output goes to stderr, NOT stdout -- scripts like
# run_full_pipeline.py treat stdout as pure JSON output; a stray print()
# here would corrupt that (this was a real bug, found by actually running
# the pipeline end-to-end: "Loaded topic mappings..." was landing in the
# JSON response stream).
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_pt_edges_path = os.path.join(_BASE_DIR, "data", "problem_topic_edges_normalized.json")
try:
    with open(_pt_edges_path) as f:
        pt_edges = json.load(f)
except FileNotFoundError:
    print(f"[!] {_pt_edges_path} not found -- bkt.py starting with an EMPTY "
          f"problem->topic mapping. process_submission() will find zero "
          f"topics for every problem until this file exists.", file=sys.stderr)
    pt_edges = []
problem_to_topics = defaultdict(list)
for edge in pt_edges:
    problem_to_topics[edge["source"]].append(edge["target"])

print(f"Loaded topic mappings for {len(problem_to_topics)} problems", file=sys.stderr)

BKT_PARAMS = {
    "P_T": 0.2,   # probability of learning after one attempt
    "P_G": 0.1,   # probability of guessing correctly without knowing
    "P_S": 0.1,   # probability of slipping even if they know
}

# Default initial P(L) per topic type
# Root topics start higher since user likely has some base knowledge
# Branch topics start lower since they are more specific
DEFAULT_P_L = {
    "root": 0.2,    # arrays, strings, math
    "branch": 0.15, # sliding window, two pointers etc
    "unknown": 0.1  # topic we have no info about
}

# Observed score below this is treated as a failed attempt -- the BKT
# learning transition (P_T) is skipped for failures, since "learning from
# a failed attempt" should not move mastery upward. Matches the original
# intent of the `if observed >= 0.5` branch before it was dropped.
LEARNING_TRANSITION_THRESHOLD = 0.5

# Hard ceiling on how much a SINGLE submission can move mastery, regardless
# of how large the raw Bayesian update computes to. The uncapped formula can
# swing a cold-start topic (P_L=0.15) to ~0.6+ off one strong submission --
# that's a step-function jump, not the gradual, incremental leveling this
# system is meant to produce (Duolingo-style: many small confirmations, not
# one lucky submission maxing out a skill). Deterministic and explainable --
# no ML, just a clamp on the delta.
#
# This is the NEUTRAL (difficulty=0.5 or unknown) cap. When a problem's
# difficulty is known, the effective cap itself scales by difficulty (see
# _CAP_SCALE_MIN/_MAX below) -- without this, a flat cap silently erases
# the difficulty signal for any submission whose raw Bayesian delta already
# exceeds it (which cold-start submissions almost always do: found via
# testing that an easy and a hard cold-start solve both landed on the exact
# same +0.12, since both raw deltas were already >> 0.12 before the cap
# even looked at difficulty). Scaling the cap itself keeps that
# differentiation visible even when the cap binds.
MAX_MASTERY_DELTA = 0.12

# Same 0.7x-1.3x range telemetry.py's difficulty_credit uses, applied to
# the cap instead of (in addition to) raw_perf -- so a trivial problem
# tops out lower (0.7 * 0.12 = 0.084) and a hard one tops out higher
# (1.3 * 0.12 = 0.156), even from a cold start where the raw delta would
# otherwise saturate either cap identically.
_CAP_SCALE_MIN = 0.7
_CAP_SCALE_MAX = 1.3

# Difficulty-aware dampening: a solve on a problem well below the user's
# current mastery on this topic is weaker evidence of growth than one at or
# above it -- they likely already knew it. Only dampens (never boosts) and
# never zeroes out a solve entirely (a solve is always some evidence).
_TRIVIAL_GAP_THRESHOLD = -0.15
_TRIVIAL_DAMPEN_FLOOR = 0.4


def update_bkt(current_p_l, observed, difficulty=None):
    """
    Update knowledge probability using Bayes theorem.

    Args:
        current_p_l: current probability user knows this topic (0 to 1)
        observed: performance score from telemetry.compute_telemetry_signal (0 to 1)
        difficulty: optional 0-1 difficulty score of the solved problem. When
            provided, a solve well below the user's current mastery on this
            topic (see _TRIVIAL_GAP_THRESHOLD) has its positive delta
            dampened -- trivial practice teaches less than appropriately
            challenging practice. Omit (default) to skip this adjustment.

    Returns:
        new_p_l: updated probability (0 to 1), guaranteed to differ from
            current_p_l by at most MAX_MASTERY_DELTA (difficulty=None) or
            MAX_MASTERY_DELTA scaled by _CAP_SCALE_MIN.._CAP_SCALE_MAX
            (difficulty given) -- see MAX_MASTERY_DELTA's docstring.
    """
    P_T = BKT_PARAMS["P_T"]
    P_G = BKT_PARAMS["P_G"]
    P_S = BKT_PARAMS["P_S"]

    # P(L | correct observation)
    p_l_correct = (current_p_l * (1 - P_S)) / (
        (current_p_l * (1 - P_S)) + ((1 - current_p_l) * P_G)
    )

    # P(L | wrong observation)
    p_l_wrong = (current_p_l * P_S) / (
        (current_p_l * P_S) + ((1 - current_p_l) * (1 - P_G))
    )

    # Blend using observed score as weight
    p_l_given_obs = observed * p_l_correct + (1 - observed) * p_l_wrong

    # Account for learning from this attempt -- ONLY on a sufficiently
    # successful observation. Applying the learning transition (P_T)
    # unconditionally lets a fully failed attempt (observed=0.0) still
    # push mastery upward, since (1 - p_l_given_obs) * P_T > 0 regardless
    # of how bad the observation was. A failed attempt should not be
    # treated as evidence of learning.
    if observed >= LEARNING_TRANSITION_THRESHOLD:
        new_p_l_raw = p_l_given_obs + (1 - p_l_given_obs) * P_T
    else:
        new_p_l_raw = p_l_given_obs

    # Smoothing cap -- bound how far this ONE submission can move mastery.
    # The cap itself scales with difficulty (see MAX_MASTERY_DELTA's
    # docstring) so difficulty keeps differentiating outcomes even when the
    # raw delta is large enough to saturate a flat cap.
    if difficulty is not None:
        cap_scale = max(_CAP_SCALE_MIN, min(_CAP_SCALE_MAX,
                         _CAP_SCALE_MIN + (_CAP_SCALE_MAX - _CAP_SCALE_MIN) * max(0.0, min(1.0, difficulty))))
        effective_cap = MAX_MASTERY_DELTA * cap_scale
    else:
        effective_cap = MAX_MASTERY_DELTA
    delta = new_p_l_raw - current_p_l
    delta = max(-effective_cap, min(effective_cap, delta))

    # Difficulty dampening for trivial solves (positive deltas only --
    # never amplifies a decrease from a failed/weak attempt).
    if difficulty is not None and delta > 0:
        gap = difficulty - current_p_l
        if gap < _TRIVIAL_GAP_THRESHOLD:
            delta *= max(_TRIVIAL_DAMPEN_FLOOR, 1.0 + gap)

    new_p_l = current_p_l + delta
    return round(min(1.0, max(0.0, new_p_l)), 4)

def process_submission(submission, user_mastery):
    """
    Process a submission and update BKT mastery for all related topics.
    
    Args:
        submission: dict with userId, problemId, verdict, testCasesPassed,
                    totalTestCases, hintsUsed, submissionCount, normalisedScore
        user_mastery: dict of {topic_slug: p_l} for this user
    
    Returns:
        updated_mastery: dict of {topic_slug: new_p_l}
        mastered_topics: list of topics that crossed mastery threshold
        results: detailed results per topic
    """
    problem_id = submission["problemId"]
    topics = problem_to_topics.get(problem_id, [])

    if not topics:
        return user_mastery, [], []

    # Shared telemetry signal (also consumed by hlr.py::process_hlr for the
    # same submission) -- see telemetry.py for the confidence-penalty logic.
    observed = compute_telemetry_signal_from_submission(submission).value
    difficulty = submission.get("problemDifficulty")

    updated_mastery = dict(user_mastery)
    mastered_topics = []
    results = []

    for topic in topics:
        # Get current P(L) or use default
        current_p_l = user_mastery.get(topic, DEFAULT_P_L["branch"])

        # Update BKT
        new_p_l = update_bkt(current_p_l, observed, difficulty=difficulty)
        updated_mastery[topic] = new_p_l

        # Check if topic just got mastered
        was_mastered = current_p_l >= MASTERY_THRESHOLD
        now_mastered = new_p_l >= MASTERY_THRESHOLD
        if now_mastered and not was_mastered:
            mastered_topics.append(topic)

        results.append({
            "topic": topic,
            "previous_p_l": current_p_l,
            "new_p_l": new_p_l,
            "mastered": now_mastered,
            "observed_score": observed
        })

    return updated_mastery, mastered_topics, results