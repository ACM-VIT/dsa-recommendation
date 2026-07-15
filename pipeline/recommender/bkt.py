import json
import os
from collections import defaultdict

from pipeline.recommender.telemetry import (
    MASTERY_THRESHOLD,
    compute_telemetry_signal_from_submission,
)

# Load problem->topic mapping. Absolute path so this works regardless of the
# working directory the process is launched from. Falls back to an empty
# mapping (with a warning) instead of crashing at import time if the file is
# missing, so an unrelated import chain doesn't take down the whole app.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_pt_edges_path = os.path.join(_BASE_DIR, "data", "problem_topic_edges_normalized.json")
try:
    with open(_pt_edges_path) as f:
        pt_edges = json.load(f)
except FileNotFoundError:
    print(f"[!] {_pt_edges_path} not found -- bkt.py starting with an EMPTY "
          f"problem->topic mapping. process_submission() will find zero "
          f"topics for every problem until this file exists.")
    pt_edges = []
problem_to_topics = defaultdict(list)
for edge in pt_edges:
    problem_to_topics[edge["source"]].append(edge["target"])

print(f"Loaded topic mappings for {len(problem_to_topics)} problems")

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


def update_bkt(current_p_l, observed):
    """
    Update knowledge probability using Bayes theorem.

    Args:
        current_p_l: current probability user knows this topic (0 to 1)
        observed: performance score from telemetry.compute_telemetry_signal (0 to 1)

    Returns:
        new_p_l: updated probability (0 to 1)
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
        new_p_l = p_l_given_obs + (1 - p_l_given_obs) * P_T
    else:
        new_p_l = p_l_given_obs

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

    updated_mastery = dict(user_mastery)
    mastered_topics = []
    results = []

    for topic in topics:
        # Get current P(L) or use default
        current_p_l = user_mastery.get(topic, DEFAULT_P_L["branch"])

        # Update BKT
        new_p_l = update_bkt(current_p_l, observed)
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