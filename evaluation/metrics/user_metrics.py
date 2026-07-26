"""
evaluation/metrics/user_metrics.py

Per-segment/per-user metrics that look at a slice of the evaluated
population (cold-start users) or a single user's topic overlap, rather
than the whole batch. Not currently called from evaluator.py/experiments.py
-- available for a caller that wants to isolate these segments itself.
"""
from typing import List, Dict, Set

def cold_start_performance(user_evals: List[Dict]) -> Dict[str, float]:
    """Averages ndcg@5/mrr specifically over cold-start users.

    Args:
        user_evals: list of per-user dicts, each expected to carry
            "is_cold_start" (bool), and the "ndcg@5"/"mrr" values already
            computed for that user (e.g. by OfflineEvaluator.evaluate_user).

    Returns:
        {"ndcg@5": avg, "mrr": avg, "count": number of cold-start users
        found}. If no user in user_evals is flagged is_cold_start, returns
        zeroed-out averages with count=0.0 rather than raising.
    """
    cold_users = [u for u in user_evals if u.get('is_cold_start', False)]
    if not cold_users:
        return {"ndcg@5": 0.0, "mrr": 0.0, "count": 0.0}

    avg_ndcg = sum(u.get('ndcg@5', 0.0) for u in cold_users) / len(cold_users)
    avg_mrr = sum(u.get('mrr', 0.0) for u in cold_users) / len(cold_users)
    return {"ndcg@5": avg_ndcg, "mrr": avg_mrr, "count": float(len(cold_users))}

def topic_coverage(recommended_topics: Set[str], user_interest_topics: Set[str]) -> float:
    """Measures what fraction of a user's topic interests were touched by
    their recommendations.

    Args:
        recommended_topics: set of topics represented in the user's
            recommended items.
        user_interest_topics: set of topics the user is interested in
            (ground truth).

    Returns:
        |recommended_topics ∩ user_interest_topics| / |user_interest_topics|,
        or 0.0 if the user has no recorded topic interests.
    """
    if not user_interest_topics:
        return 0.0
    return len(recommended_topics.intersection(user_interest_topics)) / len(user_interest_topics)