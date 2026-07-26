from typing import List, Dict, Set

def cold_start_performance(user_evals: List[Dict]) -> Dict[str, float]:
    """Metrics specifically isolated for cold-start users (<= 3 past interactions)."""
    cold_users = [u for u in user_evals if u.get('is_cold_start', False)]
    if not cold_users:
        return {"ndcg@5": 0.0, "mrr": 0.0, "count": 0.0}
    
    avg_ndcg = sum(u.get('ndcg@5', 0.0) for u in cold_users) / len(cold_users)
    avg_mrr = sum(u.get('mrr', 0.0) for u in cold_users) / len(cold_users)
    return {"ndcg@5": avg_ndcg, "mrr": avg_mrr, "count": float(len(cold_users))}

def topic_coverage(recommended_topics: Set[str], user_interest_topics: Set[str]) -> float:
    """Measures how well recommendations overlap with the user's topic interests."""
    if not user_interest_topics:
        return 0.0
    return len(recommended_topics.intersection(user_interest_topics)) / len(user_interest_topics)