"""
evaluation/metrics/pool_metrics.py

Candidate-pool attribution: what fraction of a user's shown
recommendations came from each retrieval pool (e.g. vector/heuristic/
popular), so a ranker's output can be inspected for which pools it's
actually drawing from.
"""
from typing import List, Dict

def pool_contribution(recommended_items: List[str], item_pools: Dict[str, str]) -> Dict[str, float]:
    """Calculates the percentage of recommended items coming from each
    candidate retrieval pool.

    Args:
        recommended_items: ranked (or unranked) list of recommended item IDs.
        item_pools: {item_id: pool_name}. An item not present in this dict
            is attributed to the "unknown" pool rather than dropped.

    Returns:
        {pool_name: fraction of recommended_items from that pool}. Values
        sum to 1.0 across all pools. Empty dict if recommended_items is
        empty.
    """
    if not recommended_items:
        return {}
        
    counts = {}
    total = len(recommended_items)
    for item in recommended_items:
        pool = item_pools.get(item, 'unknown')
        counts[pool] = counts.get(pool, 0) + 1
        
    return {pool: count / total for pool, count in counts.items()}