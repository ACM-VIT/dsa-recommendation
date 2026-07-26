from typing import List, Dict

def pool_contribution(recommended_items: List[str], item_pools: Dict[str, str]) -> Dict[str, float]:
    """Calculates the percentage of top-K recommendations coming from each candidate retrieval pool."""
    if not recommended_items:
        return {}
        
    counts = {}
    total = len(recommended_items)
    for item in recommended_items:
        pool = item_pools.get(item, 'unknown')
        counts[pool] = counts.get(pool, 0) + 1
        
    return {pool: count / total for pool, count in counts.items()}