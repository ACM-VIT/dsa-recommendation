import numpy as np
from typing import List, Set

def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates Precision@K: Ratio of recommended items in top-k that are relevant."""
    recommended_at_k = recommended[:k]
    if not recommended_at_k:
        return 0.0
    hits = sum(1 for item in recommended_at_k if item in relevant)
    return hits / k

def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates Recall@K: Proportion of total relevant items retrieved in top-k."""
    if not relevant:
        return 0.0
    recommended_at_k = recommended[:k]
    hits = sum(1 for item in recommended_at_k if item in relevant)
    return hits / len(relevant)

def mrr(recommended: List[str], relevant: Set[str]) -> float:
    """Calculates Mean Reciprocal Rank (MRR): Inverse rank of the first relevant item."""
    for idx, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0

def map_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates Mean Average Precision at K (MAP@K)."""
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1.0
            score += hits / idx
    return score / min(len(relevant), k)

def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    recommended_at_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(recommended_at_k, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(idx + 1)
    
    # Ideal DCG: perfect ranking where all relevant items appear first
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0