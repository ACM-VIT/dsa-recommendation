import numpy as np
from typing import List, Dict, Set

def calculate_diversity(recommendations: List[str], item_embeddings: Dict[str, list]) -> float:
    """Intra-list Diversity: Average pairwise cosine distance between recommended items."""
    vecs = [item_embeddings[item] for item in recommendations if item in item_embeddings]
    if len(vecs) < 2:
        return 0.0
    
    distances = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            v1, v2 = np.array(vecs[i]), np.array(vecs[j])
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            distances.append(1.0 - sim)
            
    return float(np.mean(distances)) if distances else 0.0

def calculate_coverage(recommended_items_all_users: List[str], catalog_items: Set[str]) -> float:
    """Catalog Coverage: Proportion of unique catalog items recommended across all users."""
    if not catalog_items:
        return 0.0
    unique_recommended = set(recommended_items_all_users)
    return len(unique_recommended.intersection(catalog_items)) / len(catalog_items)

def calculate_novelty(recommendations: List[str], item_pop_df: Dict[str, float]) -> float:
    """Novelty: Measures self-information/unexpectedness based on item popularity."""
    novelty_scores = []
    for item in recommendations:
        prob = item_pop_df.get(item, 1e-6)
        novelty_scores.append(-np.log2(prob))
    return float(np.mean(novelty_scores)) if novelty_scores else 0.0