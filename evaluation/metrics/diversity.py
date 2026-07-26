"""
evaluation/metrics/diversity.py

List- and catalog-level metrics that look beyond relevance: intra-list
diversity, catalog coverage, and novelty. Not currently called from
evaluator.py/experiments.py -- available for a caller with item
embeddings/catalog/popularity data to pass in.
"""
import numpy as np
from typing import List, Dict, Set

def calculate_diversity(recommendations: List[str], item_embeddings: Dict[str, list]) -> float:
    """Intra-list Diversity: average pairwise cosine distance between the
    embeddings of recommended items (higher = more diverse).

    Args:
        recommendations: list of recommended item IDs.
        item_embeddings: {item_id: embedding vector}. Items missing from
            this dict are skipped rather than raising.

    Returns:
        Mean pairwise cosine distance (1 - cosine similarity) over all
        pairs of embedded items; 0.0 if fewer than 2 items have embeddings.
    """
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
    """Catalog Coverage: what fraction of the full catalog was ever
    recommended to any user in the batch.

    Args:
        recommended_items_all_users: recommended item IDs pooled across
            every user in the batch (duplicates are fine, deduplicated here).
        catalog_items: the full set of item IDs available to recommend.

    Returns:
        |unique recommended items ∩ catalog| / |catalog|, or 0.0 if
        catalog_items is empty.
    """
    if not catalog_items:
        return 0.0
    unique_recommended = set(recommended_items_all_users)
    return len(unique_recommended.intersection(catalog_items)) / len(catalog_items)

def calculate_novelty(recommendations: List[str], item_pop_df: Dict[str, float]) -> float:
    """Novelty: mean self-information (-log2 popularity) of the recommended
    items -- higher means the list favors less-popular, more surprising items.

    Args:
        recommendations: list of recommended item IDs.
        item_pop_df: {item_id: popularity probability in (0, 1]}. An item
            missing from this dict is treated as extremely rare (1e-6)
            rather than raising.

    Returns:
        Mean of -log2(popularity) over recommendations; 0.0 if
        recommendations is empty.
    """
    novelty_scores = []
    for item in recommendations:
        prob = item_pop_df.get(item, 1e-6)
        novelty_scores.append(-np.log2(prob))
    return float(np.mean(novelty_scores)) if novelty_scores else 0.0