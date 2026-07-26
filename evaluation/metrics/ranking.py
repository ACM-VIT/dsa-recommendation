"""
evaluation/metrics/ranking.py

Standard information-retrieval ranking metrics: Precision@K, Recall@K,
MRR, MAP@K, and (binary or graded) NDCG@K. Each function is a pure
computation over a `recommended` list and a `relevant` set/dict -- no
shared state, no I/O, no dependency on the rest of this package.
"""
import numpy as np
from typing import Dict, List, Set, Union

# Set[str]: binary relevance (item present = relevant). Dict[str, float]:
# graded relevance, item -> relevance grade (e.g. this project's ternary
# 0/1/2 engagement labels). ndcg_at_k is the only metric that accepts both;
# every other metric in this module is binary-only, unchanged.
RelevanceInput = Union[Set[str], Dict[str, float]]

def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates Precision@K: the fraction of the top-k recommended items
    that are relevant.

    Divides by the number of items actually returned (len(recommended_at_k)),
    not by k -- fewer than k recommendations is a real, common case in this
    system (candidate slates are frequently smaller than the requested k),
    and dividing by k in that case silently understates precision.

    Args:
        recommended: ranked list of recommended item IDs, best first.
        relevant: set of ground-truth relevant item IDs.
        k: cutoff rank.

    Returns:
        hits in the top-k / number of items actually in the top-k slice
        (0.0 if recommended is empty).
    """
    recommended_at_k = recommended[:k]
    if not recommended_at_k:
        return 0.0
    hits = sum(1 for item in recommended_at_k if item in relevant)
    return hits / len(recommended_at_k)

def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates Recall@K: the fraction of all relevant items that were
    retrieved within the top-k recommendations.

    Args:
        recommended: ranked list of recommended item IDs, best first.
        relevant: set of ground-truth relevant item IDs.
        k: cutoff rank.

    Returns:
        hits in the top-k / len(relevant) (0.0 if relevant is empty).
    """
    if not relevant:
        return 0.0
    recommended_at_k = recommended[:k]
    hits = sum(1 for item in recommended_at_k if item in relevant)
    return hits / len(relevant)

def mrr(recommended: List[str], relevant: Set[str]) -> float:
    """Calculates the Reciprocal Rank: 1 / (rank of the first relevant item),
    or 0.0 if no relevant item appears anywhere in `recommended`. Named
    "mrr" because averaging this value across many users/queries (e.g. in
    OfflineEvaluator.evaluate_batch) is what turns it into the conventional
    Mean Reciprocal Rank.

    Args:
        recommended: ranked list of recommended item IDs, best first.
        relevant: set of ground-truth relevant item IDs.

    Returns:
        1.0 / rank of the first hit, or 0.0 if there is no hit.
    """
    for idx, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0

def map_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    """Calculates (Mean) Average Precision at K: the average of
    precision-at-each-hit-rank, normalized by min(len(relevant), k).

    Args:
        recommended: ranked list of recommended item IDs, best first.
        relevant: set of ground-truth relevant item IDs.
        k: cutoff rank.

    Returns:
        Average precision over the top-k, in [0.0, 1.0]; 0.0 if relevant
        is empty.
    """
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1.0
            score += hits / idx
    return score / min(len(relevant), k)

def ndcg_at_k(recommended: List[str], relevant: RelevanceInput, k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain at K (NDCG@K).

    `relevant` accepts either:
      - Set[str]: binary relevance -- exactly the original behaviour,
        unchanged (a member is relevance grade 1, everything else 0).
      - Dict[str, float]: graded relevance, item -> relevance grade (e.g.
        this project's ternary 0/1/2 engagement labels: ignored/attempted/
        solved). Needed because the production LambdaRank model this
        framework will eventually evaluate is trained on graded labels,
        and collapsing "solved" and "attempted-but-failed" into the same
        binary "relevant" bucket discards exactly the distinction that
        labeling strategy exists to capture.

    Gain is 2**grade - 1 in both cases (grade=1 -> gain=1, matching the
    original binary formula exactly) -- the same gain LightGBM's own
    lambdarank/ndcg objective uses, so a graded NDCG@K computed here is
    comparable to what the model was actually trained to optimize.

    Args:
        recommended: ranked list of recommended item IDs, best first.
        relevant: Set[str] (binary) or Dict[str, float] (graded), as above.
        k: cutoff rank.

    Returns:
        DCG@k / ideal-DCG@k, in [0.0, 1.0]; 0.0 if the ideal DCG is 0
        (e.g. relevant is empty).
    """
    def _grade(item: str) -> float:
        if isinstance(relevant, dict):
            return float(relevant.get(item, 0.0))
        return 1.0 if item in relevant else 0.0

    recommended_at_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(recommended_at_k, start=1):
        grade = _grade(item)
        if grade > 0:
            dcg += (2 ** grade - 1) / np.log2(idx + 1)

    # Ideal DCG: perfect ranking, best grades first.
    if isinstance(relevant, dict):
        ideal_grades = sorted(relevant.values(), reverse=True)[:k]
    else:
        ideal_grades = [1.0] * min(len(relevant), k)
    idcg = sum((2 ** grade - 1) / np.log2(i + 2) for i, grade in enumerate(ideal_grades))

    return dcg / idcg if idcg > 0 else 0.0