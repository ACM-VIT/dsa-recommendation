"""
evaluation/evaluator.py

OfflineEvaluator: computes the standard ranking metrics (from
evaluation.metrics.ranking) and pool attribution (from
evaluation.metrics.pool_metrics) for a single user's recommendations, and
averages them across a batch of users. Pure computation over whatever
recommended/relevant/item_pools data it is given -- it does not fetch data,
call the recommender, or load a model itself.
"""
from typing import List, Dict, Optional, Set
from evaluation.metrics.ranking import precision_at_k, recall_at_k, ndcg_at_k, map_at_k, mrr
from evaluation.metrics.pool_metrics import pool_contribution

class OfflineEvaluator:
    """Central Evaluation Engine for computing batch recommendation metrics.

    Args:
        k_values: the list of @K cutoffs (e.g. [5, 10]) that
            precision/recall/MAP/NDCG are each computed at.
    """

    def __init__(self, k_values: List[int] = [5, 10]):
        self.k_values = k_values

    def evaluate_user(self, recommended: List[str], relevant: Set[str], item_pools: Dict[str, str] = None,
                       graded_relevance: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculates all ranking and pool metrics for a single user.

        Args:
            recommended: ranked list of recommended item IDs, best first.
            relevant: set of item IDs that count as relevant/ground-truth
                for this user. Used for every metric except NDCG when
                graded_relevance is supplied (see below).
            item_pools: optional {item_id: pool_name}, for the
                candidate-pool attribution metrics (pool_<name>_ratio).
                Omitted entirely from the result if not supplied.
            graded_relevance: optional {item_id: relevance grade} (e.g.
                this project's ternary 0/1/2 engagement labels). When
                supplied, NDCG@K is computed from these grades instead of
                the binary `relevant` set, so it can distinguish e.g.
                "solved" from "attempted but not solved" -- exactly the
                distinction a binary relevant/not-relevant set collapses.
                precision/recall/MAP/MRR are unaffected either way; they
                always use the binary `relevant` set.

        Returns:
            {"mrr": ..., "precision@<k>": ..., "recall@<k>": ...,
             "map@<k>": ..., "ndcg@<k>": ..., ["pool_<name>_ratio": ...]}
            for every k in self.k_values.
        """
        metrics = {
            'mrr': mrr(recommended, relevant)
        }

        # NDCG uses graded_relevance when supplied; every other metric
        # always uses the binary relevant set, unaffected by this choice.
        ndcg_relevance = graded_relevance if graded_relevance is not None else relevant

        # Compute top-K metrics for all required thresholds (e.g. @5, @10)
        for k in self.k_values:
            metrics[f'precision@{k}'] = precision_at_k(recommended, relevant, k)
            metrics[f'recall@{k}'] = recall_at_k(recommended, relevant, k)
            metrics[f'map@{k}'] = map_at_k(recommended, relevant, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(recommended, ndcg_relevance, k)

        if item_pools:
            pool_res = pool_contribution(recommended, item_pools)
            for pool_name, ratio in pool_res.items():
                metrics[f'pool_{pool_name}_ratio'] = ratio

        return metrics

    def evaluate_batch(self, user_data: List[Dict]) -> Dict[str, float]:
        """Evaluates a batch of users and calculates average metrics across the population.

        Args:
            user_data: list of per-user entries, each shaped as:
                {
                    "recommended": ["item1", "item2", "item3"],
                    "relevant": ["item2", "item4"],
                    "item_pools": {"item1": "vector", "item2": "heuristic"}, # optional
                    "graded_relevance": {"item2": 2.0, "item4": 1.0} # optional
                }

        Returns:
            {metric_name: average_score}, one entry per metric
            evaluate_user() can produce. Each metric is averaged only over
            the users that actually contributed a value for it (see the
            per-metric `counts` below) -- not every user necessarily
            supplies item_pools, so pool_<name>_ratio metrics are averaged
            over a smaller denominator than e.g. mrr. Empty input returns
            an empty dict.
        """
        if not user_data:
            return {}

        accumulated = {}
        counts = {}

        for entry in user_data:
            res = self.evaluate_user(
                recommended=entry['recommended'],
                relevant=set(entry['relevant']),
                item_pools=entry.get('item_pools'),
                graded_relevance=entry.get('graded_relevance'),
            )

            # Sum up scores across users, tracking a separate count per
            # metric -- optional metrics (e.g. pool_*_ratio) are only
            # present for users whose entry had item_pools, so they must be
            # averaged over the users that actually contributed them, not
            # over every user in the batch.
            for key, val in res.items():
                if isinstance(val, (int, float)):
                    accumulated[key] = accumulated.get(key, 0.0) + val
                    counts[key] = counts.get(key, 0) + 1

        # Compute average (mean) for each metric over the users that
        # actually contributed it.
        summary = {key: total / counts[key] for key, total in accumulated.items()}
        return summary