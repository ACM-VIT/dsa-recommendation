from typing import List, Dict, Set, Optional, Union
from evaluation.metrics.ranking import precision_at_k, recall_at_k, ndcg_at_k, map_at_k, mrr
from evaluation.metrics.pool_metrics import pool_contribution

class OfflineEvaluator:
    """Central Evaluation Engine for computing batch recommendation metrics."""
    
    def __init__(self, k_values: Optional[List[int]] = None):
        self.k_values = k_values if k_values is not None else [5, 10]

    def evaluate_user(
        self, 
        recommended: List[str], 
        relevant: Union[Set[str], Dict[str, float]], 
        item_pools: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Calculates ranking and pool metrics for a single user."""
        metrics = {
            'mrr': mrr(recommended, relevant)
        }
        
        for k in self.k_values:
            metrics[f'precision@{k}'] = precision_at_k(recommended, relevant, k)
            metrics[f'recall@{k}'] = recall_at_k(recommended, relevant, k)
            metrics[f'map@{k}'] = map_at_k(recommended, relevant, k)
            metrics[f'ndcg@{k}'] = ndcg_at_k(recommended, relevant, k)

        if item_pools:
            pool_res = pool_contribution(recommended, item_pools)
            for pool_name, ratio in pool_res.items():
                metrics[f'pool_{pool_name}_ratio'] = ratio

        return metrics

    def evaluate_batch(self, user_data: List[Dict]) -> Dict[str, float]:
        """Evaluates a list of users and returns average scores per metric."""
        if not user_data:
            return {}

        accumulated = {}
        metric_counts = {}

        for entry in user_data:
            res = self.evaluate_user(
                recommended=entry['recommended'], 
                relevant=entry['relevant'], 
                item_pools=entry.get('item_pools')
            )
            for key, val in res.items():
                if isinstance(val, (int, float)):
                    accumulated[key] = accumulated.get(key, 0.0) + val
                    metric_counts[key] = metric_counts.get(key, 0) + 1

        # Calculate mean using the exact count of users who contributed to each metric
        summary = {key: accumulated[key] / metric_counts[key] for key in accumulated}
        return summary