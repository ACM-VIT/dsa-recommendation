from typing import List, Dict, Set
from evaluation.metrics.ranking import precision_at_k, recall_at_k, ndcg_at_k, map_at_k, mrr
from evaluation.metrics.pool_metrics import pool_contribution

class OfflineEvaluator:
    """Central Evaluation Engine for computing batch recommendation metrics."""
    
    def __init__(self, k_values: List[int] = [5, 10]):
        self.k_values = k_values

    def evaluate_user(self, recommended: List[str], relevant: Set[str], item_pools: Dict[str, str] = None) -> Dict[str, float]:
        """Calculates all ranking and pool metrics for a single user."""
        metrics = {
            'mrr': mrr(recommended, relevant)
        }
        
        # Compute top-K metrics for all required thresholds (e.g. @5, @10)
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
        """Evaluates a batch of users and calculates average metrics across the population.
        
        user_data expected structure:
        [
            {
                "recommended": ["item1", "item2", "item3"],
                "relevant": ["item2", "item4"],
                "item_pools": {"item1": "vector", "item2": "heuristic"} # optional
            }, ...
        ]
        """
        if not user_data:
            return {}

        accumulated = {}
        total_users = len(user_data)

        for entry in user_data:
            res = self.evaluate_user(
                recommended=entry['recommended'], 
                relevant=set(entry['relevant']), 
                item_pools=entry.get('item_pools')
            )
            
            # Sum up scores across users
            for key, val in res.items():
                if isinstance(val, (int, float)):
                    accumulated[key] = accumulated.get(key, 0.0) + val

        # Compute average (mean) for each metric
        summary = {key: total / total_users for key, total in accumulated.items()}
        return summary