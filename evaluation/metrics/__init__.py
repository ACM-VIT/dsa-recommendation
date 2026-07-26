"""
evaluation.metrics

Individual, dependency-free metric functions used by
evaluation.evaluator.OfflineEvaluator:

    ranking.py       -- precision_at_k, recall_at_k, mrr, map_at_k, ndcg_at_k
    pool_metrics.py  -- pool_contribution (candidate-pool attribution)
    user_metrics.py  -- cold_start_performance, topic_coverage
    diversity.py     -- calculate_diversity, calculate_coverage, calculate_novelty

Each function takes plain lists/sets/dicts and returns a float or a dict of
floats -- no shared state, no dependency on the recommender pipeline itself.
"""
