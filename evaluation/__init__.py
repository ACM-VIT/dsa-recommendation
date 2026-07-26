"""
evaluation

Offline evaluation library for recommendation output: standard information-
retrieval metrics (precision/recall/MAP/MRR/NDCG), candidate-pool
attribution, and batch/report utilities for comparing a baseline ranker
against a candidate model.

Submodules:
    evaluation.metrics     -- individual metric implementations
    evaluation.evaluator   -- OfflineEvaluator, orchestrates per-user and
                              batch metric computation
    evaluation.report      -- Markdown/CLI presentation of computed metrics
    evaluation.experiments -- example/demo runner wiring the above together
"""
