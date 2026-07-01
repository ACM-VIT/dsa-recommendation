# conftest.py — pytest root configuration
# Excludes CLI scripts that happen to have test_ names/functions
# but are not pytest test modules.
collect_ignore = [
    "database/qdrant/test_qdrant.py",       # CLI smoke-test, not a pytest module
    "pipeline/graphs/test_rgcn_embeddings.py",  # CLI evaluator, not a pytest module
]
collect_ignore_glob = []
