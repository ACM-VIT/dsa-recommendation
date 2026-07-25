"""
evaluation/data_loader.py

Builds OfflineEvaluator-shaped input directly from
training/artifacts/validation.parquet -- the held-out split the training
pipeline produced (never train.parquet), so evaluation never runs on data
either the model or the heuristic ranker's own tuning could have seen.

Each validation row is one (query, candidate) pair, already carrying:
  - candidate_id                     the item id
  - current_heuristic_rank_score     HeuristicRanker's own score for this
                                      exact candidate, computed once at
                                      dataset-generation time
                                      (feature_extractor.py) -- reused
                                      directly here, never recomputed
  - label                            ternary 0/1/2 engagement grade
  - from_pool_*                      which candidate pool(s) proposed it
  - every FeatureRegistry.model_matrix_features() column, for LightGBM

build_user_data() turns a scored DataFrame into OfflineEvaluator.evaluate_batch()
input, ordering each query group's candidates by whatever score column the
caller names. The same function builds both rankers' input (heuristic:
score_column="current_heuristic_rank_score"; LightGBM: whatever column the
caller attached the model's predictions to), so both are evaluated over
identical query groups, labels, and pool attribution -- only the candidate
ORDER differs between the two.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from training.config import VALIDATION_DATASET_PATH

# Priority order used only to pick ONE pool label per candidate for
# pool_contribution's {item: pool_name} input -- a candidate can be
# proposed by multiple pools (pool_count > 1), but pool attribution only
# tracks a single label per item. Order is otherwise arbitrary.
_POOL_COLUMNS = (
    "from_pool_A", "from_pool_B_C", "from_pool_D", "from_pool_E",
    "from_pool_F", "from_pool_G", "from_pool_vector",
)


def load_validation_dataframe(path=VALIDATION_DATASET_PATH) -> pd.DataFrame:
    """Reads the validation dataset as-is -- no filtering, resampling, or
    row reordering. Query groups must stay exactly as the training
    pipeline produced them."""
    return pd.read_parquet(path)


def iter_query_groups(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yields (query_id, group_df) for each contiguous block of rows
    sharing a query_id, in the DataFrame's own row order.
    DatasetGenerator.generate_dataframe() already sorts by
    (user_id, recommended_at, query_id) before export, so every query_id's
    rows are contiguous here -- this only finds the block boundaries, it
    does not re-sort anything.
    """
    if df.empty:
        return
    query_ids = df["query_id"].to_numpy()
    start = 0
    current = query_ids[0]
    for i in range(1, len(query_ids)):
        if query_ids[i] != current:
            yield current, df.iloc[start:i]
            start, current = i, query_ids[i]
    yield current, df.iloc[start:]


def _item_pool(row) -> str:
    """The first pool (in _POOL_COLUMNS order) whose from_pool_<X> column
    is True for this row; 'unknown' if none are set."""
    for col in _POOL_COLUMNS:
        if bool(getattr(row, col)):
            return col[len("from_pool_"):]
    return "unknown"


def build_user_data(df: pd.DataFrame, score_column: str) -> list[dict]:
    """
    Builds one OfflineEvaluator.evaluate_batch()-shaped entry per query
    group in `df`, with candidates ordered by df[score_column] descending
    (ties broken by original row order -- pandas' sort is stable, so this
    is deterministic given the same input).

    Args:
        df: validation rows (or a subset/superset with the same schema),
            containing query_id, candidate_id, label, from_pool_* columns,
            and `score_column`.
        score_column: the column to rank each query group's candidates by
            (e.g. "current_heuristic_rank_score", or a column the caller
            attached with a model's predicted scores).

    Returns:
        List of {"recommended": [...], "relevant": {...}, "graded_relevance":
        {...}, "item_pools": {...}, "query_id": ...} -- directly usable as
        OfflineEvaluator.evaluate_batch()'s user_data. "relevant" and
        "graded_relevance" are both derived from the SAME label column:
        relevant is binary (label > 0 -- attempted or solved counts as
        relevant, matching every other binary metric here), graded_relevance
        keeps the actual 0/1/2 grade for items with label > 0, so NDCG can
        distinguish "solved" from "attempted but not solved" while every
        other metric still sees the coarser binary signal.
    """
    user_data = []
    for query_id, group in iter_query_groups(df):
        ordered = group.sort_values(score_column, ascending=False, kind="stable")
        recommended = ordered["candidate_id"].tolist()

        labels_by_id = dict(zip(group["candidate_id"], group["label"]))
        relevant = {cid for cid, label in labels_by_id.items() if label > 0}
        graded_relevance = {cid: float(label) for cid, label in labels_by_id.items() if label > 0}
        item_pools = {row.candidate_id: _item_pool(row) for row in group.itertuples()}

        user_data.append({
            "query_id": query_id,
            "recommended": recommended,
            "relevant": relevant,
            "graded_relevance": graded_relevance,
            "item_pools": item_pools,
        })
    return user_data
