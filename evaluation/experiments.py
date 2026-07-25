"""
evaluation/experiments.py

Real offline evaluation pipeline (not a mock demo): loads
training/artifacts/validation.parquet -- the held-out split the training
pipeline produced -- scores it with the production LightGBMRanker, reuses
the production HeuristicRanker's own recorded score for each candidate
(evaluation.data_loader's "current_heuristic_rank_score" column, computed
once by feature_extractor.py at dataset-generation time, never recomputed
here), evaluates both over the exact same query groups and ternary 0/1/2
labels, and writes a Markdown report + JSON summary + console comparison.

No feature is recomputed and no candidate/label is fabricated: every row
this evaluates came from the real training pipeline's validation split.
"""
from __future__ import annotations

from typing import Any, Optional

from evaluation.data_loader import build_user_data, load_validation_dataframe
from evaluation.evaluator import OfflineEvaluator
from evaluation.report import generate_markdown_report, print_cli_summary, save_json_summary

from pipeline.recommender.services.lightgbm_ranker import LightGBMRanker

from training.config import VALIDATION_DATASET_PATH

# Column build_user_data() ranks the heuristic's candidates by -- already
# present in validation.parquet, computed by the real HeuristicRanker at
# dataset-generation time (feature_extractor.py), reused as-is.
HEURISTIC_SCORE_COLUMN = "current_heuristic_rank_score"
# Column this module attaches LightGBMRanker's own predictions to, so
# build_user_data() can rank LightGBM's candidates the same way.
LIGHTGBM_SCORE_COLUMN = "_lightgbm_score"


def run_experiment(heuristic_results: list, model_results: list, model_name: str = "LightGBM",
                    meta: Optional[dict[str, Any]] = None) -> tuple[dict, dict]:
    """Runs batch evaluation for baseline and candidate models, prints CLI
    comparison, and saves a Markdown report + JSON summary.

    Args:
        heuristic_results: batch input for the baseline ranker, shaped as
            OfflineEvaluator.evaluate_batch()'s user_data (list of
            {"recommended": [...], "relevant": [...], "item_pools": {...},
             "graded_relevance": {...}}).
        model_results: same shape, for the candidate model being compared.
        model_name: label used in the CLI table and the report title/filename.
        meta: optional extra context (dataset path, model version, row/query
            counts) embedded in the JSON summary as-is.

    Returns:
        (heuristic_metrics, candidate_metrics) -- the two averaged metric
        dicts, in case a caller wants them beyond the printed/saved output.
    """

    evaluator = OfflineEvaluator(k_values=[5, 10])

    print("Evaluating Baseline (Heuristic)...")
    heuristic_metrics = evaluator.evaluate_batch(heuristic_results)

    print(f"Evaluating Candidate ({model_name})...")
    candidate_metrics = evaluator.evaluate_batch(model_results)

    # Print formatted comparison in Terminal
    print_cli_summary(
        heuristic_metrics=heuristic_metrics,
        model_metrics=candidate_metrics,
        model_name=model_name
    )

    # Save markdown report and JSON summary to disk
    report_path = f"{model_name.lower()}_eval_report.md"
    generate_markdown_report(candidate_metrics, output_filepath=report_path, title=f"{model_name} Evaluation Report")

    json_path = f"{model_name.lower()}_eval_summary.json"
    save_json_summary(heuristic_metrics, candidate_metrics, output_filepath=json_path,
                       model_name=model_name, meta=meta)

    return heuristic_metrics, candidate_metrics


def run_validation_evaluation(validation_path=VALIDATION_DATASET_PATH, model_name: str = "LightGBM") -> tuple[dict, dict]:
    """
    The real, non-mock entry point: loads the validation dataset, scores it
    with the production LightGBMRanker, builds both rankers' input over the
    exact same query groups/labels via evaluation.data_loader, and runs the
    comparison via run_experiment().

    Deterministic: reads a fixed file, LightGBM inference has no randomness,
    and build_user_data()'s per-group sort is stable -- running this twice
    against the same validation.parquet and model produces identical output.
    """
    df = load_validation_dataframe(validation_path)

    ranker = LightGBMRanker()
    ranker.load_model()

    df = df.copy()
    df[LIGHTGBM_SCORE_COLUMN] = ranker.score_dataframe(df)

    heuristic_user_data = build_user_data(df, score_column=HEURISTIC_SCORE_COLUMN)
    lightgbm_user_data = build_user_data(df, score_column=LIGHTGBM_SCORE_COLUMN)

    model_info = ranker.get_model_info() or {}
    meta = {
        "validation_dataset_path": str(validation_path),
        "n_rows": int(len(df)),
        "n_query_groups": len(heuristic_user_data),
        "model_version": model_info.get("model_version"),
        "dataset_version": model_info.get("dataset_version"),
    }

    return run_experiment(heuristic_user_data, lightgbm_user_data, model_name=model_name, meta=meta)


if __name__ == "__main__":
    run_validation_evaluation()
