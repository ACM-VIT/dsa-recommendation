# Offline Evaluation Framework

## Purpose

Compares the production **HeuristicRanker** against the trained
**LightGBM LambdaRank model** on the held-out validation split the
training pipeline produced — precision/recall/MAP/MRR/NDCG@K, side by
side, computed from real recommendation data (synthetic bootstrap data
today; real `recommendation_log` data once it exists — see below).

This is an **offline batch evaluator**. It does not call the live
recommendation service, does not depend on Qdrant/Postgres/Neo4j, and is
independent of the `RANKER` environment variable that controls which
ranker actually serves traffic (see
[`pipeline/recommender/services/lightgbm_ranker.py`](../pipeline/recommender/services/lightgbm_ranker.py)).
Running an evaluation never changes which ranker is live.

## Architecture

```
evaluation/
  data_loader.py   -- reads validation.parquet, builds per-query-group
                       input (recommended order, relevant set, graded
                       relevance, pool attribution) for OfflineEvaluator
  evaluator.py      -- OfflineEvaluator: precision@K/recall@K/MAP@K/MRR/
                       NDCG@K (binary or graded) + pool-contribution ratios
  report.py         -- Markdown report, JSON summary, console comparison table
  experiments.py    -- the real entry point: wires the above together
                       against the actual trained model and dataset
  metrics/
    ranking.py       -- precision_at_k, recall_at_k, mrr, map_at_k, ndcg_at_k
    pool_metrics.py   -- pool_contribution
    user_metrics.py   -- cold_start_performance, topic_coverage (not
                          currently wired into evaluator.py -- available
                          for a caller that wants to isolate these segments)
    diversity.py      -- calculate_diversity/coverage/novelty (same status
                          as user_metrics.py: implemented, not yet wired in)
```

Nothing here is duplicated from the training pipeline. `data_loader.py`
and `experiments.py` import the real `FeatureRegistry`
(`training/feature_registry.py`) and the real `LightGBMRanker`
(`pipeline/recommender/services/lightgbm_ranker.py`) directly — there is
exactly one implementation of each in this repository.

## Required artifacts

Both must exist under `training/artifacts/` before running an evaluation:

| File | Produced by | Used for |
|---|---|---|
| `validation.parquet` | `training/generate_production_dataset.py` (via `DatasetGenerator`/`LabelGenerator`) | the held-out query groups + ternary 0/1/2 labels being evaluated |
| `lightgbm_model.txt` | `training/train_lightgbm.py` | the trained booster `LightGBMRanker` scores with |
| `model_metadata.json` | `training/train_lightgbm.py` (via `training/model_metadata.py`) | model version/training timestamp/best iteration, embedded in the JSON summary's `meta` block (optional — evaluation still runs without it, just without that provenance info) |

`training/artifacts/` is gitignored (generated output produced
locally by the training pipeline, not committed).

## Training workflow (prerequisite, not part of this package)

```bash
# 1. Generate the labeled train/validation split
python -m training.generate_production_dataset

# 2. Train the LightGBM model against that split
python -m training.train_lightgbm
```

After this, `training/artifacts/` contains everything the evaluation
workflow below needs.

## Evaluation workflow

```bash
python -m evaluation.experiments
```

What it does, in order:
1. Loads `training/artifacts/validation.parquet` as-is (no filtering/resampling).
2. Loads the trained model via `LightGBMRanker` and scores every row with
   `score_dataframe()` (batch scoring on already-extracted features — no
   feature is recomputed, no candidate is fabricated).
3. Builds two `OfflineEvaluator`-shaped batches from the *same* dataframe
   — one ordered by the heuristic ranker's own recorded score
   (`current_heuristic_rank_score`, computed once at dataset-generation
   time by the real `HeuristicRanker`, reused here rather than
   recomputed), one ordered by the LightGBM score just computed. Same
   query groups, same labels, same pool attribution for both — only the
   candidate order differs.
4. Evaluates both batches and prints/saves the comparison.

Deterministic: a fixed input file, LightGBM inference has no randomness,
and the per-group sort is stable — running this twice against the same
`validation.parquet` and model produces byte-identical output.

If a required artifact is missing, this exits with a clear message and a
non-zero exit code rather than falling back to a heuristic-only
comparison or printing a raw traceback — see
[Troubleshooting](#troubleshooting).

## Generated reports

All written to `evaluation/reports/` (gitignored — generated output):

### Markdown report (`<model_name>_eval_report.md`)
A table of the candidate model's own metric scores. Human-readable,
meant for a PR description or a quick look.

### JSON summary (`<model_name>_eval_summary.json`)
```json
{
  "model_name": "LightGBM",
  "heuristic": { "mrr": 0.85, "ndcg@5": 0.76, "...": "..." },
  "LightGBM":  { "mrr": 0.86, "ndcg@5": 0.79, "...": "..." },
  "diff":      { "mrr": 0.015, "ndcg@5": 0.036, "...": "..." },
  "meta": {
    "validation_dataset_path": "...",
    "n_rows": 16810,
    "n_query_groups": 3392,
    "model_version": "lgbm-...",
    "dataset_version": "..."
  }
}
```
Machine-readable — the shape another script or tool would consume to
compare the `diff` block programmatically, without re-parsing Markdown.

### Console comparison
A side-by-side table (`Metric | Baseline | LightGBM | Diff`) printed
during the run.

## Graded NDCG support

`evaluation/metrics/ranking.py::ndcg_at_k` accepts relevance as either:
- `Set[str]` — binary (the original behaviour, unchanged for any other caller)
- `Dict[str, float]` — graded, e.g. this project's ternary 0/1/2 engagement
  labels (0 = ignored, 1 = attempted but not solved, 2 = solved)

`data_loader.build_user_data()` always builds **both** for every query
group from the same `label` column: a binary `relevant` set (`label > 0`)
used by precision/recall/MAP/MRR, and a `graded_relevance` dict (the
actual 0/1/2 grade, for items with `label > 0`) used *only* by NDCG. This
means NDCG can tell "solved" apart from "attempted but not solved" —
exactly the distinction a plain relevant/not-relevant set would collapse
— while every other metric still sees the coarser binary signal it was
already defined against. Gain is `2**grade - 1` in both cases, the same
gain LightGBM's own `lambdarank`/`ndcg` objective uses, so the graded
NDCG@K reported here is comparable to what the model was actually
optimized for.

## Heuristic vs. LightGBM comparison

Both rankers are scored from the **same** `validation.parquet` rows —
same query groups, same labels, same pool attribution — so the only
variable between the two `OfflineEvaluator.evaluate_batch()` calls is
candidate *order*. `precision@10`/`recall@10` frequently show `0.0000`
diff in practice: both are order-insensitive once a query group has fewer
candidates than `k` (the norm here — group sizes average ~5), so only the
rank-sensitive metrics (NDCG, MRR, MAP) are expected to actually
differentiate the two rankers. That is expected behaviour, not a bug.

## Expected directory layout

```
training/artifacts/
  validation.parquet       <- required
  lightgbm_model.txt        <- required
  model_metadata.json       <- optional (evaluation degrades gracefully without it)
evaluation/
  reports/                  <- generated by running evaluation, gitignored
    <model_name>_eval_report.md
    <model_name>_eval_summary.json
```

## How future real `recommendation_log` data fits in

Today, `validation.parquet` comes from `training/synthetic_user_generator.py`'s
bootstrap data, because `recommendation_log` is still empty in production.
**No code in this package changes when that stops being true.**
`data_loader.py`/`experiments.py` are schema-based, not source-based: once
a real dataset-generation run over real `recommendation_log`/`submission`
rows produces a new `validation.parquet` with the same columns, this
evaluation framework consumes it exactly the same way, with no changes
required here.

## Troubleshooting

### "No trained LightGBM model found"
```
No trained LightGBM model found (or it failed to load).
  Reason: failed to load LightGBM model from .../lightgbm_model.txt: ...

Run:

    python -m training.train_lightgbm

before running offline evaluation.
```
`training/artifacts/lightgbm_model.txt` is missing, unreadable, or its
feature schema no longer matches `FeatureRegistry` (e.g. stale model
after a feature was added/removed). Run training again. This is a hard
stop by design — evaluation never silently falls back to a heuristic-only
comparison, since that would misrepresent what was actually evaluated.

### "Validation dataset not found"
```
Validation dataset not found.
  Reason: [Errno 2] No such file or directory: '.../validation.parquet'

Run the training pipeline to generate training/artifacts/validation.parquet ...
```
Run `python -m training.generate_production_dataset` first.

### Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `LightGBMRankerError: ... feature order does not match FeatureRegistry` | Model was trained against an older/different feature set | Retrain: `python -m training.train_lightgbm` |
| Metrics look identical between heuristic and LightGBM at `@10` but differ at `@5` | Expected — see [Heuristic vs. LightGBM comparison](#heuristic-vs-lightgbm-comparison) | Not a bug |
| `evaluation/reports/` missing after a run | The run exited early (see the two error messages above) — check the exit code | Re-run after fixing the underlying cause |
