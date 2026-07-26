"""
evaluation/report.py

Presentation layer for OfflineEvaluator's output: renders an already-computed
metrics dict as a Markdown report file, a JSON summary, and/or a terminal
comparison table. None of these functions compute, alter, or re-derive any
metric value -- all three are pure formatting/serialization over whatever
dict(s) they are given.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Default destination for generated Markdown reports, kept out of the
# package root so ad hoc report files don't clutter it or the repo root.
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def generate_markdown_report(
    metrics_dict: Dict[str, Any],
    output_filepath: str = "eval_report.md",
    title: str = "Offline Evaluation Report"
) -> str:
    """Renders `metrics_dict` as a Markdown table and writes it to disk.

    Args:
        metrics_dict: metric name -> score, e.g. the dict returned by
            OfflineEvaluator.evaluate_batch(). Values are only formatted
            for display here, never recomputed.
        output_filepath: where to write the report. A bare filename (no
            directory component -- the default, and what experiments.py
            passes) is written under evaluation/reports/, created if it
            doesn't exist yet. Pass a path that includes a directory to
            write somewhere else instead.
        title: heading text for the report.

    Returns:
        The exact Markdown string that was written to disk.
    """
    output_path = Path(output_filepath)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = REPORTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"## {title}\n",
        "| Metric | Score |",
        "| :--- | :--- |"
    ]

    for metric, score in sorted(metrics_dict.items()):
        if isinstance(score, float):
            md_lines.append(f"| **{metric}** | `{score:.4f}` |")
        else:
            md_lines.append(f"| **{metric}** | `{score}` |")

    content = "\n".join(md_lines) + "\n"

    # Save report to disk
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"## Evaluation report generated at: {output_path}")
    return content


def save_json_summary(
    heuristic_metrics: Dict[str, float],
    model_metrics: Dict[str, float],
    output_filepath: str = "eval_summary.json",
    model_name: str = "Model",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Writes a machine-readable JSON summary alongside the Markdown report
    -- the same two metric dicts print_cli_summary()/generate_markdown_report()
    use, plus their per-metric diff, so a CI job or another script can
    consume the comparison without re-parsing Markdown.

    Args:
        heuristic_metrics: metric name -> score for the baseline ranker.
        model_metrics: metric name -> score for the candidate model.
        output_filepath: where to write the JSON file. A bare filename (no
            directory component -- the default) is written under
            evaluation/reports/, same convention as generate_markdown_report.
        model_name: label recorded for the candidate model.
        meta: optional extra context to embed as-is (e.g. dataset path,
            model version, row/query counts) -- not interpreted here.

    Returns:
        The exact dict that was written to disk:
        {"model_name": ..., "heuristic": {...}, model_name: {...},
         "diff": {metric: model - heuristic}, "meta": {...}}
    """
    output_path = Path(output_filepath)
    if not output_path.is_absolute() and output_path.parent == Path("."):
        output_path = REPORTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys = sorted(set(heuristic_metrics.keys()).union(set(model_metrics.keys())))
    diff = {
        metric: model_metrics.get(metric, 0.0) - heuristic_metrics.get(metric, 0.0)
        for metric in all_keys
    }

    summary = {
        "model_name": model_name,
        "heuristic": heuristic_metrics,
        model_name: model_metrics,
        "diff": diff,
        "meta": meta or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"## JSON summary generated at: {output_path}")
    return summary


def print_cli_summary(heuristic_metrics: Dict[str, float], model_metrics: Dict[str, float], model_name: str = "Model"):
    """Prints a side-by-side comparison table (baseline vs. candidate model)
    directly to the terminal.

    Args:
        heuristic_metrics: metric name -> score for the baseline ranker.
        model_metrics: metric name -> score for the candidate model.
        model_name: label used for the candidate column/header.

    Any metric present in only one of the two dicts is shown with a 0.0
    placeholder for the side that's missing it, so the two columns always
    line up; nothing is returned, this only prints.
    """

    print("\n" + "=" * 65)
    print(f"METRICS COMPARISON: Baseline vs {model_name}")
    print("=" * 65)
    print(f"{'Metric':<22} | {'Baseline':<10} | {model_name:<10} | {'Diff':<8}")
    print("-" * 65)

    all_keys = sorted(set(heuristic_metrics.keys()).union(set(model_metrics.keys())))

    for metric in all_keys:
        b_score = heuristic_metrics.get(metric, 0.0)
        m_score = model_metrics.get(metric, 0.0)
        diff = m_score - b_score

        diff_str = f"{diff:+8.4f}" if diff != 0 else "  0.0000"
        print(f"{metric:<22} | {b_score:<10.4f} | {m_score:<10.4f} | {diff_str}")

    print("=" * 65 + "\n")
