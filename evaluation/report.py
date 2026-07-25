import os
from typing import Dict, Any

def generate_markdown_report(
    metrics_dict: Dict[str, Any], 
    output_filepath: str = "eval_report.md",
    title: str = "Offline Evaluation Report"
) -> str:
    """Generates a clean Markdown file summarizing evaluation results."""
    
    md_lines = [
        f"##{title}##\n",
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
    with open(output_filepath, "w",encoding="utf-8") as f:
        f.write(content)
        
    print(f"## Evaluation report generated at: {output_filepath}")
    return content


def print_cli_summary(heuristic_metrics: Dict[str, float], model_metrics: Dict[str, float], model_name: str = "Model"):
    """Prints a clear side-by-side comparison table directly in the terminal."""
    
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