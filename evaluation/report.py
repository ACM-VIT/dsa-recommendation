import os
from typing import Dict, Any, Optional

def generate_markdown_report(
    metrics_dict: Dict[str, Any], 
    output_filename: str = "eval_report.md",
    title: str = "Offline Evaluation Report",
    output_dir: str = "reports"
) -> str:
    """Generates a Markdown report from metrics and saves it to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, output_filename)
    
    md_lines = [
        f"# {title}\n",
        "| Metric | Score |",
        "| :--- | :--- |"
    ]
    
    for metric, score in sorted(metrics_dict.items()):
        if isinstance(score, float):
            md_lines.append(f"| **{metric}** | `{score:.4f}` |")
        else:
            md_lines.append(f"| **{metric}** | `{score}` |")
            
    content = "\n".join(md_lines) + "\n"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\n##Evaluation report saved to: {filepath}")
    return content


def print_cli_summary(
    heuristic_metrics: Dict[str, float], 
    model_metrics: Dict[str, float], 
    model_name: str = "Model"
) -> None:
    """Prints a comparison table in the terminal between Baseline and Model."""
    print("\n" + "=" * 50)
    print(f"COMPARISON SUMMARY: Baseline vs {model_name}")
    print("=" * 50)
    print(f"{'Metric':<20} | {'Baseline':<10} | {model_name:<10}")
    print("-" * 50)
    
    all_keys = sorted(set(heuristic_metrics.keys()).union(model_metrics.keys()))
    for key in all_keys:
        h_val = f"{heuristic_metrics.get(key, 0.0):.4f}"
        m_val = f"{model_metrics.get(key, 0.0):.4f}"
        print(f"{key:<20} | {h_val:<10} | {m_val:<10}")
        
    print("=" * 50 + "\n")