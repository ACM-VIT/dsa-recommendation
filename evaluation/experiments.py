from evaluation.evaluator import OfflineEvaluator
from evaluation.report import generate_markdown_report, print_cli_summary

def run_experiment(heuristic_results: list, model_results: list, model_name: str = "LightGBM"):
    """Runs batch evaluation for baseline and candidate models."""
    evaluator = OfflineEvaluator(k_values=[5, 10])
    
    print(" Evaluating Baseline (Heuristic)...")
    heuristic_metrics = evaluator.evaluate_batch(heuristic_results)
    
    print(f" Evaluating Candidate ({model_name})...")
    candidate_metrics = evaluator.evaluate_batch(model_results)
    
    # Print comparison in terminal
    print_cli_summary(
        heuristic_metrics=heuristic_metrics, 
        model_metrics=candidate_metrics, 
        model_name=model_name
    )
    
    # Generate report in the reports/ directory
    generate_markdown_report(
        metrics_dict=candidate_metrics, 
        output_filename=f"{model_name.lower()}_eval_report.md", 
        title=f"{model_name} Evaluation Report"
    )

if __name__ == "__main__":
    # Test Data
    mock_heuristic_data = [
        {
            "recommended": ["item1", "item2", "item3", "item4", "item5"],
            "relevant": {"item2": 1.0, "item4": 2.0},
            "item_pools": {"item1": "popular", "item2": "popular"}
        }
    ]
    
    mock_lgbm_data = [
        {
            "recommended": ["item2", "item4", "item1", "item3", "item5"],
            "relevant": {"item2": 1.0, "item4": 2.0},
            "item_pools": {"item2": "vector", "item4": "heuristic"}
        }
    ]
    
    run_experiment(
        heuristic_results=mock_heuristic_data, 
        model_results=mock_lgbm_data, 
        model_name="LightGBM"
    )