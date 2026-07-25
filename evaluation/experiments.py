from evaluation.evaluator import OfflineEvaluator
from evaluation.report import generate_markdown_report, print_cli_summary

def run_experiment(heuristic_results: list, model_results: list, model_name: str = "LightGBM"):
    """Runs batch evaluation for baseline and candidate models, prints CLI comparison, 
    and saves markdown report."""
    
    evaluator = OfflineEvaluator(k_values=[5, 10])
    
    print("⏳ Evaluating Baseline (Heuristic)...")
    heuristic_metrics = evaluator.evaluate_batch(heuristic_results)
    
    print(f"⏳ Evaluating Candidate ({model_name})...")
    candidate_metrics = evaluator.evaluate_batch(model_results)
    
    # Print formatted comparison in Terminal
    print_cli_summary(
        heuristic_metrics=heuristic_metrics, 
        model_metrics=candidate_metrics, 
        model_name=model_name
    )
    
    # Save markdown report to disk
    report_path = f"{model_name.lower()}_eval_report.md"
    generate_markdown_report(candidate_metrics, output_filepath=report_path, title=f"{model_name} Evaluation Report")


if __name__ == "__main__":
    # Mock user predictions to test the full pipeline locally
    mock_heuristic_data = [
        {
            "recommended": ["item1", "item2", "item3", "item4", "item5"],
            "relevant": ["item2", "item4"],
            "item_pools": {"item1": "popular", "item2": "popular", "item3": "popular"}
        },
        {
            "recommended": ["item5", "item6", "item7", "item8", "item9"],
            "relevant": ["item7"],
            "item_pools": {"item5": "popular", "item6": "popular"}
        }
    ]
    
    mock_lgbm_data = [
        {
            "recommended": ["item2", "item4", "item1", "item3", "item5"],
            "relevant": ["item2", "item4"],
            "item_pools": {"item2": "vector", "item4": "heuristic", "item1": "vector"}
        },
        {
            "recommended": ["item7", "item5", "item6", "item8", "item9"],
            "relevant": ["item7"],
            "item_pools": {"item7": "vector", "item5": "heuristic"}
        }
    ]
    
    run_experiment(
        heuristic_results=mock_heuristic_data, 
        model_results=mock_lgbm_data, 
        model_name="LightGBM"
    )