"""
Complete Example: Risk Mitigation Auto-Tuning

This script demonstrates how to use the risk mitigation tuner to optimize
clustering stability and quality.

Note: CV Enhancement tuner has been removed as AdaptiveWeightScheduler
      was not integrated into the optimization loop.

Usage:
    python example_risk_cv_tuning.py --symbol ETHUSDT --trials 30

Author: AI Assistant
Date: 2025-10-28
"""

import numpy as np
import pandas as pd
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.utils.tprint import tprint
from src.training.steps.market_analysis.clusters.risk_mitigation_tuner import (
    run_risk_mitigation_tuning,
    RiskMitigationTuner
)


def load_data_for_tuning(symbol: str):
    """
    Load features, labels, and market data for tuning.
    
    In production, this would load from actual regime_feature_selection
    and HDBSCAN clustering artifacts.
    """
    tprint(f"📊 Loading data for {symbol}...", "INFO")
    
    # TODO: Replace with actual data loading
    # features = load_artifact('regime_feature_selection', 'selected_features')
    # labels = load_artifact('regime_clustering', 'initial_labels')
    # market_data = load_artifact('feature_generation', 'market_data')
    
    # For demo: Create synthetic data
    n_samples = 500
    n_features = 25
    
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 7, n_samples)  # 7 initial clusters
    
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1h'),
        'close': 2000 + np.cumsum(np.random.randn(n_samples) * 10),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    tprint(f"✅ Loaded: {n_samples} samples, {n_features} features", "SUCCESS")
    
    return features, labels, market_data


def run_risk_tuning(features, labels, market_data, n_trials: int = 30):
    """
    Run risk mitigation tuner.
    
    Args:
        features: Feature matrix
        labels: Initial cluster labels
        market_data: Market data DataFrame
        n_trials: Number of trials
        
    Returns:
        Risk mitigation tuning results
    """
    tprint("🚀 Starting risk mitigation tuning...", "INFO")
    
    start_time = time.time()
    
    risk_results = run_risk_mitigation_tuning(
        features=features,
        initial_labels=labels,
        market_data=market_data,
        n_trials=n_trials
    )
    
    elapsed = time.time() - start_time
    
    tprint(f"✅ Tuning completed in {elapsed/60:.1f} minutes", "SUCCESS")
    
    return risk_results


def print_results_summary(risk_results):
    """Print comprehensive results summary."""
    
    tprint("\n" + "="*80, "INFO")
    tprint("📊 TUNING RESULTS SUMMARY", "INFO")
    tprint("="*80 + "\n", "INFO")
    
    # Risk Mitigation Results
    if risk_results:
        tprint("🛡️ RISK MITIGATION TUNING", "INFO")
        tprint("-" * 80, "INFO")
        
        risk_params = risk_results['best_params']
        risk_metrics = risk_results['best_metrics']
        
        tprint(f"Best Composite Score: {risk_results['best_score']:.4f}", "SUCCESS")
        tprint(f"Stability Score: {risk_metrics.get_stability_score():.4f}", "SUCCESS")
        tprint(f"\nClustering Quality:", "INFO")
        tprint(f"  CV Score:         {risk_metrics.cv_score:.3f}", "INFO")
        tprint(f"  Silhouette:       {risk_metrics.silhouette_score:.3f}", "INFO")
        tprint(f"  DBI:              {risk_metrics.dbi_score:.3f}", "INFO")
        tprint(f"  Clusters:         {risk_metrics.n_clusters}", "INFO")
        
        tprint(f"\nStability Metrics:", "INFO")
        tprint(f"  Instability Events:     {risk_metrics.instability_events}", "INFO")
        tprint(f"  Total Reassignments:    {risk_metrics.total_reassignments}", "INFO")
        tprint(f"  Convergence Rounds:     {risk_metrics.convergence_rounds}", "INFO")
        tprint(f"  Converged:              {risk_metrics.converged}", "INFO")
        
        tprint(f"\nBest Parameters:", "INFO")
        tprint(f"  Stability Threshold:    {risk_params['min_stability_score']:.3f}", "INFO")
        tprint(f"  Max Splits/Round:       {risk_params['max_splits_per_round']}", "INFO")
        tprint(f"  Local Churn Cap:        {risk_params['local_churn_cap']:.3f}", "INFO")
        tprint(f"  Global Churn Cap:       {risk_params['global_churn_cap']:.3f}", "INFO")
        tprint(f"  Convergence Threshold:  {risk_params['convergence_threshold']:.4f}", "INFO")
        
        tprint(f"\n📁 Results saved to:", "INFO")
        tprint(f"   artifacts/hyperparameter_tuning/risk_mitigation_*", "INFO")
    else:
        tprint("❌ Risk Mitigation tuning failed", "ERROR")
    
    tprint("\n" + "="*80 + "\n", "INFO")


def save_best_params_to_config(symbol: str, risk_results, config_path: str = None):
    """
    Save best parameters to configuration file.
    
    Args:
        symbol: Trading symbol
        risk_results: Risk mitigation tuning results
        cv_results: CV enhancement tuning results
        config_path: Path to config file (default: config/regime_clustering_config.yaml)
    """
    if config_path is None:
        config_path = Path('config') / 'regime_clustering_config.yaml'
    
    tprint(f"\n💾 Saving best parameters to {config_path}...", "INFO")
    
    # TODO: Implement config file update
    # For now, just print what would be saved
    
    tprint("\n📝 Add these to your config file:", "INFO")
    tprint("\n```yaml", "INFO")
    tprint("# Risk Mitigation Parameters (from tuning)", "INFO")
    tprint("risk_mitigation:", "INFO")
    if risk_results:
        for key, value in risk_results['best_params'].items():
            if isinstance(value, float):
                tprint(f"  {key}: {value:.4f}", "INFO")
            else:
                tprint(f"  {key}: {value}", "INFO")
    
    tprint("```\n", "INFO")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run Risk Mitigation & CV Enhancement Auto-Tuning'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='ETHUSDT',
        help='Trading symbol to optimize (default: ETHUSDT)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Number of trials per tuner (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/hyperparameter_tuning/',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    tprint("\n" + "="*80, "INFO")
    tprint("🎯 RISK MITIGATION AUTO-TUNING", "INFO")
    tprint("="*80, "INFO")
    tprint(f"\nSymbol: {args.symbol}", "INFO")
    tprint(f"Trials: {args.trials}", "INFO")
    tprint(f"Output directory: {args.output_dir}\n", "INFO")
    
    # Step 1: Load data
    features, labels, market_data = load_data_for_tuning(args.symbol)
    
    # Step 2: Run tuning
    tprint("🛡️ Running Risk Mitigation tuning...", "INFO")
    risk_results = run_risk_tuning(
        features, labels, market_data, 
        n_trials=args.trials
    )
    
    # Step 3: Print results
    print_results_summary(risk_results)
    
    # Step 4: Save to config
    save_best_params_to_config(args.symbol, risk_results)
    
    tprint("✅ Tuning completed successfully!", "SUCCESS")
    tprint("\n💡 Next steps:", "INFO")
    tprint("   1. Review the tuning reports in artifacts/hyperparameter_tuning/", "INFO")
    tprint("   2. Copy the best parameters to your config file", "INFO")
    tprint("   3. Run regime_clustering with the new parameters", "INFO")
    tprint("   4. Compare results with baseline\n", "INFO")


if __name__ == "__main__":
    main()
