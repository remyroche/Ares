#!/usr/bin/env python3
"""
CLI Script to Run Iterative Optimization Hyperparameter Tuning

This script loads the necessary data and runs hyperparameter optimization
to improve CV, Silhouette, and DBI scores while maintaining Balance and Temporal Smoothness.

Usage:
    python3 src/training/steps/market_analysis/clusters/run_iterative_opt_tuning.py \\
        --symbol ETHUSDT \\
        --n-trials 30 \\
        --method bayesian

Arguments:
    --symbol: Trading symbol (e.g., ETHUSDT)
    --exchange: Exchange name (default: binance)
    --n-trials: Number of optimization trials (default: 30)
    --method: Optimization method - 'bayesian' or 'multiobjective' (default: bayesian)
    --output-dir: Directory to save results (default: artifacts/hyperparameter_tuning/)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.utils.tprint import tprint
from src.utils.artifact_manager import ArtifactManager
from src.training.steps.market_analysis.clusters.iterative_optimization_tuner import run_tuning_pipeline


def load_data_for_tuning(symbol: str, exchange: str = 'binance') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the required data for hyperparameter tuning.
    
    Returns:
        features, initial_labels, market_data
    """
    tprint("📥 Loading data for hyperparameter tuning...", "INFO")
    
    artifact_manager = ArtifactManager()
    
    # Load features from regime_feature_selection
    tprint("📥 Loading selected features from regime_feature_selection...", "INFO")
    artifact_manager.set_context(
        step_name="regime_feature_selection",
        symbol=symbol,
        exchange=exchange,
        datetime=None,
        information="feature_selection",
        direction="long",
        model="Analyst"
    )
    
    selected_features_df = artifact_manager.get_artifact("selected_features", artifact_type="data")
    if selected_features_df is None or selected_features_df.empty:
        tprint("❌ Failed to load selected features", "ERROR")
        return None, None, None
    
    # Extract feature names
    if 'feature' in selected_features_df.columns:
        selected_feature_names = selected_features_df['feature'].tolist()
    else:
        selected_feature_names = selected_features_df.index.tolist()
    
    tprint(f"✅ Loaded {len(selected_feature_names)} selected features", "SUCCESS")
    
    # Load HDBSCAN regime labels (original, before merging)
    tprint("📥 Loading HDBSCAN regime labels...", "INFO")
    artifact_manager.set_context(
        step_name="hdbscan_regime_discovery",
        symbol=symbol,
        exchange=exchange,
        datetime=None,
        information="regime_discovery",
        direction="long",
        model="Analyst"
    )
    
    regime_labels_df = artifact_manager.get_artifact("regime_labels", artifact_type="data")
    if regime_labels_df is None or regime_labels_df.empty:
        tprint("❌ Failed to load HDBSCAN regime labels", "ERROR")
        return None, None, None
    
    # Extract labels
    if 'regime_label' in regime_labels_df.columns:
        initial_labels = regime_labels_df['regime_label'].values
    elif 'label' in regime_labels_df.columns:
        initial_labels = regime_labels_df['label'].values
    else:
        initial_labels = regime_labels_df.iloc[:, 0].values
    
    tprint(f"✅ Loaded {len(initial_labels)} regime labels with {len(np.unique(initial_labels))} unique clusters", "SUCCESS")
    
    # Load market data with features from feature_generation
    tprint("📥 Loading market data with features...", "INFO")
    artifact_manager.set_context(
        step_name="feature_generation_feature_generation_step",
        symbol=symbol,
        exchange=exchange,
        datetime=None,
        information="feature_generation",
        direction="long",
        model="Analyst"
    )
    
    # Try to load 1h features first (to match regime timeframe)
    market_data = artifact_manager.get_artifact("generated_features_1h", artifact_type="data")
    
    if market_data is None or (hasattr(market_data, 'empty') and market_data.empty):
        tprint("⚠️ No 1h features found, loading 15m and resampling...", "WARNING")
        market_data_15m = artifact_manager.get_artifact("generated_features_15m", artifact_type="data")
        
        if market_data_15m is not None and not (hasattr(market_data_15m, 'empty') and market_data_15m.empty):
            # Resample to 1h
            if not isinstance(market_data_15m.index, pd.DatetimeIndex):
                if 'open_time' in market_data_15m.columns:
                    market_data_15m = market_data_15m.set_index('open_time')
                elif 'timestamp' in market_data_15m.columns:
                    market_data_15m = market_data_15m.set_index('timestamp')
            
            market_data = market_data_15m.resample('1H').last()
            market_data = market_data.dropna(how='all')
            tprint(f"✅ Resampled from {len(market_data_15m)} (15m) to {len(market_data)} (1h) samples", "SUCCESS")
        else:
            tprint("❌ Failed to load market data", "ERROR")
            return None, None, None
    
    # Extract selected features from market data
    missing_features = [f for f in selected_feature_names if f not in market_data.columns]
    if missing_features:
        tprint(f"⚠️ {len(missing_features)} features missing from market data", "WARNING")
        available_features = [f for f in selected_feature_names if f in market_data.columns]
        if not available_features:
            tprint("❌ No features available", "ERROR")
            return None, None, None
        selected_feature_names = available_features
    
    # Create feature matrix
    features = market_data[selected_feature_names].values
    
    # Handle NaN values
    if np.isnan(features).any():
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        features = imputer.fit_transform(features)
        tprint("🔧 Handled NaN values in features", "INFO")
    
    tprint(f"✅ Created feature matrix: {features.shape[0]} samples × {features.shape[1]} features", "SUCCESS")
    
    # Ensure features and labels have same length
    min_length = min(len(features), len(initial_labels))
    features = features[:min_length]
    initial_labels = initial_labels[:min_length]
    market_data = market_data.iloc[:min_length]
    
    tprint(f"✅ Data aligned: {len(features)} samples", "SUCCESS")
    
    return features, initial_labels, market_data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for iterative optimization to improve clustering quality"
    )
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., ETHUSDT)')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of optimization trials')
    parser.add_argument('--method', type=str, default='bayesian', 
                       choices=['bayesian', 'multiobjective'],
                       help='Optimization method')
    parser.add_argument('--output-dir', type=str, default='artifacts/hyperparameter_tuning/',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    tprint(f"🎯 Starting Iterative Optimization Hyperparameter Tuning", "INFO")
    tprint(f"📊 Symbol: {args.symbol}", "INFO")
    tprint(f"📊 Method: {args.method}", "INFO")
    tprint(f"📊 Trials: {args.n_trials}", "INFO")
    
    # Load data
    features, initial_labels, market_data = load_data_for_tuning(args.symbol, args.exchange)
    
    if features is None:
        tprint("❌ Failed to load data. Exiting.", "ERROR")
        return 1
    
    # Run tuning
    results = run_tuning_pipeline(
        features=features,
        initial_labels=initial_labels,
        market_data=market_data,
        n_trials=args.n_trials,
        method=args.method,
        output_dir=args.output_dir
    )
    
    if results is None:
        tprint("❌ Hyperparameter tuning failed", "ERROR")
        return 1
    
    # Print summary
    tprint("\n" + "="*80, "INFO")
    tprint("📊 OPTIMIZATION COMPLETE", "SUCCESS")
    tprint("="*80 + "\n", "INFO")
    
    if 'best_params' in results and 'best_metrics' in results:
        metrics = results['best_metrics']
        tprint("🏆 Best Configuration Metrics:", "SUCCESS")
        tprint(f"   • CV Score: {metrics.cv_score:.4f}", "INFO")
        tprint(f"   • Silhouette Score: {metrics.silhouette_score:.4f}", "INFO")
        tprint(f"   • DBI Score: {metrics.dbi_score:.4f}", "INFO")
        tprint(f"   • Balance Score: {metrics.balance_score:.4f}", "INFO")
        tprint(f"   • Temporal Smoothness: {metrics.temporal_smoothness:.4f}", "INFO")
        tprint(f"   • Number of Clusters: {metrics.n_clusters}", "INFO")
        
        tprint("\n📝 Next Steps:", "INFO")
        tprint(f"   1. Review the optimization report in: {args.output_dir}", "INFO")
        tprint("   2. Update OptConfig in src/training/steps/market_analysis/clusters/iterative_optimization.py", "INFO")
        tprint("   3. Apply the best_params from the results JSON file", "INFO")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

