#!/usr/bin/env python3
"""
Enhanced Sticky Finite HMM Clustering with Detailed Quality Assessment

This script runs the sticky finite HMM clustering and then uses the 
ClusterQualityAssessor to generate comprehensive quality metrics and 
detailed CSV reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import clustering components
from src.training.steps.market_analysis.sticky_finite_hmm_clustering.standalone_runner import (
    run_sticky_finite_hmm_clustering
)

# Import quality assessor
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics
)

# Import utilities
from src.utils.tprint import (
    tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_structured, tprint_timer
)

def generate_sample_market_data(years: int = 1) -> pd.DataFrame:
    """
    Generate realistic sample market data for testing.
    
    Args:
        years: Number of years of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    tprint_info(f"📊 Generating {years} year(s) of sample market data...")
    
    # Generate date range
    start_date = datetime.now() - timedelta(days=365 * years)
    dates = pd.date_range(start=start_date, periods=365 * 24 * years, freq='1h')
    
    # Generate realistic price data with regime-like behavior
    np.random.seed(42)
    base_price = 50000
    
    # Create different regime parameters
    regime_params = [
        {'vol': 0.01, 'trend': 0.0002, 'duration': 720},   # Stable upward
        {'vol': 0.03, 'trend': -0.0001, 'duration': 480}, # Volatile downward
        {'vol': 0.015, 'trend': 0.0000, 'duration': 600}, # Sideways
        {'vol': 0.025, 'trend': 0.0003, 'duration': 360}, # Trending upward
        {'vol': 0.04, 'trend': -0.0002, 'duration': 240}  # High volatility
    ]
    
    prices = [base_price]
    current_regime = 0
    regime_counter = 0
    
    for i in range(1, len(dates)):
        # Switch regimes periodically
        if regime_counter >= regime_params[current_regime]['duration']:
            current_regime = (current_regime + 1) % len(regime_params)
            regime_counter = 0
        
        params = regime_params[current_regime]
        regime_counter += 1
        
        # Generate return with regime-specific parameters
        ret = np.random.normal(params['trend'], params['vol'])
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1000))  # Ensure price stays positive
    
    prices = prices[1:]
    
    # Create OHLCV DataFrame
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,  # Use same prices for close to avoid length mismatch
        'volume': np.random.lognormal(10, 1, len(prices))
    }, index=dates[:len(prices)])
    
    tprint_success(f"✅ Generated {len(data)} samples with realistic regime behavior")
    return data

def save_detailed_quality_report(metrics: ClusterQualityMetrics, 
                                output_dir: str = "artifacts") -> str:
    """
    Save detailed quality metrics to CSV format.
    
    Args:
        metrics: ClusterQualityMetrics object
        output_dir: Output directory for CSV files
        
    Returns:
        Path to the main CSV report
    """
    tprint_info("💾 Generating detailed quality assessment CSV...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Main summary report
    summary_data = {
        'Metric': [
            'Silhouette Score',
            'Davies-Bouldin Index', 
            'Calinski-Harabasz Index',
            'Within-Regime CV',
            'Between-Regime CV',
            'Temporal Smoothness',
            'Regime Persistence',
            'Number of Regimes',
            'Noise Ratio',
            'Balance Score',
            'Overall Quality Score'
        ],
        'Value': [
            metrics.silhouette_score,
            metrics.davies_bouldin_score,
            metrics.calinski_harabasz_score,
            metrics.within_regime_cv,
            metrics.between_regime_cv,
            metrics.temporal_smoothness,
            metrics.regime_persistence,
            metrics.n_regimes,
            metrics.noise_ratio,
            metrics.balance_score,
            metrics.quality_score
        ],
        'Description': [
            'Cluster separation quality (-1 to 1, higher better)',
            'Cluster separation quality (lower better)',
            'Cluster separation quality (higher better)',
            'Within-regime feature consistency (lower better)',
            'Between-regime feature separation (higher better)',
            'Temporal stability of regimes (0 to 1, higher better)',
            'Average regime duration in periods',
            'Number of discovered regimes',
            'Ratio of noise points (lower better)',
            'Balance of cluster sizes (0 to 1, higher better)',
            'Composite quality score (0 to 1, higher better)'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = output_path / f"clustering_quality_summary_{timestamp}.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    
    # 2. Per-regime detailed metrics
    if metrics.per_regime_metrics:
        regime_data = []
        for regime_id, regime_metrics in metrics.per_regime_metrics.items():
            regime_data.append({
                'Regime_ID': regime_id,
                'Size': regime_metrics.get('size', 0),
                'Size_Percentage': regime_metrics.get('size_pct', 0),
                'Mean_Return': regime_metrics.get('mean_return', 0),
                'Volatility': regime_metrics.get('volatility', 0),
                'Sharpe_Ratio': regime_metrics.get('sharpe', 0),
                'Max_Drawdown': regime_metrics.get('max_drawdown', 0),
                'Win_Rate': regime_metrics.get('win_rate', 0),
                'Regime_Type': regime_metrics.get('regime_type', 'unknown'),
                'Duration_Mean': regime_metrics.get('duration_mean', 0),
                'Duration_Std': regime_metrics.get('duration_std', 0)
            })
        
        regime_df = pd.DataFrame(regime_data)
        regime_csv_path = output_path / f"regime_detailed_metrics_{timestamp}.csv"
        regime_df.to_csv(regime_csv_path, index=False)
    
    # 3. Economic validation metrics
    if metrics.economic_validation:
        econ_data = {
            'Economic_Metric': [
                'Portfolio Return',
                'Portfolio Sharpe Ratio',
                'Max Drawdown',
                'Volatility',
                'Hit Rate',
                'Profit Factor',
                'Average Trade Return',
                'Target Return Achievement'
            ],
            'Value': [
                metrics.economic_validation.get('portfolio_return', 0),
                metrics.economic_validation.get('portfolio_sharpe', 0),
                metrics.economic_validation.get('max_drawdown', 0),
                metrics.economic_validation.get('portfolio_volatility', 0),
                metrics.economic_validation.get('hit_rate', 0),
                metrics.economic_validation.get('profit_factor', 0),
                metrics.economic_validation.get('avg_trade_return', 0),
                metrics.economic_validation.get('target_return_achievement', 0)
            ],
            'Benchmark': [
                'Higher better',
                'Higher better',
                'Lower better', 
                'Lower better',
                'Higher better',
                'Higher better',
                'Higher better',
                'Higher better'
            ]
        }
        
        econ_df = pd.DataFrame(econ_data)
        econ_csv_path = output_path / f"economic_validation_{timestamp}.csv"
        econ_df.to_csv(econ_csv_path, index=False)
    
    # 4. Temporal analysis metrics
    temporal_data = {
        'Temporal_Metric': [
            'Temporal Smoothness',
            'Temporal Smoothness (Raw)',
            'Flip-Flop Ratio',
            'Regime Persistence',
            'Average Duration',
            'Duration Std Dev',
            'Min Duration',
            'Max Duration'
        ],
        'Value': [
            metrics.temporal_smoothness,
            metrics.temporal_smoothness_raw,
            metrics.flip_flop_ratio,
            metrics.regime_persistence,
            metrics.regime_duration_distribution.get('mean_duration', 0),
            metrics.regime_duration_distribution.get('std_duration', 0),
            metrics.regime_duration_distribution.get('min_duration', 0),
            metrics.regime_duration_distribution.get('max_duration', 0)
        ],
        'Interpretation': [
            'Higher = more stable regimes',
            'Higher = more stable (no penalty)',
            'Lower = fewer rapid switches',
            'Higher = longer lasting regimes',
            'Average regime length in periods',
            'Variability in regime duration',
            'Shortest regime observed',
            'Longest regime observed'
        ]
    }
    
    temporal_df = pd.DataFrame(temporal_data)
    temporal_csv_path = output_path / f"temporal_analysis_{timestamp}.csv"
    temporal_df.to_csv(temporal_csv_path, index=False)
    
    tprint_success(f"✅ Detailed CSV reports saved to {output_path}")
    tprint_info(f"   📄 Summary: {summary_csv_path.name}")
    if metrics.per_regime_metrics:
        tprint_info(f"   📄 Regimes: {regime_csv_path.name}")
    if metrics.economic_validation:
        tprint_info(f"   📄 Economic: {econ_csv_path.name}")
    tprint_info(f"   📄 Temporal: {temporal_csv_path.name}")
    
    return str(summary_csv_path)

def run_enhanced_clustering_with_quality_assessment():
    """
    Main function to run enhanced clustering with detailed quality assessment.
    """
    tprint_info("🚀 Starting Enhanced Sticky Finite HMM Clustering with Quality Assessment")
    tprint_info("=" * 80)
    
    with tprint_timer("Complete Enhanced Clustering Pipeline", level="INFO"):
        # Step 1: Generate/load market data
        tprint_info("📊 Step 1: Preparing market data...")
        market_data = generate_sample_market_data(years=1)
        
        # Step 2: Run sticky finite HMM clustering
        tprint_info("🔬 Step 2: Running Sticky Finite HMM clustering...")
        clustering_results = run_sticky_finite_hmm_clustering(
            market_data=market_data,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            K=5,
            n_mixtures=1,
            base_alpha=0.5,
            kappa=10.0,
            num_iters=800,
            min_features=50,
            max_features=100,
            enable_pca=True,
            pca_components=15,
            save_results=False  # We'll save our own detailed reports
        )
        
        # Step 3: Extract clustering results for quality assessment
        tprint_info("🔍 Step 3: Preparing data for quality assessment...")
        
        cluster_labels = clustering_results['cluster_labels']
        feature_matrix = clustering_results.get('feature_matrix')
        
        if feature_matrix is None:
            tprint_warning("⚠️ No feature matrix found, using basic price features")
            # Create basic features from OHLCV data
            feature_matrix = pd.DataFrame({
                'returns': market_data['close'].pct_change(),
                'volume': market_data['volume'],
                'high_low_ratio': market_data['high'] / market_data['low'],
                'open_close_ratio': market_data['open'] / market_data['close'],
                'price_change': market_data['close'] - market_data['open']
            }).fillna(0)
        
        # Ensure alignment
        min_length = min(len(cluster_labels), len(feature_matrix))
        cluster_labels = cluster_labels[:min_length]
        feature_matrix = feature_matrix.iloc[:min_length].reset_index(drop=True)
        timestamps = market_data.index[:min_length]
        
        # Calculate forward returns for economic validation
        forward_returns = market_data['close'].pct_change().shift(-1).iloc[:min_length]
        
        tprint_success(f"✅ Data prepared: {len(cluster_labels)} samples, {feature_matrix.shape[1]} features")
        
        # Step 4: Run comprehensive quality assessment
        tprint_info("📈 Step 4: Running comprehensive quality assessment...")
        
        # Initialize quality assessor
        quality_assessor = ClusterQualityAssessor(
            artifact_manager=None,
            enable_hardware_optimization=True,
            enable_vectorization=True
        )
        
        # Assess quality
        quality_metrics = quality_assessor.assess_quality(
            regime_labels=cluster_labels,
            feature_data=feature_matrix,
            forward_returns=forward_returns,
            timestamps=timestamps,
            min_regime_size=10,
            temporal_sensitivity_mode="standard"
        )
        
        # Step 5: Generate detailed CSV reports
        tprint_info("📄 Step 5: Generating detailed CSV reports...")
        main_report_path = save_detailed_quality_report(quality_metrics)
        
        # Step 6: Display comprehensive results
        tprint_info("=" * 80)
        tprint_info("📊 ENHANCED CLUSTERING RESULTS SUMMARY")
        tprint_info("=" * 80)
        
        tprint_structured({
            "Clustering Results": {
                "Regimes Discovered": clustering_results['n_clusters'],
                "Final ELBO": clustering_results.get('final_elbo', 'N/A'),
                "Processing Time": f"{clustering_results.get('processing_time', 'N/A')}s"
            },
            "Quality Assessment": {
                "Silhouette Score": f"{quality_metrics.silhouette_score:.4f}" if quality_metrics.silhouette_score else "N/A",
                "Davies-Bouldin Index": f"{quality_metrics.davies_bouldin_score:.4f}" if quality_metrics.davies_bouldin_score else "N/A",
                "Calinski-Harabasz Index": f"{quality_metrics.calinski_harabasz_score:.2f}" if quality_metrics.calinski_harabasz_score else "N/A",
                "Temporal Smoothness": f"{quality_metrics.temporal_smoothness:.4f}" if quality_metrics.temporal_smoothness else "N/A",
                "Regime Persistence": f"{quality_metrics.regime_persistence:.2f}" if quality_metrics.regime_persistence else "N/A",
                "Overall Quality Score": f"{quality_metrics.quality_score:.4f}" if quality_metrics.quality_score else "N/A"
            },
            "Economic Validation": {
                "Portfolio Return": f"{quality_metrics.economic_validation.get('portfolio_return', 0):.2%}" if quality_metrics.economic_validation else "N/A",
                "Portfolio Sharpe": f"{quality_metrics.economic_validation.get('portfolio_sharpe', 0):.3f}" if quality_metrics.economic_validation else "N/A",
                "Max Drawdown": f"{quality_metrics.economic_validation.get('max_drawdown', 0):.2%}" if quality_metrics.economic_validation else "N/A",
                "Hit Rate": f"{quality_metrics.economic_validation.get('hit_rate', 0):.2%}" if quality_metrics.economic_validation else "N/A"
            }
        }, level="INFO")
        
        tprint_info("=" * 80)
        tprint_success(f"✅ Enhanced clustering complete! Main report: {main_report_path}")
        tprint_info("=" * 80)
        
        return {
            'clustering_results': clustering_results,
            'quality_metrics': quality_metrics.to_dict(),
            'main_report_path': main_report_path
        }

if __name__ == "__main__":
    try:
        results = run_enhanced_clustering_with_quality_assessment()
        tprint_success("🎉 Pipeline completed successfully!")
    except Exception as e:
        tprint_error(f"❌ Pipeline failed: {e}")
        raise
