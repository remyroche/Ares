"""
Optimized Triple Barrier Labeling Example

This example demonstrates the optimized triple barrier labeling system
with Optuna integration, regime-specific parameters, and comprehensive reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .core import TripleBarrierLabeler, TripleBarrierConfig, LabelingMethod
from .optimized_labeler import OptimizedTripleBarrierLabeler
from .regime_aware import RegimeAwareTripleBarrierLabeler, RegimeAwareConfig
from .quality_assessment import LabelQualityAssessor
from .cross_validation import LabelCrossValidator
from .utils import LabelingUtils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(num_rows: int = 5000) -> pd.DataFrame:
    """Generate sample market data for demonstration."""
    logger.info(f"📊 Generating {num_rows} rows of sample market data")
    
    # Create realistic price data with trends and volatility
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='1min')
    
    # Generate price series with regime changes
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, num_rows):
        # Create regime changes every 1000 periods
        regime = (i // 1000) % 3
        
        if regime == 0:  # Bull market
            drift = 0.0001
            volatility = 0.01
        elif regime == 1:  # Bear market
            drift = -0.0001
            volatility = 0.008
        else:  # Sideways market
            drift = 0.00005
            volatility = 0.005
        
        # Generate price change
        price_change = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Create OHLC data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, num_rows)
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    logger.info(f"✅ Generated sample data: {data.shape}")
    return data

def generate_hmm_regime_data(data: pd.DataFrame) -> pd.DataFrame:
    """Generate HMM regime data for demonstration."""
    logger.info("🧠 Generating HMM regime data")
    
    # Create regime changes based on price trends
    returns = data['close'].pct_change().fillna(0)
    rolling_returns = returns.rolling(window=100).mean()
    
    regimes = []
    for i, ret in enumerate(rolling_returns):
        if i < 100:
            regimes.append('bull')  # Default
        elif ret > 0.0001:
            regimes.append('bull')
        elif ret < -0.0001:
            regimes.append('bear')
        else:
            regimes.append('sideways')
    
    regime_data = pd.DataFrame({
        'regime': regimes
    }, index=data.index)
    
    logger.info(f"✅ Generated regime data: {regime_data['regime'].value_counts().to_dict()}")
    return regime_data

def run_optimized_example():
    """Run the optimized triple barrier labeling example."""
    logger.info("🚀 Starting Optimized Triple Barrier Labeling Example")
    
    # 1. Generate sample data
    market_data = generate_sample_data(5000)
    regime_data = generate_hmm_regime_data(market_data)
    
    # 2. Initialize optimized labeler
    logger.info("\n--- Initializing Optimized Triple Barrier Labeler ---")
    optimized_labeler = OptimizedTripleBarrierLabeler()
    
    # 3. Optimize regime parameters
    logger.info("\n--- Optimizing Regime Parameters ---")
    optimization_results = optimized_labeler.optimize_regime_parameters(
        data=market_data,
        regime_data=regime_data,
        n_trials=50  # Reduced for demo
    )
    
    # 4. Print optimization report
    logger.info("\n--- Optimization Results ---")
    optimized_labeler.print_optimization_report()
    
    # 5. Create optimized labels
    logger.info("\n--- Creating Optimized Labels ---")
    optimized_labels = optimized_labeler.create_optimized_labels(
        data=market_data,
        regime_data=regime_data
    )
    
    logger.info(f"✅ Generated optimized labels: {optimized_labels.shape}")
    logger.info(f"📊 Label distribution: {optimized_labels['label'].value_counts().to_dict()}")
    
    # 6. Quality assessment
    logger.info("\n--- Quality Assessment ---")
    quality_assessor = LabelQualityAssessor()
    quality_metrics = quality_assessor.assess_quality(
        labels_df=optimized_labels,
        original_data=market_data,
        regime_column='regime'
    )
    
    logger.info(f"📈 Quality metrics:")
    logger.info(f"   Overall quality: {quality_metrics.overall_quality:.3f}")
    logger.info(f"   Label distribution: {quality_metrics.label_distribution}")
    logger.info(f"   Regime balance: {quality_metrics.regime_balance}")
    logger.info(f"   Temporal consistency: {quality_metrics.temporal_consistency:.3f}")
    logger.info(f"   Profit consistency: {quality_metrics.profit_consistency:.3f}")
    
    # 7. Cross-validation
    logger.info("\n--- Cross-Validation ---")
    cv_validator = LabelCrossValidator(n_splits=3, purged_pct=0.05)
    
    # Use close price as feature for CV
    features = market_data[['close']]
    labels = optimized_labels['label']
    
    cv_results = cv_validator.validate_labels(features, labels)
    logger.info(f"📊 CV Results:")
    logger.info(f"   Mean score: {cv_results['mean_score']:.3f}")
    logger.info(f"   Std score: {cv_results['std_score']:.3f}")
    logger.info(f"   Validation passed: {cv_results['validation_passed']}")
    
    # 8. Save results
    logger.info("\n--- Saving Results ---")
    utils = LabelingUtils()
    
    # Save optimized labels
    utils.save_labels(optimized_labels, "optimized_triple_barrier_labels.parquet")
    
    # Save optimization report
    optimization_report = optimized_labeler.get_optimization_report()
    utils.serializer.save(optimization_report, "optimization_report.json")
    
    logger.info("✅ Optimized triple barrier labeling example completed successfully!")
    
    return {
        'optimized_labels': optimized_labels,
        'optimization_results': optimization_results,
        'quality_metrics': quality_metrics,
        'cv_results': cv_results
    }

def run_comparison_example():
    """Run a comparison between standard and optimized labeling."""
    logger.info("\n🔄 Running Comparison: Standard vs Optimized Labeling")
    
    # Generate data
    market_data = generate_sample_data(2000)
    regime_data = generate_hmm_regime_data(market_data)
    
    # Standard labeling
    logger.info("\n--- Standard Triple Barrier Labeling ---")
    standard_config = TripleBarrierConfig(
        pt_mult=0.01,  # 1%
        sl_mult=0.005,  # 0.5%
        max_holding_period=100,
        transaction_cost=0.0008
    )
    
    standard_labeler = TripleBarrierLabeler(standard_config)
    standard_labels = standard_labeler.create_labels(
        data=market_data,
        method=LabelingMethod.TRIPLE_BARRIER
    )
    
    # Optimized labeling
    logger.info("\n--- Optimized Triple Barrier Labeling ---")
    optimized_labeler = OptimizedTripleBarrierLabeler()
    optimized_labeler.optimize_regime_parameters(market_data, regime_data, n_trials=30)
    optimized_labels = optimized_labeler.create_optimized_labels(market_data, regime_data)
    
    # Compare results
    logger.info("\n--- Comparison Results ---")
    
    # Standard metrics
    standard_metrics = {
        'total_trades': (standard_labels['label'] != 0).sum(),
        'win_rate': (standard_labels['label'] > 0).mean(),
        'avg_profit': standard_labels['profit_pct'].mean(),
        'sharpe_ratio': standard_labels['profit_pct'].mean() / standard_labels['profit_pct'].std() if standard_labels['profit_pct'].std() > 0 else 0
    }
    
    # Optimized metrics
    optimized_metrics = {
        'total_trades': (optimized_labels['label'] != 0).sum(),
        'win_rate': (optimized_labels['label'] > 0).mean(),
        'avg_profit': optimized_labels['profit_pct'].mean(),
        'sharpe_ratio': optimized_labels['profit_pct'].mean() / optimized_labels['profit_pct'].std() if optimized_labels['profit_pct'].std() > 0 else 0
    }
    
    logger.info("📊 Standard Labeling:")
    for metric, value in standard_metrics.items():
        logger.info(f"   {metric}: {value:.4f}")
    
    logger.info("📊 Optimized Labeling:")
    for metric, value in optimized_metrics.items():
        logger.info(f"   {metric}: {value:.4f}")
    
    # Improvement analysis
    logger.info("📈 Improvements:")
    for metric in standard_metrics:
        improvement = ((optimized_metrics[metric] - standard_metrics[metric]) / standard_metrics[metric]) * 100
        logger.info(f"   {metric}: {improvement:+.1f}%")
    
    return {
        'standard_labels': standard_labels,
        'optimized_labels': optimized_labels,
        'standard_metrics': standard_metrics,
        'optimized_metrics': optimized_metrics
    }

if __name__ == "__main__":
    # Run the main example
    results = run_optimized_example()
    
    # Run comparison
    comparison = run_comparison_example()
    
    logger.info("\n🎉 All examples completed successfully!")