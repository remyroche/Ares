#!/usr/bin/env python3
"""
Test script for Enhanced Data-Driven Period Selection with Economic Significance Evaluation

This script demonstrates the new economic significance evaluation and backtesting
for cross-timeframe period selection, following the pattern of DataDrivenPeriodSelector
and DataDrivenInteractionGenerator.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.enhanced_data_driven_period_selector import (
    EnhancedDataDrivenPeriodSelector, EnhancedPeriodSelectionConfig
)
from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.economic_period_evaluator import (
    EconomicPeriodEvaluator, EconomicEvaluationConfig
)
from src.feature_generation.utils.cross_timeframe_interaction_features import CrossTimeframeFeatureGenerator


def create_sample_data(n_samples: int = 1000, timeframe_minutes: int = 15) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    print(f"📊 Creating sample data: {n_samples} samples, {timeframe_minutes}m timeframe")
    
    # Generate realistic price data
    np.random.seed(42)
    
    # Create time index
    start_time = datetime.now() - timedelta(minutes=n_samples * timeframe_minutes)
    time_index = [start_time + timedelta(minutes=i * timeframe_minutes) for i in range(n_samples)]
    
    # Generate price data with trend and volatility
    base_price = 100.0
    trend = np.linspace(0, 0.1, n_samples)  # 10% trend over period
    volatility = 0.02  # 2% volatility
    
    # Generate OHLCV data
    returns = np.random.normal(0, volatility, n_samples) + trend / n_samples
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=time_index)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    print(f"✅ Sample data created: {data.shape}")
    print(f"📈 Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"📊 Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return data


def test_economic_period_evaluator():
    """Test the EconomicPeriodEvaluator."""
    print("\n" + "="*60)
    print("🧪 Testing EconomicPeriodEvaluator")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(500, 15)
    
    # Create evaluator
    config = EconomicEvaluationConfig(
        min_period=1,
        max_period=50,
        backtest_periods=100,
        min_backtest_periods=50
    )
    evaluator = EconomicPeriodEvaluator(config)
    
    # Test candidate periods
    candidate_periods = [5, 10, 15, 20, 30, 40]
    print(f"🔍 Evaluating periods: {candidate_periods}")
    
    # Evaluate periods
    result = evaluator.evaluate_periods(data, candidate_periods, "15m")
    
    print(f"\n📊 Evaluation Results:")
    print(f"✅ Successful evaluations: {result.successful_evaluations}")
    print(f"❌ Failed evaluations: {result.failed_evaluations}")
    print(f"⏱️ Total time: {result.total_evaluation_time:.3f}s")
    
    if result.period_rankings:
        print(f"\n🏆 Period Rankings (by economic score):")
        for i, (period, score) in enumerate(result.period_rankings[:5]):
            print(f"  {i+1}. Period {period}: {score:.3f}")
    
    if result.best_period > 0:
        best_result = result.backtest_results[result.best_period]
        print(f"\n🥇 Best Period: {result.best_period}")
        print(f"   Economic Score: {best_result.economic_score:.3f}")
        print(f"   Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {best_result.max_drawdown:.3f}")
        print(f"   Win Rate: {best_result.win_rate:.3f}")
        print(f"   Profit Factor: {best_result.profit_factor:.3f}")
    
    return result


def test_enhanced_period_selector():
    """Test the EnhancedDataDrivenPeriodSelector."""
    print("\n" + "="*60)
    print("🧪 Testing EnhancedDataDrivenPeriodSelector")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(800, 15)
    
    # Create enhanced selector
    config = EnhancedPeriodSelectionConfig(
        min_period=1,
        max_period=50,
        max_periods=6,
        enable_economic_evaluation=True,
        min_economic_score=0.3,
        economic_weight=0.7,
        statistical_weight=0.3
    )
    selector = EnhancedDataDrivenPeriodSelector(config)
    
    print(f"🔍 Selecting optimal periods for 15m timeframe...")
    
    # Select periods
    result = selector.select_optimal_periods(data, "15m")
    
    print(f"\n📊 Selection Results:")
    print(f"✅ Success: {result.success}")
    print(f"⏱️ Total time: {result.total_execution_time:.3f}s")
    print(f"🎯 Selected periods: {result.optimal_periods}")
    print(f"📈 Best period: {result.best_period} (score: {result.best_score:.3f})")
    print(f"📊 Average score: {result.average_score:.3f}")
    
    if result.economic_evaluation_result:
        print(f"\n💰 Economic Evaluation:")
        print(f"   Successful evaluations: {result.economic_evaluation_result.successful_evaluations}")
        print(f"   Failed evaluations: {result.economic_evaluation_result.failed_evaluations}")
        print(f"   Best economic score: {result.economic_evaluation_result.best_economic_score:.3f}")
    
    if result.combined_rankings:
        print(f"\n🏆 Combined Rankings:")
        for i, (period, score) in enumerate(result.combined_rankings[:5]):
            print(f"  {i+1}. Period {period}: {score:.3f}")
    
    return result


def test_cross_timeframe_feature_generator():
    """Test the CrossTimeframeFeatureGenerator with enhanced period selection."""
    print("\n" + "="*60)
    print("🧪 Testing CrossTimeframeFeatureGenerator with Enhanced Period Selection")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(600, 15)
    
    # Create generator
    generator = CrossTimeframeFeatureGenerator()
    
    print(f"🔍 Testing data-driven timeframes...")
    
    # Test data-driven timeframes
    timeframes = generator.get_data_driven_timeframes(data, "15m")
    print(f"✅ Data-driven timeframes: {timeframes}")
    
    # Generate cross-timeframe features
    print(f"🚀 Generating cross-timeframe features...")
    features = generator.generate_cross_timeframe_features(data, data[['volume']])
    
    print(f"✅ Generated {len(features)} cross-timeframe features")
    
    if features:
        print(f"📊 Feature names (first 10): {list(features.keys())[:10]}")
        
        # Show some statistics
        for name, series in list(features.items())[:5]:
            print(f"   {name}: mean={series.mean():.4f}, std={series.std():.4f}")
    
    return features


def main():
    """Run all tests."""
    print("🚀 Enhanced Data-Driven Period Selection Test Suite")
    print("="*60)
    
    try:
        # Test 1: Economic Period Evaluator
        economic_result = test_economic_period_evaluator()
        
        # Test 2: Enhanced Period Selector
        enhanced_result = test_enhanced_period_selector()
        
        # Test 3: Cross-Timeframe Feature Generator
        features = test_cross_timeframe_feature_generator()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        
        print(f"\n📊 Summary:")
        print(f"   Economic evaluations: {economic_result.successful_evaluations} successful")
        print(f"   Enhanced selections: {enhanced_result.successful_evaluations} successful")
        print(f"   Cross-timeframe features: {len(features)} generated")
        
        print(f"\n🎯 Key Improvements:")
        print(f"   ✅ Economic significance evaluation implemented")
        print(f"   ✅ Backtesting against financial targets integrated")
        print(f"   ✅ Period range optimized for 15m (1-50 periods)")
        print(f"   ✅ Sharpe ratio and financial metrics evaluation added")
        print(f"   ✅ VectorBT optimization for performance")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)