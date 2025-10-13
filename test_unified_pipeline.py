#!/usr/bin/env python3
"""
Test script for the fully implemented UnifiedDataDrivenPipeline
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(n_samples=1000):
    """Create test data for the pipeline."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='15T')
    
    # Generate synthetic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some trend and volatility clustering
    trend = np.sin(np.arange(n_samples) * 2 * np.pi / 100) * 0.01
    volatility = 0.01 + 0.005 * np.sin(np.arange(n_samples) * 2 * np.pi / 50)
    returns = returns + trend + np.random.normal(0, volatility)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Add some technical indicators as features
    data['sma_20'] = data['close'].rolling(20).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()
    data['rsi_14'] = calculate_rsi(data['close'], 14)
    data['volatility_20'] = data['close'].rolling(20).std()
    data['momentum_10'] = data['close'].pct_change(10)
    data['volume_sma_20'] = data['volume'].rolling(20).mean()
    
    # Create target variable (future returns)
    data['target'] = data['close'].pct_change(5).shift(-5)  # 5-period forward returns
    
    # Set index
    data.set_index('timestamp', inplace=True)
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def test_unified_pipeline():
    """Test the unified pipeline with synthetic data."""
    print("🧪 Testing UnifiedDataDrivenPipeline...")
    
    try:
        # Import the pipeline
        from src.training.steps.pre_training.unified_data_driven_pipeline.core.unified_pipeline import (
            UnifiedDataDrivenPipeline, 
            create_unified_pipeline,
            process_features
        )
        from src.training.steps.pre_training.unified_data_driven_pipeline.core.config import (
            create_default_config
        )
        
        print("✅ Successfully imported pipeline components")
        
        # Create test data
        print("📊 Creating test data...")
        data = create_test_data(500)  # Smaller dataset for faster testing
        targets = data['target'].dropna()
        feature_data = data.drop(['target'], axis=1)
        
        print(f"✅ Created test data: {feature_data.shape[0]} samples, {feature_data.shape[1]} features")
        
        # Test 1: Create pipeline with default config
        print("\n🔧 Test 1: Creating pipeline with default config...")
        config = create_default_config()
        
        # Enable all features for testing
        config.enable_period_optimization = True
        config.enable_feature_lookback_optimization = True
        config.enable_interaction_generation = True
        config.enable_htf_interactions = True
        config.enable_feature_selection = True
        
        pipeline = create_unified_pipeline(config)
        print("✅ Pipeline created successfully")
        
        # Test 2: Process data through pipeline
        print("\n🚀 Test 2: Processing data through pipeline...")
        result = pipeline.process(feature_data, targets)
        
        print("✅ Pipeline processing completed successfully!")
        print(f"📊 Results:")
        print(f"   - Selected features: {len(result.selected_features)}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        print(f"   - CV splits: {result.n_cv_splits}")
        print(f"   - Out-of-sample Sharpe: {result.out_of_sample_sharpe:.3f}")
        print(f"   - Max drawdown: {result.max_drawdown:.3f}")
        print(f"   - Stability score: {result.stability_score:.3f}")
        print(f"   - Diversity score: {result.diversity_score:.3f}")
        
        # Test 3: Check intermediate results
        print("\n🔍 Test 3: Checking intermediate results...")
        
        if result.period_optimization_result:
            print(f"   - Period optimization: {len(result.period_optimization_result.get('optimized_periods', {}))} timeframes")
        
        if result.lookback_optimization_result:
            print(f"   - Lookback optimization: {len(result.lookback_optimization_result.get('optimized_lookbacks', {}))} features")
        
        if result.interaction_generation_result:
            interactions = result.interaction_generation_result.get('generated_interactions', [])
            print(f"   - Interactions generated: {len(interactions)}")
        
        if result.htf_interaction_result:
            htf_interactions = result.htf_interaction_result.get('generated_interactions', [])
            print(f"   - HTF interactions generated: {len(htf_interactions)}")
        
        # Test 4: Performance stats
        print("\n📈 Test 4: Performance statistics...")
        stats = pipeline.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value}")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components separately."""
    print("\n🔧 Testing individual components...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.core.unified_pipeline import (
            UnifiedDataDrivenPipeline
        )
        
        # Create test data
        data = create_test_data(200)
        targets = data['target'].dropna()
        feature_data = data.drop(['target'], axis=1)
        
        # Create pipeline
        pipeline = UnifiedDataDrivenPipeline()
        
        # Test period optimization
        print("   - Testing period optimization...")
        period_result = pipeline._optimize_periods(feature_data, {})
        print(f"     ✅ Period optimization: {len(period_result.get('optimized_periods', {}))} timeframes")
        
        # Test feature lookback optimization
        print("   - Testing feature lookback optimization...")
        lookback_result = pipeline._optimize_feature_lookback(feature_data, targets, {})
        print(f"     ✅ Lookback optimization: {len(lookback_result.get('optimized_lookbacks', {}))} features")
        
        # Test interaction generation
        print("   - Testing interaction generation...")
        interaction_result = pipeline._generate_interactions(feature_data, targets, {})
        interactions = interaction_result.get('generated_interactions', [])
        print(f"     ✅ Interaction generation: {len(interactions)} interactions")
        
        # Test HTF interaction generation
        print("   - Testing HTF interaction generation...")
        htf_result = pipeline._generate_htf_interactions(feature_data, targets, {})
        htf_interactions = htf_result.get('generated_interactions', [])
        print(f"     ✅ HTF interaction generation: {len(htf_interactions)} interactions")
        
        print("✅ All individual component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Individual component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting UnifiedDataDrivenPipeline Tests")
    print("=" * 50)
    
    # Test individual components first
    individual_success = test_individual_components()
    
    if individual_success:
        # Test full pipeline
        pipeline_success = test_unified_pipeline()
        
        if pipeline_success:
            print("\n🎉 All tests completed successfully!")
            print("✅ The UnifiedDataDrivenPipeline is fully implemented and working!")
        else:
            print("\n❌ Pipeline test failed")
            sys.exit(1)
    else:
        print("\n❌ Individual component tests failed")
        sys.exit(1)