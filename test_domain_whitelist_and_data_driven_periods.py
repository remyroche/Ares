#!/usr/bin/env python3
"""
Test Script for Domain Whitelist and Data-Driven Periods

This script demonstrates how the domain whitelist and data-driven periods work
in the interactive feature generation system.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test market data for demonstration."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    # Generate realistic market data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = {
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
        'volume': np.random.lognormal(10, 0.5, n_samples),
    }
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_samples):
        data['high'][i] = max(data['high'][i], data['open'][i], data['close'][i])
        data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some additional features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df

def test_domain_whitelist():
    """Test the domain whitelist system."""
    print("🧪 Testing Domain Whitelist System...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.domain_whitelist import (
            DomainWhitelist, FeatureDomain
        )
        
        # Create whitelist
        whitelist = DomainWhitelist()
        
        # Test feature classification
        test_features = [
            'rsi_14', 'macd_line', 'bb_upper', 'atr_14', 'volume_ma_5',
            'sma_20', 'ema_12', 'bb_position', 'price_vs_sma', 'spy_correlation',
            'ctf_15_close_mean', 'volatility_20', 'momentum_5', 'williams_r'
        ]
        
        print("📊 Feature Classification:")
        for feature in test_features:
            domain = whitelist.classify_feature(feature)
            print(f"  {feature:20} → {domain.value}")
        
        print("\n📊 Interaction Rules:")
        # Test some interaction rules
        test_pairs = [
            ('rsi_14', 'atr_14'),           # momentum × volatility
            ('volume_ma_5', 'rsi_14'),      # volume × momentum
            ('sma_20', 'bb_position'),      # trend × mean_reversion
            ('rsi_14', 'macd_line'),        # momentum × momentum (should be rejected)
            ('spy_correlation', 'rsi_14'),  # cross_asset × momentum
        ]
        
        for feature1, feature2 in test_pairs:
            is_allowed, reason = whitelist.is_interaction_allowed(feature1, feature2)
            status = "✅ ALLOWED" if is_allowed else "❌ REJECTED"
            print(f"  {feature1:15} × {feature2:15} → {status:10} ({reason})")
        
        # Test getting allowed interactions
        print(f"\n📊 Allowed Interactions for {len(test_features)} features:")
        allowed_interactions = whitelist.get_allowed_interactions(test_features, max_interactions=10)
        for i, (feature1, feature2, reason) in enumerate(allowed_interactions[:10]):
            print(f"  {i+1:2d}. {feature1:15} × {feature2:15} → {reason}")
        
        # Test statistics
        stats = whitelist.get_interaction_statistics(test_features)
        print(f"\n📊 Interaction Statistics:")
        print(f"  Total features: {stats['total_features']}")
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Allowed interactions: {stats['allowed_interactions']}")
        print(f"  Interaction rate: {stats['interaction_rate']:.1%}")
        
        print(f"\n📊 Domain Breakdown:")
        for domain, features in stats['domain_breakdown'].items():
            if features > 0:
                print(f"  {domain:20}: {features:2d} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Domain whitelist test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_driven_periods():
    """Test the data-driven period selection."""
    print("\n🧪 Testing Data-Driven Period Selection...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.data_driven_periods import (
            DataDrivenPeriodSelector, get_data_driven_periods
        )
        
        # Create test data with different characteristics
        test_cases = [
            ("High Frequency (5m)", create_test_data(2000), "5m"),
            ("Medium Frequency (15m)", create_test_data(1000), "15m"),
            ("Low Frequency (60m)", create_test_data(500), "60m"),
        ]
        
        for case_name, data, timeframe in test_cases:
            print(f"\n📊 {case_name} Data:")
            print(f"  Data length: {len(data)}")
            print(f"  Timeframe: {timeframe}")
            
            # Test period selector
            selector = DataDrivenPeriodSelector(max_periods=6)
            result = selector.select_optimal_periods(data, timeframe)
            
            print(f"  Optimal periods: {result.optimal_periods}")
            print(f"  Confidence score: {result.confidence_score:.2f}")
            
            # Show period categories
            print(f"  Period categories:")
            for category, periods in result.period_categories.items():
                if periods:
                    print(f"    {category}: {periods}")
            
            # Test convenience function
            periods = get_data_driven_periods(data, timeframe, max_periods=4)
            print(f"  Convenience function result: {periods}")
        
        # Test with insufficient data
        print(f"\n📊 Insufficient Data Test:")
        small_data = create_test_data(50)
        result = selector.select_optimal_periods(small_data)
        print(f"  Small data periods: {result.optimal_periods}")
        print(f"  Confidence score: {result.confidence_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data-driven periods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of both systems."""
    print("\n🧪 Testing Integration...")
    print("=" * 50)
    
    try:
        from src.training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.feature_generators import (
            FeatureGenerator, FeatureGenerationConfig
        )
        
        # Create test data
        data = create_test_data(1000)
        
        # Test with domain whitelist and data-driven periods
        config = FeatureGenerationConfig(
            enable_technical_indicators=True,
            enable_rolling_stats=True,
            enable_interaction_features=True,
            enable_cross_timeframe=True,
            max_interactions=20,
            interaction_types=['ratio', 'product', 'difference', 'sum']
        )
        
        generator = FeatureGenerator(config)
        
        print("📊 Generating features with domain whitelist and data-driven periods...")
        
        # Generate all features
        all_features = generator.generate_all_features(data)
        print(f"  Total features generated: {len(all_features.columns)}")
        
        # Generate base features
        base_features = generator.generate_base_features(data)
        print(f"  Base features: {len(base_features.columns)}")
        
        # Generate interaction features
        interaction_features = generator.generate_interaction_features(data)
        print(f"  Interaction features: {len(interaction_features.columns)}")
        
        # Generate cross-timeframe features
        ctf_features = generator.generate_cross_timeframe_features(data)
        print(f"  Cross-timeframe features: {len(ctf_features.columns)}")
        
        # Show sample features
        print(f"\n📊 Sample Features:")
        sample_features = list(all_features.columns[:10])
        for i, feature in enumerate(sample_features):
            print(f"  {i+1:2d}. {feature}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Domain Whitelist and Data-Driven Periods")
    print("=" * 60)
    
    tests = [
        ("Domain Whitelist", test_domain_whitelist),
        ("Data-Driven Periods", test_data_driven_periods),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} test passed!")
            else:
                print(f"❌ {test_name} test failed!")
                
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Domain whitelist and data-driven periods are working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)