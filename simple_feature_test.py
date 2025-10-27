"""
Simple Feature Enhancement Test.

This test validates the enhanced feature generation without requiring
the full project structure.
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

# Add src to path
sys.path.append('/workspace/src')

# Mock the tprint functions to avoid dependency issues
class MockTPrint:
    def info(self, msg): print(f"INFO: {msg}")
    def success(self, msg): print(f"SUCCESS: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

# Mock the tprint module
sys.modules['src.utils.tprint'] = type('MockModule', (), {
    'tprint': print,
    'tprint_info': MockTPrint().info,
    'tprint_success': MockTPrint().success,
    'tprint_warning': MockTPrint().warning,
    'tprint_error': MockTPrint().error
})()

# Import our modules
from training.steps.market_analysis.clusters.feature_analysis import create_feature_analyzer
from training.steps.market_analysis.clusters.feature_enhancement import create_feature_generator
from training.steps.market_analysis.clusters.feature_integration import create_feature_integrator


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate price data with different regimes
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create different market regimes
    regime_lengths = [200, 300, 250, 250]  # Different regime lengths
    regime_vols = [0.15, 0.25, 0.10, 0.30]  # Different volatility levels
    regime_returns = [0.0005, -0.0002, 0.0008, -0.0001]  # Different return levels
    
    prices = [100.0]
    volumes = [1000000]
    
    current_regime = 0
    regime_count = 0
    
    for i in range(1, n_samples):
        if regime_count >= regime_lengths[current_regime]:
            current_regime = (current_regime + 1) % len(regime_lengths)
            regime_count = 0
        
        # Generate return based on current regime
        daily_return = np.random.normal(regime_returns[current_regime], regime_vols[current_regime])
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
        
        # Generate volume based on volatility
        volume_multiplier = 1 + np.random.normal(0, regime_vols[current_regime] * 2)
        new_volume = max(100000, volumes[-1] * volume_multiplier)
        volumes.append(int(new_volume))
        
        regime_count += 1
    
    # Create OHLCV data
    data = {
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    }
    
    df = pd.DataFrame(data)
    
    # Ensure high >= low and high/low are reasonable
    df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
    df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    return df


def test_feature_enhancement():
    """Test feature enhancement functionality."""
    print("Testing Feature Enhancement...")
    
    try:
        # Create sample data
        market_data = create_sample_market_data(500)
        print(f"Created market data with {len(market_data)} samples")
        
        # Test generator
        generator = create_feature_generator()
        enhanced_features, enhanced_names = generator.generate_enhanced_features(market_data)
        
        print(f"✓ Feature enhancement completed")
        print(f"  - Enhanced features: {len(enhanced_names)}")
        print(f"  - Feature matrix shape: {enhanced_features.shape}")
        
        # Test specific feature categories
        vol_features = [name for name in enhanced_names if 'vol' in name.lower()]
        trend_features = [name for name in enhanced_names if any(x in name.lower() for x in ['sma', 'ema', 'trend'])]
        momentum_features = [name for name in enhanced_names if any(x in name.lower() for x in ['rsi', 'macd', 'stoch'])]
        volume_features = [name for name in enhanced_names if 'volume' in name.lower()]
        regime_features = [name for name in enhanced_names if 'regime' in name.lower()]
        
        print(f"  - Volatility features: {len(vol_features)}")
        print(f"  - Trend features: {len(trend_features)}")
        print(f"  - Momentum features: {len(momentum_features)}")
        print(f"  - Volume features: {len(volume_features)}")
        print(f"  - Regime features: {len(regime_features)}")
        
        # Check for NaN values
        nan_count = np.isnan(enhanced_features).sum()
        print(f"  - NaN values: {nan_count}")
        
        # Check for infinite values
        inf_count = np.isinf(enhanced_features).sum()
        print(f"  - Infinite values: {inf_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature enhancement failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_analysis():
    """Test feature analysis functionality."""
    print("Testing Feature Analysis...")
    
    try:
        # Create sample data
        market_data = create_sample_market_data(500)
        
        # Create sample features
        returns = market_data['close'].pct_change().dropna()
        features = np.column_stack([
            returns.rolling(20).std().fillna(0).values,
            returns.rolling(50).std().fillna(0).values,
            (market_data['close'] / market_data['close'].rolling(20).mean()).fillna(1).values
        ])
        feature_names = ['vol_20', 'vol_50', 'price_ma_ratio']
        
        # Create sample labels
        labels = np.random.randint(0, 3, len(features))
        
        # Test analyzer
        analyzer = create_feature_analyzer()
        result = analyzer.analyze_current_features(market_data, features, feature_names, labels)
        
        print(f"✓ Feature analysis completed")
        print(f"  - Current features: {len(result.current_features)}")
        print(f"  - Missing features: {len(result.missing_features)}")
        print(f"  - Separation score: {result.regime_separation_score:.3f}")
        print(f"  - Recommendations: {len(result.recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_integration():
    """Test feature integration functionality."""
    print("Testing Feature Integration...")
    
    try:
        # Create sample data
        market_data = create_sample_market_data(500)
        
        # Create sample existing features
        returns = market_data['close'].pct_change().dropna()
        existing_features = np.column_stack([
            returns.rolling(20).std().fillna(0).values,
            returns.rolling(50).std().fillna(0).values
        ])
        existing_names = ['vol_20', 'vol_50']
        
        # Create sample labels
        labels = np.random.randint(0, 3, len(existing_features))
        
        # Test integrator
        integrator = create_feature_integrator()
        result = integrator.integrate_enhanced_features(
            market_data, existing_features, existing_names, labels
        )
        
        print(f"✓ Feature integration completed")
        print(f"  - Original features: {result['original_feature_count']}")
        print(f"  - Enhanced features: {result['enhanced_feature_count']}")
        print(f"  - Final features: {result['final_feature_count']}")
        print(f"  - Enhancement ratio: {result['enhancement_ratio']:.2f}")
        print(f"  - Optimization ratio: {result['optimization_ratio']:.2f}")
        
        # Test integration report
        report = result['integration_report']
        if 'error' not in report:
            print(f"  - Separation score: {report['separation_score']:.3f}")
            print(f"  - Feature categories: {report['feature_categories']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLE FEATURE ENHANCEMENT TEST")
    print("=" * 60)
    
    tests = [
        test_feature_enhancement,
        test_feature_analysis,
        test_feature_integration
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
        print()
    
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✓ All tests passed! Feature enhancement is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)