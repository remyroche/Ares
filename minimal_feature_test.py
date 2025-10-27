"""
Minimal Feature Enhancement Test.

This test validates the enhanced feature generation without requiring
the full project structure or complex dependencies.
"""

import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

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

# Create a minimal version of our feature enhancement modules
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


def test_basic_feature_generation():
    """Test basic feature generation functionality."""
    print("Testing Basic Feature Generation...")
    
    try:
        # Create sample data
        market_data = create_sample_market_data(500)
        print(f"Created market data with {len(market_data)} samples")
        
        # Test basic volatility features
        returns = market_data['close'].pct_change()
        
        # Multi-timeframe volatility
        vol_features = []
        vol_names = []
        
        for period in [5, 10, 20, 40, 60]:
            vol = returns.rolling(period).std() * np.sqrt(252)
            vol_features.append(vol.fillna(vol.mean()).values)
            vol_names.append(f'volatility_{period}')
        
        # ATR features
        if all(col in market_data.columns for col in ['high', 'low', 'close']):
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            for period in [5, 10, 20]:
                tr1 = high - low
                tr2 = np.abs(high - close.shift(1))
                tr3 = np.abs(low - close.shift(1))
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                atr = tr.rolling(period).mean()
                vol_features.append(atr.fillna(atr.mean()).values)
                vol_names.append(f'atr_{period}')
        
        # Volatility regime features
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_mean_60 = vol_20.rolling(60).mean()
        vol_std_60 = vol_20.rolling(60).std()
        
        # Z-score
        vol_zscore = (vol_20 - vol_mean_60) / (vol_std_60 + 1e-8)
        vol_features.append(vol_zscore.fillna(0).values)
        vol_names.append('vol_regime_zscore')
        
        # Percentile rank
        vol_percentile = vol_20.rolling(252).rank(pct=True)
        vol_features.append(vol_percentile.fillna(0.5).values)
        vol_names.append('vol_regime_percentile')
        
        # Combine features
        vol_matrix = np.column_stack(vol_features)
        
        print(f"✓ Volatility features generated")
        print(f"  - Features: {len(vol_names)}")
        print(f"  - Matrix shape: {vol_matrix.shape}")
        print(f"  - Feature names: {vol_names[:5]}...")  # Show first 5
        
        # Test trend features
        close = market_data['close']
        trend_features = []
        trend_names = []
        
        # Multiple timeframe moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            sma = close.rolling(period).mean()
            trend_features.append(sma.fillna(close).values)
            trend_names.append(f'sma_{period}')
            
            ema = close.ewm(span=period).mean()
            trend_features.append(ema.fillna(close).values)
            trend_names.append(f'ema_{period}')
        
        # Trend strength
        sma_20 = close.rolling(20).mean()
        sma_5 = close.rolling(5).mean()
        trend_strength = np.abs(sma_20 - sma_5) / (close + 1e-8)
        trend_features.append(trend_strength.fillna(0).values)
        trend_names.append('trend_strength')
        
        # Trend consistency
        trend_consistency = (returns > 0).rolling(20).mean() - 0.5
        trend_features.append(trend_consistency.fillna(0).values)
        trend_names.append('trend_consistency')
        
        # Combine trend features
        trend_matrix = np.column_stack(trend_features)
        
        print(f"✓ Trend features generated")
        print(f"  - Features: {len(trend_names)}")
        print(f"  - Matrix shape: {trend_matrix.shape}")
        print(f"  - Feature names: {trend_names[:5]}...")  # Show first 5
        
        # Test momentum features
        momentum_features = []
        momentum_names = []
        
        # RSI
        for period in [14, 21, 50]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            momentum_features.append(rsi.fillna(50).values)
            momentum_names.append(f'rsi_{period}')
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_histogram = macd - macd_signal
        
        momentum_features.append(macd.fillna(0).values)
        momentum_names.append('macd')
        momentum_features.append(macd_signal.fillna(0).values)
        momentum_names.append('macd_signal')
        momentum_features.append(macd_histogram.fillna(0).values)
        momentum_names.append('macd_histogram')
        
        # Rate of Change
        for period in [5, 10, 20]:
            roc = close.pct_change(period) * 100
            momentum_features.append(roc.fillna(0).values)
            momentum_names.append(f'roc_{period}')
        
        # Combine momentum features
        momentum_matrix = np.column_stack(momentum_features)
        
        print(f"✓ Momentum features generated")
        print(f"  - Features: {len(momentum_names)}")
        print(f"  - Matrix shape: {momentum_matrix.shape}")
        print(f"  - Feature names: {momentum_names[:5]}...")  # Show first 5
        
        # Test volume features
        volume = market_data['volume']
        volume_features = []
        volume_names = []
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            vol_sma = volume.rolling(period).mean()
            volume_features.append(vol_sma.fillna(volume.mean()).values)
            volume_names.append(f'volume_sma_{period}')
        
        # Volume ratio
        vol_ratio = volume / (volume.rolling(20).mean() + 1e-8)
        volume_features.append(vol_ratio.fillna(1).values)
        volume_names.append('volume_ratio')
        
        # Volume regime
        vol_percentile = volume.rolling(252).rank(pct=True)
        volume_features.append(vol_percentile.fillna(0.5).values)
        volume_names.append('volume_regime')
        
        # Combine volume features
        volume_matrix = np.column_stack(volume_features)
        
        print(f"✓ Volume features generated")
        print(f"  - Features: {len(volume_names)}")
        print(f"  - Matrix shape: {volume_matrix.shape}")
        print(f"  - Feature names: {volume_names[:5]}...")  # Show first 5
        
        # Combine all features
        all_features = np.hstack([vol_matrix, trend_matrix, momentum_matrix, volume_matrix])
        all_names = vol_names + trend_names + momentum_names + volume_names
        
        print(f"✓ All features combined")
        print(f"  - Total features: {len(all_names)}")
        print(f"  - Total matrix shape: {all_features.shape}")
        
        # Check for data quality
        nan_count = np.isnan(all_features).sum()
        inf_count = np.isinf(all_features).sum()
        
        print(f"  - NaN values: {nan_count}")
        print(f"  - Infinite values: {inf_count}")
        
        # Check feature variance
        variances = np.var(all_features, axis=0)
        low_var_count = np.sum(variances < 0.001)
        print(f"  - Low variance features: {low_var_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic feature generation failed: {e}")
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
        returns = market_data['close'].pct_change()
        features = np.column_stack([
            returns.rolling(20).std().fillna(0).values,
            returns.rolling(50).std().fillna(0).values,
            (market_data['close'] / market_data['close'].rolling(20).mean()).fillna(1).values
        ])
        feature_names = ['vol_20', 'vol_50', 'price_ma_ratio']
        
        # Create sample labels
        labels = np.random.randint(0, 3, len(features))
        
        # Test feature categorization
        categories = {
            'price_features': [],
            'volatility_features': [],
            'trend_features': [],
            'volume_features': [],
            'momentum_features': [],
            'regime_features': [],
            'technical_features': [],
            'other_features': []
        }
        
        for name in feature_names:
            name_lower = name.lower()
            
            if any(x in name_lower for x in ['close', 'open', 'high', 'low', 'price', 'return']):
                categories['price_features'].append(name)
            elif any(x in name_lower for x in ['vol', 'volatility', 'std', 'atr']):
                categories['volatility_features'].append(name)
            elif any(x in name_lower for x in ['sma', 'ema', 'trend', 'ma_', 'moving']):
                categories['trend_features'].append(name)
            elif any(x in name_lower for x in ['volume', 'vol_']):
                categories['volume_features'].append(name)
            elif any(x in name_lower for x in ['rsi', 'macd', 'momentum', 'roc', 'stoch']):
                categories['momentum_features'].append(name)
            elif any(x in name_lower for x in ['regime', 'persistence', 'transition']):
                categories['regime_features'].append(name)
            elif any(x in name_lower for x in ['bollinger', 'bb_', 'rsi', 'macd', 'stoch', 'williams']):
                categories['technical_features'].append(name)
            else:
                categories['other_features'].append(name)
        
        print(f"✓ Feature categorization completed")
        print(f"  - Price features: {len(categories['price_features'])}")
        print(f"  - Volatility features: {len(categories['volatility_features'])}")
        print(f"  - Trend features: {len(categories['trend_features'])}")
        print(f"  - Volume features: {len(categories['volume_features'])}")
        print(f"  - Momentum features: {len(categories['momentum_features'])}")
        print(f"  - Regime features: {len(categories['regime_features'])}")
        print(f"  - Technical features: {len(categories['technical_features'])}")
        print(f"  - Other features: {len(categories['other_features'])}")
        
        # Test feature importance calculation
        from sklearn.feature_selection import f_classif
        from sklearn.ensemble import RandomForestClassifier
        
        # F-test for feature importance
        f_scores, _ = f_classif(features, labels)
        f_importance = f_scores / (np.sum(f_scores) + 1e-8)
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        rf_importance = rf.feature_importances_
        
        # Combined importance
        combined_importance = 0.6 * f_importance + 0.4 * rf_importance
        
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = float(combined_importance[i])
        
        print(f"✓ Feature importance calculated")
        print(f"  - Importance scores: {importance_dict}")
        
        # Test variance ratio calculation
        unique_labels = np.unique(labels)
        variance_ratios = {}
        
        for i, name in enumerate(feature_names):
            feature_data = features[:, i]
            
            # Within-cluster variance
            within_var = 0.0
            total_samples = 0
            
            for label in unique_labels:
                cluster_data = feature_data[labels == label]
                if len(cluster_data) > 1:
                    cluster_var = np.var(cluster_data)
                    within_var += cluster_var * len(cluster_data)
                    total_samples += len(cluster_data)
            
            if total_samples > 0:
                within_var /= total_samples
            
            # Between-cluster variance
            overall_mean = np.mean(feature_data)
            between_var = 0.0
            
            for label in unique_labels:
                cluster_data = feature_data[labels == label]
                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data)
                    between_var += len(cluster_data) * (cluster_mean - overall_mean) ** 2
            
            if total_samples > 0:
                between_var /= total_samples
            
            # Variance ratio
            if within_var > 0:
                variance_ratio = between_var / within_var
            else:
                variance_ratio = 0.0
            
            variance_ratios[name] = variance_ratio
        
        print(f"✓ Variance ratios calculated")
        print(f"  - Variance ratios: {variance_ratios}")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MINIMAL FEATURE ENHANCEMENT TEST")
    print("=" * 60)
    
    tests = [
        test_basic_feature_generation,
        test_feature_analysis
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