"""
Enhanced Usage Examples for Unified Feature Generation System

This module demonstrates how to use the enhanced feature generation system
with different base calculations (price returns, returns-based VWAP, etc.)
and interaction features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Import the enhanced feature generation system
from .. import (
    # Base calculations
    BaseCalculationType,
    create_base_calculator,
    calculate_price_returns,
    calculate_returns_vwap,
    
    # Feature generators
    RSIGenerator,
    MACDGenerator,
    BollingerBandsGenerator,
    SMAGenerator,
    
    # Interaction features
    InteractionFeatureGenerator,
    CrossTimeframeInteractionGenerator,
    FeatureRatioGenerator,
    create_interaction_generators,
    
    # Core system
    FeatureBank,
    get_feature_generator
)

def example_1_rsi_with_different_bases():
    """
    Example 1: RSI with different base calculations
    
    This example shows how to generate RSI indicators based on:
    - Price returns
    - Returns-based VWAP
    - Price levels
    """
    print("=== Example 1: RSI with Different Base Calculations ===")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and close is between high and low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # 1. RSI based on price returns
    rsi_returns = RSIGenerator(
        period=14,
        base_calculation=BaseCalculationType.PRICE_RETURNS
    )
    rsi_returns_features = rsi_returns.generate(data)
    print(f"RSI (price returns): {rsi_returns_features.name}")
    print(f"Sample values: {rsi_returns_features.dropna().head()}")
    
    # 2. RSI based on returns-based VWAP
    rsi_vwap = RSIGenerator(
        period=14,
        base_calculation=BaseCalculationType.RETURNS_VWAP,
        vwap_period=20
    )
    rsi_vwap_features = rsi_vwap.generate(data)
    print(f"\nRSI (returns VWAP): {rsi_vwap_features.name}")
    print(f"Sample values: {rsi_vwap_features.dropna().head()}")
    
    # 3. RSI based on price levels (traditional)
    rsi_levels = RSIGenerator(
        period=14,
        base_calculation=BaseCalculationType.PRICE_LEVELS
    )
    rsi_levels_features = rsi_levels.generate(data)
    print(f"\nRSI (price levels): {rsi_levels_features.name}")
    print(f"Sample values: {rsi_levels_features.dropna().head()}")

def example_2_macd_with_different_bases():
    """
    Example 2: MACD with different base calculations
    
    This example shows how to generate MACD indicators based on:
    - Price levels (traditional)
    - Returns-based VWAP
    - Volume-weighted prices
    """
    print("\n=== Example 2: MACD with Different Base Calculations ===")
    
    # Create sample data (reuse from previous example)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and close is between high and low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # 1. MACD based on price levels (traditional)
    macd_levels = MACDGenerator(
        fast=12,
        slow=26,
        signal=9,
        base_calculation=BaseCalculationType.PRICE_LEVELS
    )
    macd_levels_features = macd_levels.generate(data)
    print(f"MACD (price levels): {macd_levels_features.name}")
    print(f"Sample values: {macd_levels_features.dropna().head()}")
    
    # 2. MACD based on returns-based VWAP
    macd_vwap = MACDGenerator(
        fast=12,
        slow=26,
        signal=9,
        base_calculation=BaseCalculationType.RETURNS_VWAP,
        vwap_period=20
    )
    macd_vwap_features = macd_vwap.generate(data)
    print(f"\nMACD (returns VWAP): {macd_vwap_features.name}")
    print(f"Sample values: {macd_vwap_features.dropna().head()}")
    
    # 3. MACD based on volume-weighted prices
    macd_volume = MACDGenerator(
        fast=12,
        slow=26,
        signal=9,
        base_calculation=BaseCalculationType.VOLUME_WEIGHTED,
        vwap_period=20
    )
    macd_volume_features = macd_volume.generate(data)
    print(f"\nMACD (volume weighted): {macd_volume_features.name}")
    print(f"Sample values: {macd_volume_features.dropna().head()}")

def example_3_bollinger_bands_with_different_bases():
    """
    Example 3: Bollinger Bands with different base calculations
    
    This example shows how to generate Bollinger Bands based on:
    - Price levels (traditional)
    - Returns-based VWAP
    - Price returns
    """
    print("\n=== Example 3: Bollinger Bands with Different Base Calculations ===")
    
    # Create sample data (reuse from previous example)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and close is between high and low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # 1. Bollinger Bands based on price levels (traditional)
    bb_levels = BollingerBandsGenerator(
        period=20,
        std_dev=2.0,
        base_calculation=BaseCalculationType.PRICE_LEVELS,
        band_type="upper"
    )
    bb_levels_features = bb_levels.generate(data)
    print(f"BB Upper (price levels): {bb_levels_features.name}")
    print(f"Sample values: {bb_levels_features.dropna().head()}")
    
    # 2. Bollinger Bands based on returns-based VWAP
    bb_vwap = BollingerBandsGenerator(
        period=20,
        std_dev=2.0,
        base_calculation=BaseCalculationType.RETURNS_VWAP,
        band_type="upper",
        vwap_period=20
    )
    bb_vwap_features = bb_vwap.generate(data)
    print(f"\nBB Upper (returns VWAP): {bb_vwap_features.name}")
    print(f"Sample values: {bb_vwap_features.dropna().head()}")
    
    # 3. Bollinger Bands based on price returns
    bb_returns = BollingerBandsGenerator(
        period=20,
        std_dev=2.0,
        base_calculation=BaseCalculationType.PRICE_RETURNS,
        band_type="upper"
    )
    bb_returns_features = bb_returns.generate(data)
    print(f"\nBB Upper (price returns): {bb_returns_features.name}")
    print(f"Sample values: {bb_returns_features.dropna().head()}")

def example_4_interaction_features():
    """
    Example 4: Interaction Features
    
    This example shows how to generate various interaction features:
    - Cross-timeframe interactions
    - Feature ratios
    - Polynomial features
    - Correlation interactions
    """
    print("\n=== Example 4: Interaction Features ===")
    
    # Create sample data (reuse from previous example)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and close is between high and low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # 1. Cross-timeframe interaction (ratio)
    cross_timeframe = CrossTimeframeInteractionGenerator(
        short_period=5,
        long_period=20,
        interaction_type="ratio"
    )
    cross_timeframe_features = cross_timeframe.generate(data)
    print(f"Cross-timeframe ratio: {cross_timeframe_features.name}")
    print(f"Sample values: {cross_timeframe_features.dropna().head()}")
    
    # 2. Feature ratio (SMA ratio)
    feature_ratio = FeatureRatioGenerator(
        numerator_period=5,
        denominator_period=20,
        feature_type="sma"
    )
    feature_ratio_features = feature_ratio.generate(data)
    print(f"\nSMA ratio: {feature_ratio_features.name}")
    print(f"Sample values: {feature_ratio_features.dropna().head()}")
    
    # 3. Create multiple interaction generators
    interaction_generators = create_interaction_generators({
        'cross_timeframe': {
            'short_periods': [5, 10],
            'long_periods': [20],
            'interaction_types': ['ratio', 'difference']
        },
        'feature_ratios': {
            'periods': [(5, 20)],
            'feature_types': ['sma', 'ema']
        }
    })
    
    print(f"\nCreated {len(interaction_generators)} interaction generators:")
    for i, generator in enumerate(interaction_generators):
        print(f"  {i+1}. {generator.config.name}")

def example_5_feature_bank_with_enhanced_features():
    """
    Example 5: Feature Bank with Enhanced Features
    
    This example shows how to use the Feature Bank to generate
    features with different base calculations and interaction features.
    """
    print("\n=== Example 5: Feature Bank with Enhanced Features ===")
    
    # Create sample data (reuse from previous example)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and close is between high and low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # Initialize feature bank
    bank = FeatureBank()
    
    # Generate features with different base calculations
    print("Generating features with different base calculations...")
    
    # RSI with different bases
    rsi_returns = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
    rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    
    # MACD with different bases
    macd_levels = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS)
    macd_vwap = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    
    # Generate features
    features = pd.DataFrame(index=data.index)
    
    # Add RSI features
    features['rsi_returns'] = rsi_returns.generate(data)
    features['rsi_vwap'] = rsi_vwap.generate(data)
    
    # Add MACD features
    features['macd_levels'] = macd_levels.generate(data)
    features['macd_vwap'] = macd_vwap.generate(data)
    
    # Add interaction features
    cross_timeframe = CrossTimeframeInteractionGenerator(5, 20, "ratio")
    features['cross_timeframe_ratio'] = cross_timeframe.generate(data)
    
    print(f"Generated {len(features.columns)} features:")
    for col in features.columns:
        non_null_count = features[col].notna().sum()
        print(f"  {col}: {non_null_count} non-null values")
    
    # Store in feature bank
    bank.add_features("enhanced_features", features)
    print(f"\nStored features in bank under category 'enhanced_features'")
    
    # Retrieve features
    retrieved_features = bank.get_features("enhanced_features")
    print(f"Retrieved features shape: {retrieved_features.shape}")

def example_6_convenience_functions():
    """
    Example 6: Convenience Functions for Base Calculations
    
    This example shows how to use the convenience functions for
    base calculations directly.
    """
    print("\n=== Example 6: Convenience Functions for Base Calculations ===")
    
    # Create sample data (reuse from previous example)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.1) + np.random.rand(100) * 2,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.1) - np.random.rand(100) * 2,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low and close is between high and low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    # 1. Calculate price returns
    price_returns = calculate_price_returns(data, lookback_period=1)
    print(f"Price returns: {price_returns.name}")
    print(f"Sample values: {price_returns.dropna().head()}")
    
    # 2. Calculate returns-based VWAP
    returns_vwap = calculate_returns_vwap(data, vwap_period=20, lookback_period=1)
    print(f"\nReturns-based VWAP: {returns_vwap.name}")
    print(f"Sample values: {returns_vwap.dropna().head()}")
    
    # 3. Calculate price levels
    price_levels = calculate_price_levels(data)
    print(f"\nPrice levels: {price_levels.name}")
    print(f"Sample values: {price_levels.head()}")
    
    # 4. Calculate volume-weighted values
    volume_weighted = calculate_volume_weighted(data, period=20)
    print(f"\nVolume-weighted values: {volume_weighted.name}")
    print(f"Sample values: {volume_weighted.dropna().head()}")

def run_all_examples():
    """Run all examples."""
    print("🚀 Running Enhanced Feature Generation Examples")
    print("=" * 60)
    
    try:
        example_1_rsi_with_different_bases()
        example_2_macd_with_different_bases()
        example_3_bollinger_bands_with_different_bases()
        example_4_interaction_features()
        example_5_feature_bank_with_enhanced_features()
        example_6_convenience_functions()
        
        print("\n✅ All examples completed successfully!")
        print("🎉 Enhanced feature generation system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
        traceback.print_exc()

if __name__ == "__main__":
    run_all_examples()
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
