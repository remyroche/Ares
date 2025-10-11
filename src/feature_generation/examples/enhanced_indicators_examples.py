"""
Enhanced Indicators Examples

This module demonstrates all the enhanced indicators that now support
different base calculations (price returns, returns-based VWAP, etc.).
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
    
    # Enhanced indicators
    RSIGenerator,
    MACDGenerator,
    BollingerBandsGenerator,
    SMAGenerator,
    EMAGenerator,
    ATRGenerator,
    StochasticGenerator,
    WilliamsRGenerator,
    ROCGenerator,
    MomentumGenerator,
    VWAPGenerator,
    
    # Core system
    FeatureBank,
    get_feature_generator
)

def example_1_enhanced_momentum_indicators():
    """
    Example 1: Enhanced Momentum Indicators with Base Calculations
    
    This example shows how to generate momentum indicators based on:
    - Price returns
    - Returns-based VWAP
    - Price levels
    """
    print("=== Example 1: Enhanced Momentum Indicators ===")
    
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
    
    # 1. RSI with different base calculations
    print("1. RSI with different base calculations:")
    
    rsi_returns = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
    rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    rsi_levels = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    rsi_returns_features = rsi_returns.generate(data)
    rsi_vwap_features = rsi_vwap.generate(data)
    rsi_levels_features = rsi_levels.generate(data)
    
    print(f"  RSI (price returns): {rsi_returns_features.name}")
    print(f"  RSI (returns VWAP): {rsi_vwap_features.name}")
    print(f"  RSI (price levels): {rsi_levels_features.name}")
    
    # 2. MACD with different base calculations
    print("\n2. MACD with different base calculations:")
    
    macd_returns = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_RETURNS)
    macd_vwap = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    macd_levels = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    macd_returns_features = macd_returns.generate(data)
    macd_vwap_features = macd_vwap.generate(data)
    macd_levels_features = macd_levels.generate(data)
    
    print(f"  MACD (price returns): {macd_returns_features.name}")
    print(f"  MACD (returns VWAP): {macd_vwap_features.name}")
    print(f"  MACD (price levels): {macd_levels_features.name}")
    
    # 3. Stochastic with different base calculations
    print("\n3. Stochastic with different base calculations:")
    
    stoch_returns = StochasticGenerator(k_period=14, d_period=3, base_calculation=BaseCalculationType.PRICE_RETURNS)
    stoch_vwap = StochasticGenerator(k_period=14, d_period=3, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    stoch_levels = StochasticGenerator(k_period=14, d_period=3, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    stoch_returns_features = stoch_returns.generate(data)
    stoch_vwap_features = stoch_vwap.generate(data)
    stoch_levels_features = stoch_levels.generate(data)
    
    print(f"  Stochastic (price returns): {stoch_returns_features.name}")
    print(f"  Stochastic (returns VWAP): {stoch_vwap_features.name}")
    print(f"  Stochastic (price levels): {stoch_levels_features.name}")
    
    # 4. Williams %R with different base calculations
    print("\n4. Williams %R with different base calculations:")
    
    williams_returns = WilliamsRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
    williams_vwap = WilliamsRGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    williams_levels = WilliamsRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    williams_returns_features = williams_returns.generate(data)
    williams_vwap_features = williams_vwap.generate(data)
    williams_levels_features = williams_levels.generate(data)
    
    print(f"  Williams %R (price returns): {williams_returns_features.name}")
    print(f"  Williams %R (returns VWAP): {williams_vwap_features.name}")
    print(f"  Williams %R (price levels): {williams_levels_features.name}")
    
    # 5. ROC with different base calculations
    print("\n5. ROC with different base calculations:")
    
    roc_returns = ROCGenerator(period=10, base_calculation=BaseCalculationType.PRICE_RETURNS)
    roc_vwap = ROCGenerator(period=10, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    roc_levels = ROCGenerator(period=10, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    roc_returns_features = roc_returns.generate(data)
    roc_vwap_features = roc_vwap.generate(data)
    roc_levels_features = roc_levels.generate(data)
    
    print(f"  ROC (price returns): {roc_returns_features.name}")
    print(f"  ROC (returns VWAP): {roc_vwap_features.name}")
    print(f"  ROC (price levels): {roc_levels_features.name}")
    
    # 6. Momentum with different base calculations
    print("\n6. Momentum with different base calculations:")
    
    momentum_returns = MomentumGenerator(period=10, base_calculation=BaseCalculationType.PRICE_RETURNS)
    momentum_vwap = MomentumGenerator(period=10, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    momentum_levels = MomentumGenerator(period=10, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    momentum_returns_features = momentum_returns.generate(data)
    momentum_vwap_features = momentum_vwap.generate(data)
    momentum_levels_features = momentum_levels.generate(data)
    
    print(f"  Momentum (price returns): {momentum_returns_features.name}")
    print(f"  Momentum (returns VWAP): {momentum_vwap_features.name}")
    print(f"  Momentum (price levels): {momentum_levels_features.name}")

def example_2_enhanced_trend_indicators():
    """
    Example 2: Enhanced Trend Indicators with Base Calculations
    
    This example shows how to generate trend indicators based on:
    - Price returns
    - Returns-based VWAP
    - Price levels
    """
    print("\n=== Example 2: Enhanced Trend Indicators ===")
    
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
    
    # 1. SMA with different base calculations
    print("1. SMA with different base calculations:")
    
    sma_returns = SMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_RETURNS)
    sma_vwap = SMAGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    sma_levels = SMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    sma_returns_features = sma_returns.generate(data)
    sma_vwap_features = sma_vwap.generate(data)
    sma_levels_features = sma_levels.generate(data)
    
    print(f"  SMA (price returns): {sma_returns_features.name}")
    print(f"  SMA (returns VWAP): {sma_vwap_features.name}")
    print(f"  SMA (price levels): {sma_levels_features.name}")
    
    # 2. EMA with different base calculations
    print("\n2. EMA with different base calculations:")
    
    ema_returns = EMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_RETURNS)
    ema_vwap = EMAGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    ema_levels = EMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    ema_returns_features = ema_returns.generate(data)
    ema_vwap_features = ema_vwap.generate(data)
    ema_levels_features = ema_levels.generate(data)
    
    print(f"  EMA (price returns): {ema_returns_features.name}")
    print(f"  EMA (returns VWAP): {ema_vwap_features.name}")
    print(f"  EMA (price levels): {ema_levels_features.name}")

def example_3_enhanced_volatility_indicators():
    """
    Example 3: Enhanced Volatility Indicators with Base Calculations
    
    This example shows how to generate volatility indicators based on:
    - Price returns
    - Returns-based VWAP
    - Price levels
    """
    print("\n=== Example 3: Enhanced Volatility Indicators ===")
    
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
    
    # 1. Bollinger Bands with different base calculations
    print("1. Bollinger Bands with different base calculations:")
    
    bb_returns = BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_RETURNS, band_type="upper")
    bb_vwap = BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20, band_type="upper")
    bb_levels = BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS, band_type="upper")
    
    bb_returns_features = bb_returns.generate(data)
    bb_vwap_features = bb_vwap.generate(data)
    bb_levels_features = bb_levels.generate(data)
    
    print(f"  BB Upper (price returns): {bb_returns_features.name}")
    print(f"  BB Upper (returns VWAP): {bb_vwap_features.name}")
    print(f"  BB Upper (price levels): {bb_levels_features.name}")
    
    # 2. ATR with different base calculations
    print("\n2. ATR with different base calculations:")
    
    atr_returns = ATRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
    atr_vwap = ATRGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    atr_levels = ATRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    atr_returns_features = atr_returns.generate(data)
    atr_vwap_features = atr_vwap.generate(data)
    atr_levels_features = atr_levels.generate(data)
    
    print(f"  ATR (price returns): {atr_returns_features.name}")
    print(f"  ATR (returns VWAP): {atr_vwap_features.name}")
    print(f"  ATR (price levels): {atr_levels_features.name}")

def example_4_enhanced_volume_indicators():
    """
    Example 4: Enhanced Volume Indicators with Base Calculations
    
    This example shows how to generate volume indicators based on:
    - Price returns
    - Returns-based VWAP
    - Price levels
    """
    print("\n=== Example 4: Enhanced Volume Indicators ===")
    
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
    
    # 1. VWAP with different base calculations
    print("1. VWAP with different base calculations:")
    
    vwap_returns = VWAPGenerator(period=20, base_calculation=BaseCalculationType.PRICE_RETURNS)
    vwap_vwap = VWAPGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    vwap_levels = VWAPGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS)
    
    vwap_returns_features = vwap_returns.generate(data)
    vwap_vwap_features = vwap_vwap.generate(data)
    vwap_levels_features = vwap_levels.generate(data)
    
    print(f"  VWAP (price returns): {vwap_returns_features.name}")
    print(f"  VWAP (returns VWAP): {vwap_vwap_features.name}")
    print(f"  VWAP (price levels): {vwap_levels_features.name}")

def example_5_comprehensive_feature_generation():
    """
    Example 5: Comprehensive Feature Generation with All Enhanced Indicators
    
    This example shows how to generate a comprehensive set of features
    using all enhanced indicators with different base calculations.
    """
    print("\n=== Example 5: Comprehensive Feature Generation ===")
    
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
    
    # Create comprehensive feature set
    features = pd.DataFrame(index=data.index)
    
    # Momentum indicators with different base calculations
    print("Generating momentum indicators...")
    momentum_indicators = [
        RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS),
        RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20),
        MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS),
        MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20),
        StochasticGenerator(k_period=14, d_period=3, base_calculation=BaseCalculationType.PRICE_LEVELS),
        WilliamsRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS),
        ROCGenerator(period=10, base_calculation=BaseCalculationType.PRICE_LEVELS),
        MomentumGenerator(period=10, base_calculation=BaseCalculationType.PRICE_LEVELS)
    ]
    
    for indicator in momentum_indicators:
        features[indicator.name] = indicator.generate(data)
    
    # Trend indicators with different base calculations
    print("Generating trend indicators...")
    trend_indicators = [
        SMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS),
        SMAGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20),
        EMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS),
        EMAGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    ]
    
    for indicator in trend_indicators:
        features[indicator.name] = indicator.generate(data)
    
    # Volatility indicators with different base calculations
    print("Generating volatility indicators...")
    volatility_indicators = [
        BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS, band_type="upper"),
        BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20, band_type="upper"),
        ATRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS),
        ATRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_RETURNS)
    ]
    
    for indicator in volatility_indicators:
        features[indicator.name] = indicator.generate(data)
    
    # Volume indicators with different base calculations
    print("Generating volume indicators...")
    volume_indicators = [
        VWAPGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS),
        VWAPGenerator(period=20, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)
    ]
    
    for indicator in volume_indicators:
        features[indicator.name] = indicator.generate(data)
    
    # Store in feature bank
    bank.add_features("enhanced_indicators", features)
    
    print(f"\nGenerated {len(features.columns)} enhanced indicators:")
    for col in features.columns:
        non_null_count = features[col].notna().sum()
        print(f"  {col}: {non_null_count} non-null values")
    
    # Retrieve features
    retrieved_features = bank.get_features("enhanced_indicators")
    print(f"\nRetrieved features shape: {retrieved_features.shape}")
    
    return features

def run_all_enhanced_indicator_examples():
    """Run all enhanced indicator examples."""
    print("🚀 Running Enhanced Indicators Examples")
    print("=" * 60)
    
    try:
        example_1_enhanced_momentum_indicators()
        example_2_enhanced_trend_indicators()
        example_3_enhanced_volatility_indicators()
        example_4_enhanced_volume_indicators()
        features = example_5_comprehensive_feature_generation()
        
        print("\n✅ All enhanced indicator examples completed successfully!")
        print("🎉 Enhanced indicators with base calculations are working correctly!")
        print(f"📊 Total features generated: {len(features.columns)}")
        
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
    run_all_enhanced_indicator_examples()
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
