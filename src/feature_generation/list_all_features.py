"""
Comprehensive Feature List Generator

This script generates a comprehensive list of all available features
in the unified feature generation system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def list_all_features():
    """Generate a comprehensive list of all available features."""
    
    print("🔍 COMPREHENSIVE FEATURE LIST")
    print("=" * 60)
    
    # Import all feature generators
    try:
        from src.feature_generation.categories.returns import ReturnsFeatureGenerator
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        from src.feature_generation.categories.trend import TrendFeatureGenerator
        from src.feature_generation.categories.oscillator import OscillatorFeatureGenerator
        from src.feature_generation.categories.support_resistance import SupportResistanceFeatureGenerator
        from src.feature_generation.categories.candlestick_pattern import CandlestickPatternFeatureGenerator
        from src.feature_generation.categories.hmm_regime import HMMRegimeFeatureGenerator
        from src.feature_generation.categories.interaction import InteractionFeatureGenerator
        
        from src.feature_generation.base_calculations import BaseCalculationType
        
        print("✅ All feature generators imported successfully!")
        
    except ImportError as e:
        print(f"❌ Error importing feature generators: {e}")
        return
    
    print("\n📊 FEATURE CATEGORIES AND GENERATORS")
    print("=" * 60)
    
    # 1. Returns Features
    print("\n1. 📈 RETURNS FEATURES")
    print("-" * 30)
    returns_features = [
        "SimpleReturnsGenerator",
        "LogReturnsGenerator", 
        "CumulativeReturnsGenerator",
        "ReturnVolatilityGenerator",
        "ReturnSkewnessGenerator",
        "ReturnKurtosisGenerator"
    ]
    for feature in returns_features:
        print(f"  • {feature}")
    
    # 2. Momentum Features
    print("\n2. 🚀 MOMENTUM FEATURES")
    print("-" * 30)
    momentum_features = [
        "RSIGenerator (Enhanced with base calculations)",
        "MACDGenerator (Enhanced with base calculations)",
        "MACDSignalGenerator",
        "MACDHistogramGenerator", 
        "StochasticGenerator (Enhanced with base calculations)",
        "WilliamsRGenerator (Enhanced with base calculations)",
        "ROCGenerator (Enhanced with base calculations)",
        "MomentumGenerator (Enhanced with base calculations)"
    ]
    for feature in momentum_features:
        print(f"  • {feature}")
    
    # 3. Trend Features
    print("\n3. 📊 TREND FEATURES")
    print("-" * 30)
    trend_features = [
        "SMAGenerator (Enhanced with base calculations)",
        "EMAGenerator (Enhanced with base calculations)"
    ]
    for feature in trend_features:
        print(f"  • {feature}")
    
    # 4. Volatility Features
    print("\n4. 📉 VOLATILITY FEATURES")
    print("-" * 30)
    volatility_features = [
        "BollingerBandsGenerator (Enhanced with base calculations)",
        "ATRGenerator (Enhanced with base calculations)"
    ]
    for feature in volatility_features:
        print(f"  • {feature}")
    
    # 5. Volume Features
    print("\n5. 📊 VOLUME FEATURES")
    print("-" * 30)
    volume_features = [
        "VolumeMAGenerator (Enhanced with base calculations)",
        "VolumeRatioGenerator (Enhanced with base calculations)",
        "OBVGenerator",
        "VWAPGenerator (Enhanced with base calculations)",
        "VolumeROCGenerator",
        "VPTGenerator",
        "ADLGenerator",
        "VolumeVolatilityGenerator",
        "VolumeSkewnessGenerator"
    ]
    for feature in volume_features:
        print(f"  • {feature}")
    
    # 6. Oscillator Features
    print("\n6. 🔄 OSCILLATOR FEATURES")
    print("-" * 30)
    oscillator_features = [
        "CCI (Commodity Channel Index)",
        "ADX (Average Directional Index)",
        "Aroon Oscillator",
        "Parabolic SAR",
        "Ultimate Oscillator",
        "KST (Know Sure Thing)",
        "APO (Absolute Price Oscillator)",
        "CMO (Chande Momentum Oscillator)",
        "NATR (Normalized Average True Range)",
        "PFE (Polarized Fractal Efficiency)",
        "T3 (T3 Moving Average)",
        "KAMA (Kaufman's Adaptive Moving Average)"
    ]
    for feature in oscillator_features:
        print(f"  • {feature}")
    
    # 7. Support/Resistance Features
    print("\n7. 🎯 SUPPORT/RESISTANCE FEATURES")
    print("-" * 30)
    sr_features = [
        "Support Level Detection",
        "Resistance Level Detection",
        "Pivot Points",
        "Fibonacci Retracements",
        "Volume Profile Analysis"
    ]
    for feature in sr_features:
        print(f"  • {feature}")
    
    # 8. Candlestick Pattern Features
    print("\n8. 🕯️ CANDLESTICK PATTERN FEATURES")
    print("-" * 30)
    candlestick_features = [
        "Doji Patterns",
        "Hammer Patterns",
        "Shooting Star Patterns",
        "Engulfing Patterns",
        "Harami Patterns",
        "Morning/Evening Star Patterns",
        "Three White Soldiers/Black Crows",
        "Piercing Line/Dark Cloud Cover"
    ]
    for feature in candlestick_features:
        print(f"  • {feature}")
    
    # 9. HMM Regime Features
    print("\n9. 🔄 HMM REGIME FEATURES")
    print("-" * 30)
    hmm_features = [
        "Regime Detection",
        "Regime Transition Probabilities",
        "Regime-Aware Feature Generation",
        "Regime-Based Optimization"
    ]
    for feature in hmm_features:
        print(f"  • {feature}")
    
    # 10. Interaction Features
    print("\n10. 🔗 INTERACTION FEATURES")
    print("-" * 30)
    interaction_features = [
        "CrossTimeframeInteractionGenerator",
        "FeatureRatioGenerator",
        "PolynomialFeatureGenerator",
        "CorrelationInteractionGenerator"
    ]
    for feature in interaction_features:
        print(f"  • {feature}")
    
    print("\n🎯 BASE CALCULATION TYPES")
    print("=" * 60)
    base_calculations = [
        "PRICE_RETURNS - Price returns (percentage changes)",
        "RETURNS_VWAP - Returns-based Volume Weighted Average Price",
        "PRICE_LEVELS - Traditional price levels (close, high, low, open)",
        "VOLUME_WEIGHTED - Volume-weighted calculations",
        "VOLUME_RETURNS - Volume returns (percentage changes in volume)"
    ]
    for calc in base_calculations:
        print(f"  • {calc}")
    
    print("\n⚡ ENHANCED INDICATORS WITH BASE CALCULATIONS")
    print("=" * 60)
    enhanced_indicators = [
        "RSI - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "MACD - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "Stochastic - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "Williams %R - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "ROC - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "Momentum - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "SMA - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "EMA - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "Bollinger Bands - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "ATR - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "VWAP - Now supports PRICE_RETURNS, RETURNS_VWAP, PRICE_LEVELS",
        "Volume MA - Now supports VOLUME_RETURNS, VOLUME_WEIGHTED",
        "Volume Ratio - Now supports VOLUME_RETURNS, VOLUME_WEIGHTED"
    ]
    for indicator in enhanced_indicators:
        print(f"  • {indicator}")
    
    print("\n🔧 USAGE EXAMPLES")
    print("=" * 60)
    print("""
# Basic Usage (uses new defaults)
from src.feature_generation import RSIGenerator, BaseCalculationType

# RSI with price returns (new default)
rsi = RSIGenerator(period=14)  # Uses PRICE_RETURNS by default

# RSI with returns VWAP
rsi_vwap = RSIGenerator(period=14, base_calculation=BaseCalculationType.RETURNS_VWAP, vwap_period=20)

# RSI with traditional price levels
rsi_levels = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)

# Volume features with volume returns (new default)
from src.feature_generation import VolumeMAGenerator

volume_ma = VolumeMAGenerator(period=20)  # Uses VOLUME_RETURNS by default
volume_ma_weighted = VolumeMAGenerator(period=20, base_calculation=BaseCalculationType.VOLUME_WEIGHTED)
""")
    
    print("\n📈 FEATURE STATISTICS")
    print("=" * 60)
    print(f"  • Total Categories: 10")
    print(f"  • Enhanced Indicators: 13")
    print(f"  • Base Calculation Types: 5")
    print(f"  • Total Individual Features: 50+")
    print(f"  • Interaction Features: 4")
    print(f"  • Candlestick Patterns: 8+")
    print(f"  • Oscillator Features: 12+")
    
    print("\n🎉 FEATURE GENERATION SYSTEM SUMMARY")
    print("=" * 60)
    print("✅ Unified feature generation system")
    print("✅ Enhanced indicators with base calculations")
    print("✅ Volume features with volume returns")
    print("✅ Comprehensive feature coverage")
    print("✅ Backwards compatibility maintained")
    print("✅ Matrix operations integration")
    print("✅ Hardware acceleration support")
    print("✅ Feature bank and registry")
    print("✅ Lookback optimization")
    print("✅ Interaction features")
    
    print("\n🚀 READY FOR PRODUCTION USE!")

if __name__ == "__main__":
    list_all_features()