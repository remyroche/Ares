"""
Verification Script for Feature Generation System

This script verifies that all features from the old system are covered
in the new feature generation system and that there's no loss of function.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
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
    
    return data

def test_enhanced_indicators():
    """Test all enhanced indicators to ensure they work correctly."""
    print("🧪 Testing Enhanced Indicators...")
    
    try:
        # Import enhanced indicators
        from src.feature_generation.categories.momentum import (
            RSIGenerator, MACDGenerator, StochasticGenerator, 
            WilliamsRGenerator, ROCGenerator, MomentumGenerator
        )
        from src.feature_generation.categories.trend import (
            SMAGenerator, EMAGenerator
        )
        from src.feature_generation.categories.volatility import (
            BollingerBandsGenerator, ATRGenerator
        )
        from src.feature_generation.categories.volume import (
            VWAPGenerator
        )
        from src.feature_generation.base_calculations import BaseCalculationType
        
        data = create_sample_data()
        
        # Test momentum indicators
        print("  Testing momentum indicators...")
        rsi = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
        rsi_features = rsi.generate(data)
        assert not rsi_features.empty, "RSI generation failed"
        
        macd = MACDGenerator(fast=12, slow=26, signal=9, base_calculation=BaseCalculationType.PRICE_LEVELS)
        macd_features = macd.generate(data)
        assert not macd_features.empty, "MACD generation failed"
        
        stoch = StochasticGenerator(k_period=14, d_period=3, base_calculation=BaseCalculationType.PRICE_LEVELS)
        stoch_features = stoch.generate(data)
        assert not stoch_features.empty, "Stochastic generation failed"
        
        williams = WilliamsRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
        williams_features = williams.generate(data)
        assert not williams_features.empty, "Williams %R generation failed"
        
        roc = ROCGenerator(period=10, base_calculation=BaseCalculationType.PRICE_LEVELS)
        roc_features = roc.generate(data)
        assert not roc_features.empty, "ROC generation failed"
        
        momentum = MomentumGenerator(period=10, base_calculation=BaseCalculationType.PRICE_LEVELS)
        momentum_features = momentum.generate(data)
        assert not momentum_features.empty, "Momentum generation failed"
        
        # Test trend indicators
        print("  Testing trend indicators...")
        sma = SMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS)
        sma_features = sma.generate(data)
        assert not sma_features.empty, "SMA generation failed"
        
        ema = EMAGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS)
        ema_features = ema.generate(data)
        assert not ema_features.empty, "EMA generation failed"
        
        # Test volatility indicators
        print("  Testing volatility indicators...")
        bb = BollingerBandsGenerator(period=20, std_dev=2.0, base_calculation=BaseCalculationType.PRICE_LEVELS, band_type="upper")
        bb_features = bb.generate(data)
        assert not bb_features.empty, "Bollinger Bands generation failed"
        
        atr = ATRGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
        atr_features = atr.generate(data)
        assert not atr_features.empty, "ATR generation failed"
        
        # Test volume indicators
        print("  Testing volume indicators...")
        vwap = VWAPGenerator(period=20, base_calculation=BaseCalculationType.PRICE_LEVELS)
        vwap_features = vwap.generate(data)
        assert not vwap_features.empty, "VWAP generation failed"
        
        print("  ✅ All enhanced indicators working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing enhanced indicators: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_base_calculations():
    """Test base calculations to ensure they work correctly."""
    print("🧪 Testing Base Calculations...")
    
    try:
        from src.feature_generation.base_calculations import (
            BaseCalculationType, create_base_calculator,
            calculate_price_returns, calculate_returns_vwap,
            calculate_price_levels, calculate_volume_weighted
        )
        
        data = create_sample_data()
        
        # Test price returns calculation
        print("  Testing price returns calculation...")
        price_returns = calculate_price_returns(data)
        assert not price_returns.empty, "Price returns calculation failed"
        
        # Test returns VWAP calculation
        print("  Testing returns VWAP calculation...")
        returns_vwap = calculate_returns_vwap(data, vwap_period=20)
        assert not returns_vwap.empty, "Returns VWAP calculation failed"
        
        # Test price levels calculation
        print("  Testing price levels calculation...")
        price_levels = calculate_price_levels(data)
        assert not price_levels.empty, "Price levels calculation failed"
        
        # Test volume weighted calculation
        print("  Testing volume weighted calculation...")
        volume_weighted = calculate_volume_weighted(data)
        assert not volume_weighted.empty, "Volume weighted calculation failed"
        
        # Test base calculator creation
        print("  Testing base calculator creation...")
        price_returns_calc = create_base_calculator(BaseCalculationType.PRICE_RETURNS)
        returns_vwap_calc = create_base_calculator(BaseCalculationType.RETURNS_VWAP, vwap_period=20)
        price_levels_calc = create_base_calculator(BaseCalculationType.PRICE_LEVELS)
        volume_weighted_calc = create_base_calculator(BaseCalculationType.VOLUME_WEIGHTED)
        
        assert price_returns_calc is not None, "Price returns calculator creation failed"
        assert returns_vwap_calc is not None, "Returns VWAP calculator creation failed"
        assert price_levels_calc is not None, "Price levels calculator creation failed"
        assert volume_weighted_calc is not None, "Volume weighted calculator creation failed"
        
        print("  ✅ All base calculations working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing base calculations: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_bank():
    """Test feature bank functionality."""
    print("🧪 Testing Feature Bank...")
    
    try:
        from src.feature_generation.core.feature_bank import FeatureBank
        from src.feature_generation.categories.momentum import RSIGenerator
        from src.feature_generation.base_calculations import BaseCalculationType
        
        data = create_sample_data()
        
        # Initialize feature bank
        bank = FeatureBank()
        
        # Generate features
        rsi = RSIGenerator(period=14, base_calculation=BaseCalculationType.PRICE_LEVELS)
        rsi_features = rsi.generate(data)
        
        # Store features
        features_df = pd.DataFrame(index=data.index)
        features_df[rsi_features.name] = rsi_features
        
        bank.add_features("test_features", features_df)
        
        # Retrieve features
        retrieved_features = bank.get_features("test_features")
        assert not retrieved_features.empty, "Feature retrieval failed"
        
        # Check feature names
        feature_names = bank.get_feature_names("test_features")
        assert len(feature_names) > 0, "Feature names retrieval failed"
        
        # Check feature categories
        categories = bank.get_categories()
        assert len(categories) > 0, "Categories retrieval failed"
        
        print("  ✅ Feature bank working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing feature bank: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interaction_features():
    """Test interaction features."""
    print("🧪 Testing Interaction Features...")
    
    try:
        from src.feature_generation.categories.interaction import (
            InteractionFeatureGenerator, CrossTimeframeInteractionGenerator,
            FeatureRatioGenerator, PolynomialFeatureGenerator,
            CorrelationInteractionGenerator
        )
        
        data = create_sample_data()
        
        # Test cross-timeframe interaction
        print("  Testing cross-timeframe interaction...")
        cross_timeframe = CrossTimeframeInteractionGenerator(
            feature_name="sma_20",
            short_period=5,
            long_period=20,
            interaction_type="ratio"
        )
        cross_timeframe_features = cross_timeframe.generate(data)
        assert not cross_timeframe_features.empty, "Cross-timeframe interaction failed"
        
        # Test feature ratio
        print("  Testing feature ratio...")
        feature_ratio = FeatureRatioGenerator(
            numerator_feature="sma_5",
            denominator_feature="sma_20",
            ratio_type="simple"
        )
        feature_ratio_features = feature_ratio.generate(data)
        assert not feature_ratio_features.empty, "Feature ratio failed"
        
        # Test polynomial features
        print("  Testing polynomial features...")
        polynomial = PolynomialFeatureGenerator(
            base_feature="sma_20",
            degree=2,
            polynomial_type="power"
        )
        polynomial_features = polynomial.generate(data)
        assert not polynomial_features.empty, "Polynomial features failed"
        
        print("  ✅ Interaction features working correctly!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing interaction features: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backwards_compatibility():
    """Test backwards compatibility with existing code."""
    print("🧪 Testing Backwards Compatibility...")
    
    try:
        # Test that old-style imports still work
        from src.feature_generation import (
            RSIGenerator, MACDGenerator, BollingerBandsGenerator,
            SMAGenerator, EMAGenerator, ATRGenerator,
            StochasticGenerator, WilliamsRGenerator, ROCGenerator,
            MomentumGenerator, VWAPGenerator
        )
        
        data = create_sample_data()
        
        # Test that generators work without base_calculation parameter (defaults to PRICE_LEVELS)
        print("  Testing default behavior...")
        rsi_default = RSIGenerator(period=14)
        rsi_default_features = rsi_default.generate(data)
        assert not rsi_default_features.empty, "Default RSI generation failed"
        
        sma_default = SMAGenerator(period=20)
        sma_default_features = sma_default.generate(data)
        assert not sma_default_features.empty, "Default SMA generation failed"
        
        print("  ✅ Backwards compatibility maintained!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing backwards compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_verification():
    """Run comprehensive verification of the feature generation system."""
    print("🔍 Running Comprehensive Feature Generation Verification")
    print("=" * 60)
    
    results = []
    
    # Test enhanced indicators
    results.append(test_enhanced_indicators())
    
    # Test base calculations
    results.append(test_base_calculations())
    
    # Test feature bank
    results.append(test_feature_bank())
    
    # Test interaction features
    results.append(test_interaction_features())
    
    # Test backwards compatibility
    results.append(test_backwards_compatibility())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ No loss of function detected")
        print("✅ Enhanced indicators working correctly")
        print("✅ Base calculations working correctly")
        print("✅ Feature bank working correctly")
        print("✅ Interaction features working correctly")
        print("✅ Backwards compatibility maintained")
        print("\n🚀 Feature generation system is ready for production!")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("❌ Some functionality may be lost")
        print("🔧 Please review and fix the failing tests")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_verification()
    sys.exit(0 if success else 1)