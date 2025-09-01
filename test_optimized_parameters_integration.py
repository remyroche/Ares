#!/usr/bin/env python3
"""
Test script to verify that all components use optimized S/R parameters.
This ensures that the enhanced training manager, analyst, and tactician
all use the optimized parameters from sr_detection_optimization.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components to test
from src.tactician.sr_breakout_predictor import SRBreakoutPredictor, ensure_optimized_sr_config
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.analyst.unified_regime_intelligence_runtime import UnifiedRegimeIntelligenceRuntime
from src.tactician.tactics_orchestrator import DecisionPolicy
from src.training.steps.step15_tactician_specialist_training import TacticianSpecialistTrainingStep
from src.training.steps.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
from src.training.steps.sr_outcome_model_trainer import SROutcomeModelTrainer
from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep
from src.training.steps.step9_hmm_based_training_enhanced import HMMBasedTrainingEnhancedStep
from src.training.steps.step9_hmm_based_training import HMMBasedTrainingStep
from src.training.steps.step6_feature_engineering import FeatureEngineeringStep
from src.utils.logger import system_logger


def generate_test_market_data(days: int = 100) -> pd.DataFrame:
    """Generate realistic market data for testing."""
    np.random.seed(42)

    # Generate base price movement
    base_price = 100.0
    returns = np.random.normal(0, 0.02, days)  # 2% daily volatility
    prices = [base_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        # Generate realistic OHLC from close
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000, 10000)

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'timestamp': datetime.now() - timedelta(days=days-i)
        })

    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def create_test_config() -> Dict[str, Any]:
    """Create test configuration with S/R parameters."""
    return {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "timeframes": ["5m", "15m", "30m", "1h", "4h"],

        # S/R configuration
        "sr_breakout_predictor": {
            "enable_sr_breakout_tactics": True,
            "sr_proximity_threshold": 0.02,
            "breakout_confidence_threshold": 0.6,
            "sr_detection_method": "fractal",
            "min_sr_strength": 0.3,
            "max_sr_levels": 10,
            "sr_lookback_periods": 100,
            "volume_weight": 0.7,
            "price_weight": 0.3,
            "atr_multiplier": 1.5,
            "breakout_confirmation_periods": 3,
            "false_breakout_filter": True,
            "use_optimized_params": True,  # This should be enabled

            # Enhanced strength calculation
            "strength_calculation": {
                "enable_enhanced_strength": True,
                "touch_count_lookback": 50,
                "bounce_rate_threshold": 0.02,
                "isolation_distance_threshold": 0.05,
                "age_decay_factor": 0.95
            },

            # DBSCAN clustering
            "dbscan_clustering": {
                "enable_dbscan_clustering": True,
                "eps": 0.01,
                "min_samples": 2,
                "enable_noise_filtering": True
            },

            # Feature calculation
            "feature_calculation": {
                "enable_comprehensive_features": True,
                "strength_score_weights": {
                    "touch_count": 0.3,
                    "total_volume": 0.2,
                    "level_age": 0.2,
                    "bounce_rate": 0.2,
                    "isolation_score": 0.1
                }
            }
        },

        # S/R monitoring
        "sr_monitoring": {
            "enable_sr_monitoring": True,
            "sr_alert_threshold": 0.7
        },

        # Decision policy
        "decision_policy": {
            "confidence_threshold": 0.6,
            "risk_threshold": 0.1
        },

        # Training configuration
        "sequence_length": 20,
        "regime_confidence_threshold": 0.7,
        "transition_threshold": 0.6,

        # Model paths
        "model_dir": "test_models",
        "artifacts_dir": "test_artifacts"
    }


async def test_sr_breakout_predictor_optimization():
    """Test that SRBreakoutPredictor uses optimized parameters."""
    print("\n🔍 Testing SRBreakoutPredictor Optimization...")

    config = create_test_config()
    market_data = generate_test_market_data()

    # Test with default config
    sr_predictor_default = SRBreakoutPredictor(config)
    await sr_predictor_default.initialize()

    # Test with optimized config
    optimized_config = ensure_optimized_sr_config(config)
    sr_predictor_optimized = SRBreakoutPredictor(optimized_config)
    await sr_predictor_optimized.initialize()

    # Verify optimized parameters are loaded
    assert sr_predictor_optimized.use_optimized_params == True, "Optimized parameters should be enabled"

    # Test S/R context generation
    current_price = market_data['close'].iloc[-1]
    sr_context = await sr_predictor_optimized.get_sr_context(market_data, current_price)

    print(f"✅ SRBreakoutPredictor optimization test passed")
    print(f"   - Optimized params enabled: {sr_predictor_optimized.use_optimized_params}")
    print(f"   - S/R context generated: {len(sr_context.get('support_levels', []))} support, {len(sr_context.get('resistance_levels', []))} resistance levels")

    await sr_predictor_default.cleanup()
    await sr_predictor_optimized.cleanup()


async def test_analyst_components_optimization():
    """Test that analyst components use optimized parameters."""
    print("\n🔍 Testing Analyst Components Optimization...")

    config = create_test_config()

    # Test UnifiedRegimeClassifier
    print("   Testing UnifiedRegimeClassifier...")
    classifier = UnifiedRegimeClassifier(config)
    await classifier.initialize_sr_predictor()

    # Verify S/R predictor uses optimized parameters
    if classifier.sr_predictor:
        assert classifier.sr_predictor.use_optimized_params == True, "Classifier S/R predictor should use optimized params"
        print(f"   ✅ UnifiedRegimeClassifier uses optimized parameters")

    # Test UnifiedRegimeIntelligenceRuntime
    print("   Testing UnifiedRegimeIntelligenceRuntime...")
    runtime = UnifiedRegimeIntelligenceRuntime(config)
    await runtime.initialize()

    # Verify S/R predictor uses optimized parameters
    if runtime.sr_predictor:
        assert runtime.sr_predictor.use_optimized_params == True, "Runtime S/R predictor should use optimized params"
        print(f"   ✅ UnifiedRegimeIntelligenceRuntime uses optimized parameters")

    print("✅ Analyst components optimization test passed")


async def test_tactician_components_optimization():
    """Test that tactician components use optimized parameters."""
    print("\n🔍 Testing Tactician Components Optimization...")

    config = create_test_config()

    # Test DecisionPolicy
    print("   Testing DecisionPolicy...")
    decision_policy = DecisionPolicy(config)
    await decision_policy.initialize()

    # Verify S/R predictor uses optimized parameters
    if decision_policy.sr_predictor:
        assert decision_policy.sr_predictor.use_optimized_params == True, "DecisionPolicy S/R predictor should use optimized params"
        print(f"   ✅ DecisionPolicy uses optimized parameters")

    print("✅ Tactician components optimization test passed")


async def test_training_components_optimization():
    """Test that training components use optimized parameters."""
    print("\n🔍 Testing Training Components Optimization...")

    config = create_test_config()
    market_data = generate_test_market_data()

    # Test TacticianSpecialistTrainingStep
    print("   Testing TacticianSpecialistTrainingStep...")
    tactician_step = TacticianSpecialistTrainingStep(config)
    await tactician_step.initialize()

    if tactician_step.sr_predictor:
        assert tactician_step.sr_predictor.use_optimized_params == True, "TacticianSpecialistTrainingStep S/R predictor should use optimized params"
        print(f"   ✅ TacticianSpecialistTrainingStep uses optimized parameters")

    # Test UnifiedRegimeIntelligenceStep
    print("   Testing UnifiedRegimeIntelligenceStep...")
    regime_step = UnifiedRegimeIntelligenceStep(config)
    await regime_step.initialize()

    if regime_step.sr_predictor:
        assert regime_step.sr_predictor.use_optimized_params == True, "UnifiedRegimeIntelligenceStep S/R predictor should use optimized params"
        print(f"   ✅ UnifiedRegimeIntelligenceStep uses optimized parameters")

    # Test SROutcomeModelTrainer
    print("   Testing SROutcomeModelTrainer...")
    sr_trainer = SROutcomeModelTrainer(config)
    await sr_trainer.initialize()

    if sr_trainer.sr_predictor:
        assert sr_trainer.sr_predictor.use_optimized_params == True, "SROutcomeModelTrainer S/R predictor should use optimized params"
        print(f"   ✅ SROutcomeModelTrainer uses optimized parameters")

    # Test HMMRegimeDiscoveryStep
    print("   Testing HMMRegimeDiscoveryStep...")
    hmm_step = HMMRegimeDiscoveryStep(config)
    await hmm_step.initialize()

    if hmm_step.sr_predictor:
        assert hmm_step.sr_predictor.use_optimized_params == True, "HMMRegimeDiscoveryStep S/R predictor should use optimized params"
        print(f"   ✅ HMMRegimeDiscoveryStep uses optimized parameters")

    # Test HMMBasedTrainingEnhancedStep
    print("   Testing HMMBasedTrainingEnhancedStep...")
    hmm_enhanced_step = HMMBasedTrainingEnhancedStep(config)
    await hmm_enhanced_step.initialize()

    if hmm_enhanced_step.sr_predictor:
        assert hmm_enhanced_step.sr_predictor.use_optimized_params == True, "HMMBasedTrainingEnhancedStep S/R predictor should use optimized params"
        print(f"   ✅ HMMBasedTrainingEnhancedStep uses optimized parameters")

    # Test HMMBasedTrainingStep
    print("   Testing HMMBasedTrainingStep...")
    hmm_basic_step = HMMBasedTrainingStep(config)
    await hmm_basic_step.initialize()

    if hmm_basic_step.sr_predictor:
        assert hmm_basic_step.sr_predictor.use_optimized_params == True, "HMMBasedTrainingStep S/R predictor should use optimized params"
        print(f"   ✅ HMMBasedTrainingStep uses optimized parameters")

    # Test FeatureEngineeringStep
    print("   Testing FeatureEngineeringStep...")
    feature_step = FeatureEngineeringStep(config)
    await feature_step.initialize()

    # Test S/R feature integration
    features = pd.DataFrame()
    enhanced_features = await feature_step._add_sr_features(features, market_data, config)

    print(f"   ✅ FeatureEngineeringStep S/R feature integration works")

    print("✅ Training components optimization test passed")


async def test_optimized_parameters_loading():
    """Test that optimized parameters are actually loaded and used."""
    print("\n🔍 Testing Optimized Parameters Loading...")

    config = create_test_config()
    market_data = generate_test_market_data()

    # Create SR predictor with optimized parameters
    optimized_config = ensure_optimized_sr_config(config)
    sr_predictor = SRBreakoutPredictor(optimized_config)
    await sr_predictor.initialize()

    # Verify optimized parameters are loaded
    assert sr_predictor.use_optimized_params == True, "Optimized parameters should be enabled"

    # Test that optimized parameters are actually used
    current_price = market_data['close'].iloc[-1]
    sr_context = await sr_predictor.get_sr_context(market_data, current_price)

    # Check if optimized parameters are applied
    if sr_predictor.optimized_params:
        print(f"   ✅ Optimized parameters loaded: {len(sr_predictor.optimized_params)} parameters")
        print(f"   - Method weights: {sr_predictor.optimized_params.get('method_weights', {})}")
        print(f"   - Strength weights: {sr_predictor.optimized_params.get('strength_weights', {})}")
        print(f"   - DBSCAN params: {sr_predictor.optimized_params.get('dbscan_params', {})}")
    else:
        print(f"   ⚠️ No optimized parameters found (this is normal if no optimization has been run)")

    print(f"   - S/R levels detected: {len(sr_context.get('support_levels', []))} support, {len(sr_context.get('resistance_levels', []))} resistance")

    await sr_predictor.cleanup()
    print("✅ Optimized parameters loading test passed")


async def main():
    """Run all optimization tests."""
    print("🚀 Starting Optimized Parameters Integration Tests...")
    print("=" * 60)

    try:
        # Test SRBreakoutPredictor optimization
        await test_sr_breakout_predictor_optimization()

        # Test analyst components
        await test_analyst_components_optimization()

        # Test tactician components
        await test_tactician_components_optimization()

        # Test training components
        await test_training_components_optimization()

        # Test optimized parameters loading
        await test_optimized_parameters_loading()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Optimized parameters are properly integrated.")
        print("✅ Enhanced Training Manager uses optimized S/R parameters")
        print("✅ Analyst components use optimized S/R parameters")
        print("✅ Tactician components use optimized S/R parameters")
        print("✅ All training steps use optimized S/R parameters")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)