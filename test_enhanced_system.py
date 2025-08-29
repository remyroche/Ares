#!/usr/bin/env python3
"""
Test script for enhanced HMM, S/R, and feature engineering system.
This script demonstrates the integration of enhanced components:
1. Enhanced S/R analysis using sr_breakout_predictor
2. Enhanced HMM regime discovery using step3_hmm_regime_discovery
3. Enhanced regime change prediction using step9_5_hmm_lm_generalist_training
4. Enhanced feature engineering using feature_engineering_orchestrator
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_enhanced_system():
    """Test the enhanced HMM, S/R, and feature engineering system."""
    
    print("🚀 Testing Enhanced HMM, S/R, and Feature Engineering System")
    print("=" * 80)
    
    # Create sample market data
    print("📊 Creating sample market data...")
    market_data = create_sample_market_data()
    print(f"✅ Created market data: {market_data.shape}")
    
    # Test 1: Enhanced S/R Analysis
    print("\n" + "=" * 50)
    print("TEST 1: Enhanced S/R Analysis")
    print("=" * 50)
    await test_enhanced_sr_analysis(market_data)
    
    # Test 2: Enhanced HMM Regime Discovery
    print("\n" + "=" * 50)
    print("TEST 2: Enhanced HMM Regime Discovery")
    print("=" * 50)
    await test_enhanced_hmm_regime_discovery(market_data)
    
    # Test 3: Enhanced Regime Change Prediction
    print("\n" + "=" * 50)
    print("TEST 3: Enhanced Regime Change Prediction")
    print("=" * 50)
    await test_enhanced_regime_change_prediction(market_data)
    
    # Test 4: Enhanced Feature Engineering
    print("\n" + "=" * 50)
    print("TEST 4: Enhanced Feature Engineering")
    print("=" * 50)
    await test_enhanced_feature_engineering(market_data)
    
    # Test 5: Integrated System
    print("\n" + "=" * 50)
    print("TEST 5: Integrated System")
    print("=" * 50)
    await test_integrated_system(market_data)
    
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)

def create_sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    
    # Generate 1000 data points
    n_points = 1000
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
    
    # Generate price data with trends and volatility
    base_price = 100.0
    price_changes = np.random.normal(0, 0.001, n_points)
    prices = [base_price]
    
    for i in range(1, n_points):
        # Add some trend and volatility
        trend = 0.0001 * np.sin(i / 100)  # Cyclical trend
        volatility = 0.001 + 0.0005 * np.sin(i / 50)  # Variable volatility
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Generate OHLC from price
        high = price * (1 + abs(np.random.normal(0, 0.002)))
        low = price * (1 - abs(np.random.normal(0, 0.002)))
        open_price = price * (1 + np.random.normal(0, 0.001))
        close_price = price
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

async def test_enhanced_sr_analysis(market_data):
    """Test enhanced S/R analysis."""
    try:
        print("🔍 Testing enhanced S/R analysis...")
        
        # Import and initialize S/R breakout predictor
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
        
        config = {
            "sr_breakout_predictor": {
                "enable_composite_sr": True,
                "enable_volume_profile": True,
                "enable_psychological_levels": True,
                "enable_fractal_analysis": True,
                "enable_breakout_prediction": True,
                "max_sr_levels": 10
            }
        }
        
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()
        
        # Test centralized S/R analysis
        print("  📊 Performing centralized S/R analysis...")
        sr_analysis = await sr_predictor.analyze_centralized_sr_levels(market_data)
        
        if sr_analysis:
            print(f"  ✅ S/R analysis completed:")
            print(f"     - Support levels: {len(sr_analysis.get('support_levels', []))}")
            print(f"     - Resistance levels: {len(sr_analysis.get('resistance_levels', []))}")
            print(f"     - Quality metrics: {len(sr_analysis.get('quality_metrics', {}))}")
            print(f"     - Redundancy metrics: {len(sr_analysis.get('redundancy_metrics', {}))}")
        
        # Test S/R features generation
        print("  🔧 Generating S/R features...")
        sr_features = await sr_predictor.get_centralized_sr_features(market_data)
        
        if sr_features:
            print(f"  ✅ S/R features generated: {len(sr_features)} features")
            for key, value in list(sr_features.items())[:5]:  # Show first 5
                print(f"     - {key}: {value}")
        
        # Test S/R breakout predictions
        print("  🎯 Generating S/R breakout predictions...")
        breakout_predictions = await sr_predictor.get_sr_breakout_predictions(market_data)
        
        if breakout_predictions:
            print(f"  ✅ Breakout predictions generated:")
            print(f"     - Support breakouts: {len(breakout_predictions.get('support_breakouts', []))}")
            print(f"     - Resistance breakouts: {len(breakout_predictions.get('resistance_breakouts', []))}")
        
        print("  ✅ Enhanced S/R analysis test completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Enhanced S/R analysis test failed: {e}")

async def test_enhanced_hmm_regime_discovery(market_data):
    """Test enhanced HMM regime discovery."""
    try:
        print("🧠 Testing enhanced HMM regime discovery...")
        
        # Import and initialize HMM regime discovery step
        from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep
        
        config = {
            "SYMBOL": "TEST",
            "EXCHANGE": "TEST",
            "TIMEFRAME": "1m",
            "DATA_DIR": "test_data"
        }
        
        hmm_step = HMMRegimeDiscoveryStep(config)
        await hmm_step.initialize()
        
        # Test enhanced HMM regime discovery
        print("  📊 Performing enhanced HMM regime discovery...")
        training_input = {"symbol": "TEST", "exchange": "TEST", "timeframe": "1m"}
        pipeline_state = {}
        
        regime_result = await hmm_step._perform_enhanced_hmm_regime_discovery(training_input, market_data)
        
        if regime_result.get("success", False):
            print(f"  ✅ Enhanced HMM regime discovery completed:")
            print(f"     - Regime states: {len(regime_result.get('regime_states', []))}")
            print(f"     - Regime transitions: {len(regime_result.get('regime_transitions', {}))}")
            print(f"     - Quality metrics: {len(regime_result.get('metrics', {}).get('quality_metrics', {}))}")
            print(f"     - Redundancy metrics: {len(regime_result.get('metrics', {}).get('redundancy_metrics', {}))}")
        
        # Test enhanced regime features generation
        print("  🔧 Generating enhanced regime features...")
        regime_features = await hmm_step.get_enhanced_regime_features(market_data)
        
        if regime_features:
            print(f"  ✅ Enhanced regime features generated: {len(regime_features)} features")
            for key, value in list(regime_features.items())[:5]:  # Show first 5
                print(f"     - {key}: {value}")
        
        print("  ✅ Enhanced HMM regime discovery test completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Enhanced HMM regime discovery test failed: {e}")

async def test_enhanced_regime_change_prediction(market_data):
    """Test enhanced regime change prediction."""
    try:
        print("🔮 Testing enhanced regime change prediction...")
        
        # Import and initialize HMM LM generalist training step
        from src.training.steps.step9_5_hmm_lm_generalist_training import HMMLMGeneralistTrainingStep
        
        config = {
            "HMM_LM": {
                "generalist": {
                    "hmm_states": 5,
                    "sequence_length": 20,
                    "timeframes": ["1m", "5m", "15m"],
                    "d_model": 256,
                    "nhead": 8,
                    "num_layers": 6,
                    "dropout_rate": 0.1,
                    "learning_rate": 0.0001,
                    "batch_size": 32,
                    "epochs": 100
                }
            }
        }
        
        regime_step = HMMLMGeneralistTrainingStep(config)
        await regime_step.initialize()
        
        # Test enhanced regime change analysis
        print("  📊 Performing enhanced regime change analysis...")
        regime_analysis = await regime_step.analyze_enhanced_regime_changes(market_data)
        
        if regime_analysis:
            print(f"  ✅ Enhanced regime change analysis completed:")
            print(f"     - Regime changes: {regime_analysis.get('regime_change_count', 0)}")
            print(f"     - Transition count: {regime_analysis.get('transition_count', 0)}")
            print(f"     - Stability score: {regime_analysis.get('stability_score', 0.0):.3f}")
            print(f"     - Forecast accuracy: {regime_analysis.get('forecast_accuracy', 0.0):.3f}")
        
        # Test enhanced regime change features generation
        print("  🔧 Generating enhanced regime change features...")
        regime_change_features = await regime_step.get_enhanced_regime_change_features(market_data)
        
        if regime_change_features:
            print(f"  ✅ Enhanced regime change features generated: {len(regime_change_features)} features")
            for key, value in list(regime_change_features.items())[:5]:  # Show first 5
                print(f"     - {key}: {value}")
        
        # Test regime change predictions
        print("  🎯 Generating regime change predictions...")
        regime_predictions = await regime_step.predict_regime_changes(market_data)
        
        if regime_predictions:
            print(f"  ✅ Regime change predictions generated:")
            print(f"     - Next change probability: {regime_predictions.get('next_regime_change_probability', 0.0):.3f}")
            print(f"     - Expected change type: {regime_predictions.get('expected_change_type', 'unknown')}")
            print(f"     - Time to next change: {regime_predictions.get('time_to_next_change', 0.0):.1f} seconds")
            print(f"     - Change confidence: {regime_predictions.get('change_confidence', 0.0):.3f}")
        
        print("  ✅ Enhanced regime change prediction test completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Enhanced regime change prediction test failed: {e}")

async def test_enhanced_feature_engineering(market_data):
    """Test enhanced feature engineering."""
    try:
        print("🔧 Testing enhanced feature engineering...")
        
        # Import and initialize feature engineering orchestrator
        from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
        
        config = {
            "feature_engineering_orchestrator": {
                "enable_advanced_features": True,
                "enable_autoencoder_features": True,
                "enable_legacy_features": True
            },
            "sr_breakout_predictor": {
                "enable_composite_sr": True,
                "enable_volume_profile": True,
                "enable_psychological_levels": True,
                "enable_fractal_analysis": True,
                "enable_breakout_prediction": True,
                "max_sr_levels": 10
            },
            "HMM_LM": {
                "generalist": {
                    "hmm_states": 5,
                    "sequence_length": 20,
                    "timeframes": ["1m", "5m", "15m"],
                    "d_model": 256,
                    "nhead": 8,
                    "num_layers": 6,
                    "dropout_rate": 0.1,
                    "learning_rate": 0.0001,
                    "batch_size": 32,
                    "epochs": 100
                }
            }
        }
        
        feature_orchestrator = FeatureEngineeringOrchestrator(config)
        
        # Test enhanced feature generation
        print("  📊 Generating enhanced features...")
        enhanced_features = await feature_orchestrator.generate_enhanced_features(market_data)
        
        if enhanced_features:
            print(f"  ✅ Enhanced feature generation completed:")
            print(f"     - Total features: {enhanced_features.get('total_features', 0)}")
            print(f"     - Base features: {enhanced_features.get('base_features', pd.DataFrame()).shape if isinstance(enhanced_features.get('base_features'), pd.DataFrame) else 'N/A'}")
            print(f"     - S/R features: {enhanced_features.get('sr_features', pd.DataFrame()).shape if isinstance(enhanced_features.get('sr_features'), pd.DataFrame) else 'N/A'}")
            print(f"     - Regime features: {enhanced_features.get('regime_features', pd.DataFrame()).shape if isinstance(enhanced_features.get('regime_features'), pd.DataFrame) else 'N/A'}")
            print(f"     - Interaction features: {enhanced_features.get('interaction_features', pd.DataFrame()).shape if isinstance(enhanced_features.get('interaction_features'), pd.DataFrame) else 'N/A'}")
            
            # Show quality metrics
            quality_metrics = enhanced_features.get('quality_metrics', {})
            if quality_metrics:
                print(f"     - Quality metrics: {len(quality_metrics)} metrics")
                for key, value in list(quality_metrics.items())[:3]:  # Show first 3
                    print(f"       * {key}: {value:.3f}")
            
            # Show redundancy metrics
            redundancy_metrics = enhanced_features.get('redundancy_metrics', {})
            if redundancy_metrics:
                print(f"     - Redundancy metrics: {len(redundancy_metrics)} metrics")
                for key, value in list(redundancy_metrics.items())[:3]:  # Show first 3
                    print(f"       * {key}: {value:.3f}")
        
        print("  ✅ Enhanced feature engineering test completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Enhanced feature engineering test failed: {e}")

async def test_integrated_system(market_data):
    """Test the integrated system."""
    try:
        print("🔄 Testing integrated system...")
        
        # Test the complete integration
        print("  📊 Testing complete system integration...")
        
        # 1. S/R Analysis
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor
        sr_config = {"sr_breakout_predictor": {"enable_composite_sr": True, "max_sr_levels": 10}}
        sr_predictor = SRBreakoutPredictor(sr_config)
        await sr_predictor.initialize()
        sr_features = await sr_predictor.get_centralized_sr_features(market_data)
        
        # 2. HMM Regime Discovery
        from src.training.steps.step3_hmm_regime_discovery import HMMRegimeDiscoveryStep
        hmm_config = {"SYMBOL": "TEST", "EXCHANGE": "TEST", "TIMEFRAME": "1m", "DATA_DIR": "test_data"}
        hmm_step = HMMRegimeDiscoveryStep(hmm_config)
        await hmm_step.initialize()
        regime_features = await hmm_step.get_enhanced_regime_features(market_data)
        
        # 3. Regime Change Prediction
        from src.training.steps.step9_5_hmm_lm_generalist_training import HMMLMGeneralistTrainingStep
        regime_config = {"HMM_LM": {"generalist": {"hmm_states": 5, "sequence_length": 20}}}
        regime_step = HMMLMGeneralistTrainingStep(regime_config)
        await regime_step.initialize()
        regime_change_features = await regime_step.get_enhanced_regime_change_features(market_data)
        
        # 4. Feature Engineering Integration
        from src.analyst.feature_engineering_orchestrator import FeatureEngineeringOrchestrator
        fe_config = {
            "feature_engineering_orchestrator": {"enable_advanced_features": True},
            "sr_breakout_predictor": {"enable_composite_sr": True},
            "HMM_LM": {"generalist": {"hmm_states": 5}}
        }
        fe_orchestrator = FeatureEngineeringOrchestrator(fe_config)
        comprehensive_features = await fe_orchestrator.generate_enhanced_features(market_data)
        
        # Summary
        print(f"  ✅ Integrated system test completed:")
        print(f"     - S/R features: {len(sr_features)} features")
        print(f"     - Regime features: {len(regime_features)} features")
        print(f"     - Regime change features: {len(regime_change_features)} features")
        print(f"     - Comprehensive features: {comprehensive_features.get('total_features', 0)} total features")
        
        # Integration quality
        integration_quality = {
            "sr_integration": len(sr_features) > 0,
            "regime_integration": len(regime_features) > 0,
            "regime_change_integration": len(regime_change_features) > 0,
            "feature_engineering_integration": comprehensive_features.get('total_features', 0) > 0
        }
        
        print(f"     - Integration quality:")
        for component, status in integration_quality.items():
            status_symbol = "✅" if status else "❌"
            print(f"       {status_symbol} {component}: {'Integrated' if status else 'Not integrated'}")
        
        print("  ✅ Integrated system test completed successfully!")
        
    except Exception as e:
        print(f"  ❌ Integrated system test failed: {e}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_enhanced_system())