#!/usr/bin/env python3
"""
Test Scenario-Based Tactician Implementation

This script tests the complete implementation of the probabilistic scenario analysis
plan for the Tactician, including all configurable parameters for step17 optimization.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_market_data(
    start_date: str = "2024-01-01",
    periods: int = 1000,
    base_price: float = 100.0,
    volatility: float = 0.01,
    timeframe_minutes: int = 1
) -> pd.DataFrame:
    """Create realistic test market data for scenario testing."""
    dates = pd.date_range(start_date, periods=periods, freq=f"{timeframe_minutes}min")

    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible results

    # Create price series with trend and volatility
    returns = np.random.normal(0, volatility, periods)
    prices = [base_price]

    for i in range(1, periods):
        # Add some trend and mean reversion
        trend = 0.0001 * np.sin(i / 100)  # Small cyclical trend
        price_change = returns[i] + trend
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, periods)
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data

def create_test_config() -> Dict[str, Any]:
    """Create test configuration with scenario analysis parameters."""
    return {
        "step17_optimization": {
            "scenario_analysis": {
                # Scenario definitions
                "profit_zone_1_target": 0.005,
                "profit_zone_1_stop_loss": -0.005,
                "profit_zone_2_target": 0.01,
                "profit_zone_2_stop_loss": -0.005,
                "profit_zone_3_target": 0.015,
                "profit_zone_3_stop_loss": -0.005,
                "risk_zone_1_target": 0.005,
                "risk_zone_1_stop_loss": -0.005,
                "risk_zone_2_target": 0.01,
                "risk_zone_2_stop_loss": -0.005,
                "neutral_target": 0.0,
                "neutral_stop_loss": 0.0,

                # Time limit
                "time_limit_minutes": 30,

                # Model configuration
                "n_estimators": 50,  # Reduced for testing
                "learning_rate": 0.1,
                "max_depth": 4,      # Reduced for testing
                "num_leaves": 15,    # Reduced for testing
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,

                # Decision thresholds
                "profit_zone_combined_threshold": 0.6,
                "risk_zone_combined_threshold": 0.2,
                "exit_risk_threshold": 0.5,
                "neutral_threshold": 0.3,
                "confidence_threshold": 0.7,

                # Feature engineering
                "lookback_periods": 20,
                "volatility_window": 20,
                "rsi_period": 14,
                "ma_short_period": 5,
                "ma_long_period": 20,
                "volume_ma_period": 10
            },
            "ml_tactics": {
                "enable_ml_tactics": True,
                "confidence_threshold": 0.7,
                "regime_threshold": 0.6,
                "ml_weight": 0.8,
                "regime_weight": 0.2,
                "confidence_boost_factor": 1.2,
                "risk_adjustment_factor": 1.0,
                "fifty_percent_threshold": 0.75,
                "twenty_five_percent_threshold": 0.8,
                "combined_threshold": 0.7,
                "exit_fifty_percent_threshold": 0.4,
                "exit_twenty_five_percent_threshold": 0.35,
                "combined_exit_threshold": 0.45,
                "analyst_confidence_weight": 0.3,
                "fifty_percent_1m_weight": 0.25,
                "twenty_five_percent_1m_weight": 0.15,
                "fifty_percent_5m_weight": 0.2,
                "twenty_five_percent_5m_weight": 0.1
            }
        }
    }

async def test_scenario_based_predictor():
    """Test the ScenarioBasedPredictor class."""
    logger.info("🧪 Testing ScenarioBasedPredictor...")

    try:
        from src.tactician.scenario_based_predictor import ScenarioBasedPredictor

        # Create test configuration
        config = create_test_config()

        # Initialize predictor
        predictor = ScenarioBasedPredictor(config)
        success = await predictor.initialize()

        assert success, "ScenarioBasedPredictor initialization failed"
        logger.info("✅ ScenarioBasedPredictor initialized successfully")

        # Test configuration validation
        config_summary = predictor.get_configuration_summary()
        assert "scenarios" in config_summary, "Configuration summary missing scenarios"
        assert "decision_thresholds" in config_summary, "Configuration summary missing thresholds"
        assert len(config_summary["scenarios"]) == 6, f"Expected 6 scenarios, got {len(config_summary['scenarios'])}"

        logger.info("✅ Configuration validation passed")

        # Test feature extraction
        market_data = create_test_market_data(periods=100)
        features = predictor.extract_features(market_data)

        assert len(features) == 15, f"Expected 15 features, got {len(features)}"
        assert not np.any(np.isnan(features)), "Features contain NaN values"
        assert not np.any(np.isinf(features)), "Features contain infinite values"

        logger.info("✅ Feature extraction passed")

        # Test scenario labeling
        X = np.random.randn(50, 15)  # 50 samples, 15 features
        scenario_labels = predictor.prepare_scenario_targets(X, market_data.iloc[:50])

        assert len(scenario_labels) == 50, f"Expected 50 labels, got {len(scenario_labels)}"
        assert all(0 <= label <= 5 for label in scenario_labels), "Invalid scenario labels"

        logger.info("✅ Scenario labeling passed")

        # Test model training
        success = await predictor.train_model(X, scenario_labels, market_data=market_data.iloc[:50])

        assert success, "Model training failed"
        assert predictor.is_trained, "Model not marked as trained"
        assert predictor.last_training_time is not None, "Training time not recorded"

        logger.info("✅ Model training passed")

        # Test scenario prediction
        test_features = np.random.randn(1, 15)
        predictions = await predictor.predict_scenarios(test_features, market_data.iloc[-20:])

        assert "probabilities" in predictions, "Predictions missing probabilities"
        assert "predicted_scenario" in predictions, "Predictions missing scenario"
        assert "confidence" in predictions, "Predictions missing confidence"
        assert "scenario_analysis" in predictions, "Predictions missing analysis"

        # Validate probabilities
        probs = predictions["probabilities"]
        assert len(probs) == 6, f"Expected 6 probabilities, got {len(probs)}"
        assert abs(sum(probs.values()) - 1.0) < 0.01, "Probabilities don't sum to 1"

        # Validate scenario analysis
        analysis = predictions["scenario_analysis"]
        assert "profit_zone_probability" in analysis, "Analysis missing profit zone probability"
        assert "risk_zone_probability" in analysis, "Analysis missing risk zone probability"
        assert "dominant_zone" in analysis, "Analysis missing dominant zone"

        logger.info("✅ Scenario prediction passed")

        return True

    except Exception as e:
        logger.error(f"❌ ScenarioBasedPredictor test failed: {e}")
        return False

async def test_enhanced_ml_tactics_manager():
    """Test the enhanced MLTacticsManager with scenario analysis."""
    logger.info("🧪 Testing Enhanced MLTacticsManager...")

    try:
        from src.tactician.ml_tactics_manager import MLTacticsManager

        # Create test configuration
        config = create_test_config()

        # Initialize manager
        manager = MLTacticsManager(config)
        success = await manager.initialize()

        assert success, "MLTacticsManager initialization failed"
        assert manager.scenario_predictor is not None, "Scenario predictor not initialized"

        logger.info("✅ Enhanced MLTacticsManager initialized successfully")

        # Test enhanced predictions
        market_data = create_test_market_data(periods=200)
        analyst_barriers = {
            "upper_barrier": 0.02,  # 2% profit target
            "lower_barrier": -0.01  # 1% stop loss
        }

        enhanced_predictions = await manager.generate_enhanced_predictions(
            market_data=market_data,
            analyst_barriers=analyst_barriers,
            symbol="BTCUSDT",
            timeframe="1m",
            analyst_confidence=0.8
        )

        # Validate enhanced predictions structure
        assert "multi_output" in enhanced_predictions, "Missing multi_output predictions"
        assert "scenario_analysis" in enhanced_predictions, "Missing scenario analysis"
        assert "enhanced_decisions" in enhanced_predictions, "Missing enhanced decisions"
        assert "metadata" in enhanced_predictions, "Missing metadata"

        # Validate enhanced decisions
        decisions = enhanced_predictions["enhanced_decisions"]
        assert "entry_signal" in decisions, "Missing entry signal"
        assert "confidence" in decisions, "Missing confidence"
        assert "reasoning" in decisions, "Missing reasoning"
        assert "scenario_analysis" in decisions, "Missing scenario analysis details"

        # Validate scenario analysis details
        if decisions["scenario_analysis"]:
            scenario_details = decisions["scenario_analysis"]
            assert "profit_zone_probability" in scenario_details, "Missing profit zone probability"
            assert "risk_zone_probability" in scenario_details, "Missing risk zone probability"
            assert "predicted_scenario" in scenario_details, "Missing predicted scenario"
            assert "scenario_name" in scenario_details, "Missing scenario name"

        logger.info("✅ Enhanced predictions generation passed")

        # Test decision logic
        entry_signal = decisions["entry_signal"]
        confidence = decisions["confidence"]
        reasoning = decisions["reasoning"]

        logger.info(f"Entry Signal: {entry_signal}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Reasoning: {reasoning}")

        assert isinstance(entry_signal, bool), "Entry signal should be boolean"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        assert len(reasoning) > 0, "Reasoning should not be empty"

        logger.info("✅ Decision logic validation passed")

        return True

    except Exception as e:
        logger.error(f"❌ Enhanced MLTacticsManager test failed: {e}")
        return False

async def test_scenario_optimization_parameters():
    """Test that all scenario parameters are configurable for step17 optimization."""
    logger.info("🧪 Testing Scenario Optimization Parameters...")

    try:
        from src.tactician.scenario_based_predictor import ScenarioBasedPredictor

        # Test different parameter configurations
        test_configs = [
            {
                "step17_optimization": {
                    "scenario_analysis": {
                        "profit_zone_1_target": 0.003,  # Different target
                        "profit_zone_1_stop_loss": -0.003,  # Different stop loss
                        "time_limit_minutes": 15,  # Different time limit
                        "profit_zone_combined_threshold": 0.7,  # Different threshold
                        "risk_zone_combined_threshold": 0.15,  # Different threshold
                        "confidence_threshold": 0.8,  # Different threshold
                        "n_estimators": 200,  # Different model params
                        "max_depth": 8,
                        "learning_rate": 0.05
                    }
                }
            },
            {
                "step17_optimization": {
                    "scenario_analysis": {
                        "profit_zone_1_target": 0.008,  # Higher target
                        "profit_zone_1_stop_loss": -0.008,  # Higher stop loss
                        "time_limit_minutes": 45,  # Longer time limit
                        "profit_zone_combined_threshold": 0.5,  # Lower threshold
                        "risk_zone_combined_threshold": 0.25,  # Higher threshold
                        "confidence_threshold": 0.6,  # Lower threshold
                        "n_estimators": 75,  # Different model params
                        "max_depth": 3,
                        "learning_rate": 0.15
                    }
                }
            }
        ]

        for i, config in enumerate(test_configs):
            logger.info(f"Testing configuration {i+1}...")

            # Initialize predictor with different config
            predictor = ScenarioBasedPredictor(config)
            success = await predictor.initialize()

            assert success, f"Predictor initialization failed for config {i+1}"

            # Verify parameters are correctly loaded
            config_summary = predictor.get_configuration_summary()
            scenarios = config_summary["scenarios"]
            thresholds = config_summary["decision_thresholds"]
            model_config = config_summary["model_config"]

            # Check scenario parameters
            scenario_config = config["step17_optimization"]["scenario_analysis"]
            assert scenarios[0]["profit_target"] == scenario_config["profit_zone_1_target"], "Profit target not loaded correctly"
            assert scenarios[0]["stop_loss"] == scenario_config["profit_zone_1_stop_loss"], "Stop loss not loaded correctly"
            assert predictor.time_limit_minutes == scenario_config["time_limit_minutes"], "Time limit not loaded correctly"

            # Check thresholds
            assert thresholds["profit_zone_combined"] == scenario_config["profit_zone_combined_threshold"], "Profit threshold not loaded correctly"
            assert thresholds["risk_zone_combined"] == scenario_config["risk_zone_combined_threshold"], "Risk threshold not loaded correctly"
            assert thresholds["confidence_threshold"] == scenario_config["confidence_threshold"], "Confidence threshold not loaded correctly"

            # Check model parameters
            assert model_config["n_estimators"] == scenario_config["n_estimators"], "N estimators not loaded correctly"
            assert model_config["max_depth"] == scenario_config["max_depth"], "Max depth not loaded correctly"
            assert model_config["learning_rate"] == scenario_config["learning_rate"], "Learning rate not loaded correctly"

            logger.info(f"✅ Configuration {i+1} parameter loading passed")

        logger.info("✅ All optimization parameter tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Scenario optimization parameters test failed: {e}")
        return False

async def test_integration_with_existing_system():
    """Test integration with existing Tactician system."""
    logger.info("🧪 Testing Integration with Existing System...")

    try:
        from src.tactician.ml_tactics_manager import MLTacticsManager

        # Create configuration that includes both old and new systems
        config = create_test_config()

        # Add existing ML tactics configuration
        config["ml_tactics_manager"] = {
            "enable_ml_tactics": True,
            "confidence_threshold": 0.7,
            "regime_threshold": 0.6
        }

        # Initialize manager
        manager = MLTacticsManager(config)
        success = await manager.initialize()

        assert success, "Integration initialization failed"

        # Test that both systems work together
        market_data = create_test_market_data(periods=200)
        analyst_barriers = {
            "upper_barrier": 0.02,
            "lower_barrier": -0.01
        }

        # Test enhanced predictions (new system)
        enhanced_predictions = await manager.generate_enhanced_predictions(
            market_data=market_data,
            analyst_barriers=analyst_barriers,
            symbol="BTCUSDT",
            timeframe="1m",
            analyst_confidence=0.8
        )

        # Test original multi-output predictions (old system)
        multi_output_predictions = await manager.generate_multi_output_predictions(
            market_data=market_data,
            analyst_barriers=analyst_barriers,
            symbol="BTCUSDT",
            timeframe="1m",
            analyst_confidence=0.8
        )

        # Verify both systems work
        assert enhanced_predictions is not None, "Enhanced predictions failed"
        assert multi_output_predictions is not None, "Multi-output predictions failed"

        # Verify enhanced predictions include multi-output results
        assert enhanced_predictions["multi_output"] is not None, "Enhanced predictions missing multi-output"

        logger.info("✅ Integration with existing system passed")
        return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

async def test_error_handling_and_fallbacks():
    """Test error handling and fallback mechanisms."""
    logger.info("🧪 Testing Error Handling and Fallbacks...")

    try:
        from src.tactician.scenario_based_predictor import ScenarioBasedPredictor

        # Test with invalid configuration
        invalid_config = {
            "step17_optimization": {
                "scenario_analysis": {
                    "profit_zone_1_target": -0.001,  # Invalid negative target
                    "time_limit_minutes": -5,  # Invalid negative time
                    "profit_zone_combined_threshold": 1.5  # Invalid threshold > 1
                }
            }
        }

        predictor = ScenarioBasedPredictor(invalid_config)
        success = await predictor.initialize()

        # Should fail gracefully
        assert not success, "Should fail with invalid configuration"

        # Test fallback predictions
        valid_config = create_test_config()
        predictor = ScenarioBasedPredictor(valid_config)
        await predictor.initialize()

        # Test prediction without training (should use fallback)
        market_data = create_test_market_data(periods=50)
        features = predictor.extract_features(market_data)
        features = features.reshape(1, -1)

        predictions = await predictor.predict_scenarios(features, market_data)

        # Should return fallback predictions
        assert predictions is not None, "Fallback predictions failed"
        assert predictions["metadata"]["model_type"] == "scenario_based_fallback", "Should use fallback model"

        logger.info("✅ Error handling and fallbacks passed")
        return True

    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all tests."""
    logger.info("🚀 Starting Scenario-Based Tactician Implementation Tests")

    tests = [
        ("ScenarioBasedPredictor", test_scenario_based_predictor),
        ("Enhanced MLTacticsManager", test_enhanced_ml_tactics_manager),
        ("Optimization Parameters", test_scenario_optimization_parameters),
        ("Integration", test_integration_with_existing_system),
        ("Error Handling", test_error_handling_and_fallbacks)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*60}")

        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED - {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Scenario-based Tactician implementation is ready.")
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please review the implementation.")

    return passed == total

if __name__ == "__main__":
    asyncio.run(main())