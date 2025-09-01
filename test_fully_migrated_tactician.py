#!/usr/bin/env python3
"""
Test Fully Migrated Tactician Implementation

This script tests the complete fully migrated Tactician implementation with:
- Enhanced scenario-based predictor with fractal scenarios
- All step7 technical indicators
- 15-minute look-ahead period
- Complete step17 optimization
- Full migration from old multi-output system
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
    """Create realistic test market data for enhanced scenario testing."""
    dates = pd.date_range(start_date, periods=periods, freq=f"{timeframe_minutes}min")

    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible results

    # Create price series with trend and volatility
    returns = np.random.normal(0, volatility, periods)
    prices = [base_price]

    for i in range(1, periods):
    pass
    pass
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

def create_enhanced_test_config() -> Dict[str, Any]:
    pass
    pass
    """Create enhanced test configuration with all step17 optimization parameters."""
    return {
        "step17_optimization": {
            "enhanced_scenario_analysis": {
                # Fractal scenario definitions
                "profit_zone_1_target": 0.0025,
                "profit_zone_1_stop_loss": -0.005,
                "profit_zone_2_target": 0.005,
                "profit_zone_2_stop_loss": -0.005,
                "profit_zone_3_target": 0.0075,
                "profit_zone_3_stop_loss": -0.005,
                "profit_zone_4_target": 0.01,
                "profit_zone_4_stop_loss": -0.005,
                "profit_zone_5_target": 0.0125,
                "profit_zone_5_stop_loss": -0.005,
                "profit_zone_6_target": 0.015,
                "profit_zone_6_stop_loss": -0.005,
                "profit_zone_7_target": 0.0175,
                "profit_zone_7_stop_loss": -0.005,
                "profit_zone_8_target": 0.02,
                "profit_zone_8_stop_loss": -0.005,

                "risk_zone_1_target": 0.005,
                "risk_zone_1_stop_loss": -0.0025,
                "risk_zone_2_target": 0.005,
                "risk_zone_2_stop_loss": -0.005,
                "risk_zone_3_target": 0.005,
                "risk_zone_3_stop_loss": -0.0075,
                "risk_zone_4_target": 0.005,
                "risk_zone_4_stop_loss": -0.01,
                "risk_zone_5_target": 0.005,
                "risk_zone_5_stop_loss": -0.0125,
                "risk_zone_6_target": 0.005,
                "risk_zone_6_stop_loss": -0.015,
                "risk_zone_7_target": 0.005,
                "risk_zone_7_stop_loss": -0.0175,
                "risk_zone_8_target": 0.005,
                "risk_zone_8_stop_loss": -0.02,

                "neutral_target": 0.0,
                "neutral_stop_loss": 0.0,

                # Time limit (15 minutes)
                "time_limit_minutes": 15,

                # Enhanced model configuration
                "n_estimators": 100,  # Reduced for testing
                "learning_rate": 0.1,
                "max_depth": 6,       # Reduced for testing
                "num_leaves": 31,     # Reduced for testing
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,

                # Enhanced decision thresholds
                "profit_zone_combined_threshold": 0.6,
                "risk_zone_combined_threshold": 0.2,
                "exit_risk_threshold": 0.5,
                "neutral_threshold": 0.3,
                "confidence_threshold": 0.7,
                "profit_risk_ratio_threshold": 2.0,
                "scenario_dominance_threshold": 0.4,

                # Step7 technical indicator parameters
                "rsi_lookback_period": 14,
                "rsi_overbought_threshold": 70,
                "rsi_oversold_threshold": 30,
                "macd_fast_period": 12,
                "macd_slow_period": 26,
                "macd_signal_period": 9,
                "bb_lookback_period": 20,
                "bb_std_dev": 2.0,
                "bb_squeeze_threshold": 0.2,
                "sma_short_period": 10,
                "sma_long_period": 30,
                "ema_short_period": 10,
                "ema_long_period": 30,
                "atr_lookback_period": 14,
                "stoch_k_period": 14,
                "stoch_d_period": 3,
                "stoch_overbought": 80,
                "stoch_oversold": 20,
                "adx_lookback_period": 14,
                "adx_threshold": 25,
                "cci_lookback_period": 14,
                "cci_constant": 0.015,

                # Enhanced feature engineering
                "lookback_periods": 20,
                "volatility_window": 20,
                "volume_ma_period": 10,
                "price_momentum_periods": [5, 10, 20],
                "volatility_periods": [5, 10, 20]
            },
            "fully_migrated_tactician": {
                # Decision thresholds
                "entry_profit_threshold": 0.6,
                "entry_risk_threshold": 0.2,
                "entry_confidence_threshold": 0.7,
                "entry_profit_risk_ratio": 2.0,
                "entry_scenario_dominance": 0.4,
                "exit_risk_threshold": 0.5,
                "exit_confidence_drop": 0.2,
                "position_size_multiplier": 1.0,
                "leverage_multiplier": 1.0,

                # Risk management
                "max_position_size": 0.1,
                "max_leverage": 3.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "max_drawdown": 0.05,
                "correlation_threshold": 0.8
            }
        }
    }

async def test_enhanced_scenario_predictor():
    """Test the EnhancedScenarioBasedPredictor with fractal scenarios."""
    logger.info("🧪 Testing EnhancedScenarioBasedPredictor...")

    try:
        from src.tactician.enhanced_scenario_based_predictor import EnhancedScenarioBasedPredictor

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        # Create enhanced test configuration
import config = create_enhanced_test_config
        config = create_enhanced_test_config()

        # Initialize predictor
        predictor = EnhancedScenarioBasedPredictor(config)
        success = await predictor.initialize()

        assert success, "EnhancedScenarioBasedPredictor initialization failed"
        logger.info("✅ EnhancedScenarioBasedPredictor initialized successfully")

        # Test fractal scenarios
        config_summary = predictor.get_enhanced_configuration_summary()
        assert "scenarios" in config_summary, "Configuration summary missing scenarios"
        assert len(config_summary["scenarios"]) == 17, f"Expected 17 scenarios, got {len(config_summary['scenarios'])}"

        # Verify fractal structure
        scenarios = config_summary["scenarios"]
        profit_scenarios = [s for s in scenarios.values() if s["zone_type"] == "profit"]
        risk_scenarios = [s for s in scenarios.values() if s["zone_type"] == "risk"]
        neutral_scenarios = [s for s in scenarios.values() if s["zone_type"] == "neutral"]

        assert len(profit_scenarios) == 8, f"Expected 8 profit scenarios, got {len(profit_scenarios)}"
        assert len(risk_scenarios) == 8, f"Expected 8 risk scenarios, got {len(risk_scenarios)}"
        assert len(neutral_scenarios) == 1, f"Expected 1 neutral scenario, got {len(neutral_scenarios)}"

        logger.info("✅ Fractal scenario structure validated")

        # Test comprehensive feature extraction
        market_data = create_test_market_data(periods=200)
        features = predictor.extract_comprehensive_features(market_data)

        assert len(features) == 150, f"Expected 150 features, got {len(features)}"
        assert not np.any(np.isnan(features)), "Features contain NaN values"
        assert not np.any(np.isinf(features)), "Features contain infinite values"

        logger.info("✅ Comprehensive feature extraction passed")

        # Test scenario labeling with 15-minute look-ahead
        X = np.random.randn(50, 150)  # 50 samples, 150 features
        scenario_labels = predictor.prepare_scenario_targets(X, market_data.iloc[:50])

        assert len(scenario_labels) == 50, f"Expected 50 labels, got {len(scenario_labels)}"
        assert all(0 <= label <= 16 for label in scenario_labels), "Invalid scenario labels"

        logger.info("✅ Scenario labeling with 15-minute look-ahead passed")

        # Test enhanced model training
        success = await predictor.train_model(X, scenario_labels, market_data=market_data.iloc[:50])

        assert success, "Enhanced model training failed"
        assert predictor.is_trained, "Enhanced model not marked as trained"
        assert predictor.last_training_time is not None, "Training time not recorded"

        logger.info("✅ Enhanced model training passed")

        # Test enhanced scenario prediction
        test_features = np.random.randn(1, 150)
        predictions = await predictor.predict_scenarios(test_features, market_data.iloc[-50:])

        assert "probabilities" in predictions, "Predictions missing probabilities"
        assert "predicted_scenario" in predictions, "Predictions missing scenario"
        assert "confidence" in predictions, "Predictions missing confidence"
        assert "scenario_analysis" in predictions, "Predictions missing analysis"

        # Validate enhanced probabilities
        probs = predictions["probabilities"]
        assert len(probs) == 17, f"Expected 17 probabilities, got {len(probs)}"
        assert abs(sum(probs.values()) - 1.0) < 0.01, "Probabilities don't sum to 1"

        # Validate enhanced scenario analysis
        analysis = predictions["scenario_analysis"]
        assert "profit_zone_probability" in analysis, "Analysis missing profit zone probability"
        assert "risk_zone_probability" in analysis, "Analysis missing risk zone probability"
        assert "dominant_zone" in analysis, "Analysis missing dominant zone"
        assert "scenario_dominance" in analysis, "Analysis missing scenario dominance"
        assert "zone_distribution" in analysis, "Analysis missing zone distribution"

        logger.info("✅ Enhanced scenario prediction passed")

        return True

    except Exception as e:
        logger.error(f"❌ EnhancedScenarioBasedPredictor test failed: {e}")
        return False

async def test_fully_migrated_tactician():
    """Test the FullyMigratedTactician."""
    logger.info("🧪 Testing FullyMigratedTactician...")

    try:
        from src.tactician.fully_migrated_tactician import FullyMigratedTactician

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        # Create enhanced test configuration
import config = create_enhanced_test_config
        config = create_enhanced_test_config()

        # Initialize fully migrated Tactician
        tactician = FullyMigratedTactician(config)
        success = await tactician.initialize()

        assert success, "FullyMigratedTactician initialization failed"
        assert tactician.scenario_predictor is not None, "Scenario predictor not initialized"

        logger.info("✅ FullyMigratedTactician initialized successfully")

        # Test enhanced predictions
        market_data = create_test_market_data(periods=300)
        analyst_barriers = {
            "upper_barrier": 0.02,  # 2% profit target
            "lower_barrier": -0.01  # 1% stop loss
        }

        predictions = await tactician.generate_predictions(
            market_data=market_data,
            analyst_barriers=analyst_barriers,
            symbol="BTCUSDT",
            timeframe="1m",
            analyst_confidence=0.8
        )

        # Validate enhanced predictions structure
        assert "scenario_predictions" in predictions, "Missing scenario predictions"
        assert "trading_decisions" in predictions, "Missing trading decisions"
        assert "position_management" in predictions, "Missing position management"
        assert "metadata" in predictions, "Missing metadata"

        # Validate trading decisions
        decisions = predictions["trading_decisions"]
        assert "entry_signal" in decisions, "Missing entry signal"
        assert "exit_signal" in decisions, "Missing exit signal"
        assert "direction" in decisions, "Missing direction"
        assert "confidence" in decisions, "Missing confidence"
        assert "reasoning" in decisions, "Missing reasoning"
        assert "scenario_metrics" in decisions, "Missing scenario metrics"

        # Validate position management
        position_mgmt = predictions["position_management"]
        assert "position_size" in position_mgmt, "Missing position size"
        assert "leverage" in position_mgmt, "Missing leverage"
        assert "stop_loss" in position_mgmt, "Missing stop loss"
        assert "take_profit" in position_mgmt, "Missing take profit"
        assert "risk_metrics" in position_mgmt, "Missing risk metrics"

        logger.info("✅ Enhanced predictions generation passed")

        # Test decision logic
        entry_signal = decisions["entry_signal"]
        exit_signal = decisions["exit_signal"]
        direction = decisions["direction"]
        confidence = decisions["confidence"]
        reasoning = decisions["reasoning"]

        logger.info(f"Entry Signal: {entry_signal}")
        logger.info(f"Exit Signal: {exit_signal}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Confidence: {confidence:.3f}")
        logger.info(f"Reasoning: {reasoning}")

        assert isinstance(entry_signal, bool), "Entry signal should be boolean"
        assert isinstance(exit_signal, bool), "Exit signal should be boolean"
        assert direction in ["LONG", "NEUTRAL", "EXIT"], "Invalid direction"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        assert len(reasoning) > 0, "Reasoning should not be empty"

        logger.info("✅ Decision logic validation passed")

        # Test position management
        position_size = position_mgmt["position_size"]
        leverage = position_mgmt["leverage"]
        stop_loss = position_mgmt["stop_loss"]
        take_profit = position_mgmt["take_profit"]

        assert 0.0 <= position_size <= 0.1, "Position size should be within limits"
        assert 1.0 <= leverage <= 3.0, "Leverage should be within limits"
        assert stop_loss < 0, "Stop loss should be negative"
        assert take_profit > 0, "Take profit should be positive"

        logger.info("✅ Position management validation passed")

        # Test performance tracking
        tactician.update_position({
            "symbol": "BTCUSDT",
            "side": "LONG",
            "size": 0.05,
            "entry_price": 100.0,
            "entry_confidence": 0.8
        })

        tactician.update_performance_metrics({
            "profit": 0.02,
            "duration": 300
        })

        performance = tactician.get_performance_summary()
        assert "performance_metrics" in performance, "Missing performance metrics"
        assert "current_position" in performance, "Missing current position"
        assert "scenario_predictor_status" in performance, "Missing scenario predictor status"

        logger.info("✅ Performance tracking passed")

        return True

    except Exception as e:
        logger.error(f"❌ FullyMigratedTactician test failed: {e}")
        return False

async def test_step17_optimization_parameters():
    """Test that all step17 optimization parameters are configurable."""
    logger.info("🧪 Testing Step17 Optimization Parameters...")

    try:
        from src.tactician.enhanced_scenario_based_predictor import EnhancedScenarioBasedPredictor
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        from src.tactician.fully_migrated_tactician import FullyMigratedTactician

        # Test different parameter configurations
import test_configs = [
        test_configs = [
            {
                "step17_optimization": {
                    "enhanced_scenario_analysis": {
                        "time_limit_minutes": 10,  # Different time limit
                        "profit_zone_1_target": 0.003,  # Different target
                        "profit_zone_1_stop_loss": -0.003,  # Different stop loss
                        "n_estimators": 150,  # Different model params
                        "max_depth": 10,
                        "learning_rate": 0.03,
                        "profit_zone_combined_threshold": 0.7,  # Different threshold
                        "risk_zone_combined_threshold": 0.15,  # Different threshold
                        "confidence_threshold": 0.8,  # Different threshold
                        "rsi_lookback_period": 21,  # Different technical indicator params
                        "macd_fast_period": 8,
                        "bb_std_dev": 2.5
                    },
                    "fully_migrated_tactician": {
                        "entry_profit_threshold": 0.7,  # Different decision threshold
                        "entry_risk_threshold": 0.15,  # Different decision threshold
                        "entry_confidence_threshold": 0.8,  # Different decision threshold
                        "max_position_size": 0.15,  # Different risk management
                        "max_leverage": 2.5,  # Different risk management
                        "stop_loss_multiplier": 1.2,  # Different risk management
                        "take_profit_multiplier": 1.2  # Different risk management
                    }
                }
            },
            {
                "step17_optimization": {
                    "enhanced_scenario_analysis": {
                        "time_limit_minutes": 20,  # Different time limit
                        "profit_zone_1_target": 0.002,  # Different target
                        "profit_zone_1_stop_loss": -0.008,  # Different stop loss
                        "n_estimators": 75,  # Different model params
                        "max_depth": 4,
                        "learning_rate": 0.15,
                        "profit_zone_combined_threshold": 0.5,  # Different threshold
                        "risk_zone_combined_threshold": 0.25,  # Different threshold
                        "confidence_threshold": 0.6,  # Different threshold
                        "rsi_lookback_period": 7,  # Different technical indicator params
                        "macd_fast_period": 16,
                        "bb_std_dev": 1.5
                    },
                    "fully_migrated_tactician": {
                        "entry_profit_threshold": 0.5,  # Different decision threshold
                        "entry_risk_threshold": 0.25,  # Different decision threshold
                        "entry_confidence_threshold": 0.6,  # Different decision threshold
                        "max_position_size": 0.05,  # Different risk management
                        "max_leverage": 4.0,  # Different risk management
                        "stop_loss_multiplier": 0.8,  # Different risk management
                        "take_profit_multiplier": 0.8  # Different risk management
                    }
                }
            }
        ]

        for i, config in enumerate(test_configs):
    pass
    pass
            logger.info(f"Testing configuration {i+1}...")

            # Test EnhancedScenarioBasedPredictor
            predictor = EnhancedScenarioBasedPredictor(config)
            success = await predictor.initialize()

            assert success, f"Predictor initialization failed for config {i+1}"

            # Verify parameters are correctly loaded
            config_summary = predictor.get_enhanced_configuration_summary()
            scenario_config = config["step17_optimization"]["enhanced_scenario_analysis"]

            assert predictor.time_limit_minutes == scenario_config["time_limit_minutes"], "Time limit not loaded correctly"
            assert predictor.scenarios[0]["profit_target"] == scenario_config["profit_zone_1_target"], "Profit target not loaded correctly"
            assert predictor.scenarios[0]["stop_loss"] == scenario_config["profit_zone_1_stop_loss"], "Stop loss not loaded correctly"
            assert predictor.model_config["n_estimators"] == scenario_config["n_estimators"], "N estimators not loaded correctly"
            assert predictor.model_config["max_depth"] == scenario_config["max_depth"], "Max depth not loaded correctly"
            assert predictor.model_config["learning_rate"] == scenario_config["learning_rate"], "Learning rate not loaded correctly"
            assert predictor.decision_thresholds["profit_zone_combined"] == scenario_config["profit_zone_combined_threshold"], "Profit threshold not loaded correctly"
            assert predictor.decision_thresholds["risk_zone_combined"] == scenario_config["risk_zone_combined_threshold"], "Risk threshold not loaded correctly"
            assert predictor.decision_thresholds["confidence_threshold"] == scenario_config["confidence_threshold"], "Confidence threshold not loaded correctly"
            assert predictor.technical_indicators["RSI"]["lookback_period"] == scenario_config["rsi_lookback_period"], "RSI lookback not loaded correctly"
            assert predictor.technical_indicators["MACD"]["fast_period"] == scenario_config["macd_fast_period"], "MACD fast period not loaded correctly"
            assert predictor.technical_indicators["Bollinger_Bands"]["std_dev"] == scenario_config["bb_std_dev"], "BB std dev not loaded correctly"

            # Test FullyMigratedTactician
            tactician = FullyMigratedTactician(config)
            success = await tactician.initialize()

            assert success, f"Tactician initialization failed for config {i+1}"

            # Verify parameters are correctly loaded
            tactician_config = config["step17_optimization"]["fully_migrated_tactician"]

            assert tactician.decision_thresholds["entry_profit_threshold"] == tactician_config["entry_profit_threshold"], "Entry profit threshold not loaded correctly"
            assert tactician.decision_thresholds["entry_risk_threshold"] == tactician_config["entry_risk_threshold"], "Entry risk threshold not loaded correctly"
            assert tactician.decision_thresholds["entry_confidence_threshold"] == tactician_config["entry_confidence_threshold"], "Entry confidence threshold not loaded correctly"
            assert tactician.risk_management["max_position_size"] == tactician_config["max_position_size"], "Max position size not loaded correctly"
            assert tactician.risk_management["max_leverage"] == tactician_config["max_leverage"], "Max leverage not loaded correctly"
            assert tactician.risk_management["stop_loss_multiplier"] == tactician_config["stop_loss_multiplier"], "Stop loss multiplier not loaded correctly"
            assert tactician.risk_management["take_profit_multiplier"] == tactician_config["take_profit_multiplier"], "Take profit multiplier not loaded correctly"

            logger.info(f"✅ Configuration {i+1} parameter loading passed")

        logger.info("✅ All step17 optimization parameter tests passed")
        return True

    except Exception as e:
        logger.error(f"❌ Step17 optimization parameters test failed: {e}")
        return False

async def test_full_migration_compatibility():
    """Test full migration compatibility and performance."""
    logger.info("🧪 Testing Full Migration Compatibility...")

    try:
        from src.tactician.fully_migrated_tactician import FullyMigratedTactician

    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
        # Create enhanced test configuration
import config = create_enhanced_test_config
        config = create_enhanced_test_config()

        # Initialize fully migrated Tactician
        tactician = FullyMigratedTactician(config)
        success = await tactician.initialize()

        assert success, "Full migration initialization failed"

        # Test multiple prediction cycles
        market_data = create_test_market_data(periods=500)
        analyst_barriers = {
            "upper_barrier": 0.02,
            "lower_barrier": -0.01
        }

        results = []
        for i in range(10):  # Test 10 prediction cycles
            subset_data = market_data.iloc[i*50:(i+1)*50]
            if len(subset_data) >= 50:
    pass
    pass
                prediction = await tactician.generate_predictions(
                    market_data=subset_data,
                    analyst_barriers=analyst_barriers,
                    symbol="BTCUSDT",
                    timeframe="1m",
                    analyst_confidence=0.7 + (i * 0.02)  # Varying analyst confidence
                )
                results.append(prediction)

        # Verify all predictions have correct structure
        for i, result in enumerate(results):
    pass
    pass
            assert "scenario_predictions" in result, f"Missing scenario predictions in result {i}"
            assert "trading_decisions" in result, f"Missing trading decisions in result {i}"
            assert "position_management" in result, f"Missing position management in result {i}"
            assert "metadata" in result, f"Missing metadata in result {i}"

            # Verify scenario predictions
            scenario_preds = result["scenario_predictions"]
            assert "probabilities" in scenario_preds, f"Missing probabilities in result {i}"
            assert len(scenario_preds["probabilities"]) == 17, f"Wrong number of probabilities in result {i}"

            # Verify trading decisions
            decisions = result["trading_decisions"]
            assert "entry_signal" in decisions, f"Missing entry signal in result {i}"
            assert "exit_signal" in decisions, f"Missing exit signal in result {i}"
            assert "direction" in decisions, f"Missing direction in result {i}"
            assert "confidence" in decisions, f"Missing confidence in result {i}"

            # Verify position management
            position_mgmt = result["position_management"]
            assert "position_size" in position_mgmt, f"Missing position size in result {i}"
            assert "leverage" in position_mgmt, f"Missing leverage in result {i}"
            assert "stop_loss" in position_mgmt, f"Missing stop loss in result {i}"
            assert "take_profit" in position_mgmt, f"Missing take profit in result {i}"

        logger.info("✅ Full migration compatibility passed")

        # Test configuration summary
        config_summary = tactician.get_configuration_summary()
        assert "decision_thresholds" in config_summary, "Missing decision thresholds in config summary"
        assert "risk_management" in config_summary, "Missing risk management in config summary"
        assert "scenario_predictor_config" in config_summary, "Missing scenario predictor config in config summary"
        assert "is_initialized" in config_summary, "Missing initialization status in config summary"

        logger.info("✅ Configuration summary validation passed")

        return True

    except Exception as e:
        logger.error(f"❌ Full migration compatibility test failed: {e}")
        return False

async def main():
    """Run all enhanced tests."""
    logger.info("🚀 Starting Fully Migrated Tactician Implementation Tests")

    tests = [
        ("EnhancedScenarioBasedPredictor", test_enhanced_scenario_predictor),
        ("FullyMigratedTactician", test_fully_migrated_tactician),
        ("Step17 Optimization Parameters", test_step17_optimization_parameters),
        ("Full Migration Compatibility", test_full_migration_compatibility)
    ]

    results = {}

    for test_name, test_func in tests:
    pass
    pass
        logger.info(f"\\\n{'='*60}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*60}")

        try:
            result = await test_func()
    except Exception as e:
        pass
    except Exception as e:
        pass
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: ❌ FAILED - {e}")

    # Summary
    logger.info(f"\\\n{'='*60}")
    logger.info("ENHANCED TEST SUMMARY")
    logger.info(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
    pass
    pass
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\\\nOverall: {passed}/{total} tests passed")

    if passed == total:
    pass
    pass
        logger.info("🎉 All enhanced tests passed! Fully migrated Tactician implementation is ready.")
        logger.info("📊 Key Features Implemented:")
        logger.info("   ✅ Fractal scenarios (17 scenarios: 8 profit, 8 risk, 1 neutral)")
        logger.info("   ✅ All step7 technical indicators (RSI, MACD, BB, SMA, EMA, ATR, Stochastic, ADX, CCI)")
        logger.info("   ✅ 15-minute look-ahead period")
        logger.info("   ✅ Complete step17 optimization for all parameters")
        logger.info("   ✅ Full migration from old multi-output system")
        logger.info("   ✅ Enhanced decision logic with scenario dominance")
        logger.info("   ✅ Comprehensive position management")
        logger.info("   ✅ Performance tracking and risk management")
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Please review the implementation.")

    return passed == total

if __name__ == "__main__":
    pass
    pass
    asyncio.run(main())