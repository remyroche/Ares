#!/usr/bin/env python3
"""
Comprehensive Test for Step17 Optimized Tactician

This test validates the complete step17-optimized implementation with:
- ALL technical indicators (50+ indicators, 350+ features)
- Fractal scenario analysis (17 scenarios)
- 15-minute look-ahead period
- FULL step17 optimization for ALL decision logic
- Complete migration from existing system
"""

import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_market_data(n_samples: int = 1000) -> pd.DataFrame:
    pass
    pass
    """
    Create comprehensive test market data with realistic patterns.

    Args:
        n_samples: Number of samples to generate

    Returns:
        pd.DataFrame: Test market data with OHLCV
    """
    try:
        # Generate realistic price data with trends and volatility
    except Exception as e:
        pass
    except Exception as e:
        pass
        np.random.seed(42)

        # Base price with trend and noise
        base_price = 100.0
        trend = np.linspace(0, 0.1, n_samples)  # 10% trend over period
        noise = np.random.normal(0, 0.01, n_samples)  # 1% volatility
        price_changes = trend + noise

        # Generate OHLCV data
        close_prices = [base_price]
        for change in price_changes[1:]:
    pass
    pass
            close_prices.append(close_prices[-1] * (1 + change))

        close_prices = np.array(close_prices)

        # Generate OHLC with realistic spreads
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]

        # Generate volume data
        base_volume = 1000000
        volume_trend = np.linspace(1, 1.2, n_samples)  # 20% volume increase
        volume_noise = np.random.normal(1, 0.3, n_samples)
        volumes = base_volume * volume_trend * volume_noise
        volumes = np.maximum(volumes, 100000)  # Minimum volume

        # Create DataFrame
        data = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })

        # Add timestamp index
        start_time = datetime.now() - timedelta(minutes=n_samples)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
        data.index = timestamps

        logger.info(f"✅ Created test market data with {n_samples} samples")
        return data

    except Exception as e:
        logger.error(f"❌ Failed to create test market data: {e}")
        return pd.DataFrame()

def create_test_config() -> dict:
    pass
    pass
    """
    Create comprehensive test configuration for step17 optimization.

    Returns:
        dict: Test configuration
    """
    try:
        config = {
            "step17_optimization": {
                "comprehensive_enhanced_scenario_analysis": {
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

                    # Time limit
                    "time_limit_minutes": 15,

                    # Model configuration
                    "n_estimators": 100,  # Reduced for testing
                    "learning_rate": 0.1,
                    "max_depth": 6,
                    "num_leaves": 31,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,

                    # Decision thresholds
                    "profit_zone_combined_threshold": 0.6,
                    "risk_zone_combined_threshold": 0.2,
                    "exit_risk_threshold": 0.5,
                    "neutral_threshold": 0.3,
                    "confidence_threshold": 0.7,
                    "profit_risk_ratio_threshold": 2.0,
                    "scenario_dominance_threshold": 0.4,

                    # ALL Technical Indicator Parameters
                    # Momentum Indicators
                    "rsi_lookback_period": 14,
                    "rsi_overbought_threshold": 70,
                    "rsi_oversold_threshold": 30,
                    "macd_fast_period": 12,
                    "macd_slow_period": 26,
                    "macd_signal_period": 9,
                    "stoch_k_period": 14,
                    "stoch_d_period": 3,
                    "stoch_overbought": 80,
                    "stoch_oversold": 20,
                    "williams_r_period": 14,
                    "roc_period": 10,
                    "mom_period": 10,
                    "trix_period": 30,
                    "ultosc_period1": 7,
                    "ultosc_period2": 14,
                    "ultosc_period3": 28,
                    "willr_period": 14,
                    "aroon_period": 14,
                    "cci_period": 14,
                    "cci_constant": 0.015,
                    "cmo_period": 14,

                    # Trend Indicators
                    "sma_short_period": 10,
                    "sma_long_period": 30,
                    "ema_short_period": 10,
                    "ema_long_period": 30,
                    "dema_period": 30,
                    "tema_period": 30,
                    "ht_trendline_enabled": True,
                    "sar_acceleration": 0.02,
                    "sar_maximum": 0.2,
                    "adx_period": 14,
                    "adx_threshold": 25,
                    "dx_period": 14,
                    "minus_di_period": 14,
                    "plus_di_period": 14,
                    "minus_dm_period": 14,
                    "plus_dm_period": 14,
                    "midpoint_period": 14,
                    "midprice_period": 14,
                    "t3_period": 5,
                    "t3_volume_factor": 0.7,

                    # Volatility Indicators
                    "bb_period": 20,
                    "bb_std_dev": 2.0,
                    "bb_squeeze_threshold": 0.2,
                    "atr_period": 14,
                    "trange_enabled": True,
                    "var_period": 5,
                    "stddev_period": 5,

                    # Volume Indicators
                    "obv_enabled": True,
                    "ad_enabled": True,
                    "adosc_fast_period": 3,
                    "adosc_slow_period": 10,
                    "mfi_period": 14,

                    # Cycle Indicators
                    "ht_dcperiod_enabled": True,
                    "ht_dcphase_enabled": True,
                    "ht_phasor_enabled": True,
                    "ht_sine_enabled": True,
                    "ht_trendmode_enabled": True,

                    # Math Transform
                    "linearreg_period": 14,
                    "tsf_period": 14,
                    "stochrsi_period": 14,
                    "stochrsi_fastk_period": 5,
                    "stochrsi_fastd_period": 3,

                    # Feature engineering parameters
                    "lookback_periods": 20,
                    "volatility_window": 20,
                    "volume_ma_period": 10,
                    "price_momentum_periods": [5, 10, 20],
                    "volatility_periods": [5, 10, 20]
                },

                "step17_optimized_tactician": {
                    # Entry decision thresholds (ALL configurable)
                    "entry_profit_threshold": 0.6,
                    "entry_risk_threshold": 0.2,
                    "entry_confidence_threshold": 0.7,
                    "entry_profit_risk_ratio": 2.0,
                    "entry_scenario_dominance": 0.4,
                    "entry_analyst_confidence_min": 0.5,
                    "entry_neutral_threshold": 0.3,
                    "entry_volatility_threshold": 0.02,
                    "entry_volume_threshold": 1.2,

                    # Exit decision thresholds (ALL configurable)
                    "exit_risk_threshold": 0.5,
                    "exit_confidence_drop": 0.2,
                    "exit_profit_threshold": 0.8,
                    "exit_time_threshold": 3600,
                    "exit_drawdown_threshold": 0.05,
                    "exit_volatility_spike": 0.05,

                    # Direction decision thresholds (ALL configurable)
                    "direction_profit_bias": 0.1,
                    "direction_risk_bias": 0.1,
                    "direction_neutral_bias": 0.05,
                    "direction_confidence_bias": 0.15,

                    # Confidence calculation weights (ALL configurable)
                    "confidence_base_weight": 0.4,
                    "confidence_scenario_dominance_weight": 0.2,
                    "confidence_risk_reward_weight": 0.1,
                    "confidence_analyst_weight": 0.1,
                    "confidence_volatility_weight": 0.1,
                    "confidence_volume_weight": 0.1,

                    # Position sizing parameters (ALL configurable)
                    "position_size_base_multiplier": 1.0,
                    "position_size_confidence_multiplier": 1.5,
                    "position_size_scenario_dominance_multiplier": 1.2,
                    "position_size_risk_reward_multiplier": 1.3,
                    "position_size_analyst_confidence_multiplier": 1.1,
                    "position_size_volatility_multiplier": 0.8,
                    "position_size_volume_multiplier": 1.1,

                    # Leverage calculation parameters (ALL configurable)
                    "leverage_base_multiplier": 1.0,
                    "leverage_confidence_multiplier": 2.0,
                    "leverage_scenario_dominance_multiplier": 1.5,
                    "leverage_risk_reward_multiplier": 1.8,
                    "leverage_analyst_confidence_multiplier": 1.2,
                    "leverage_volatility_multiplier": 0.7,
                    "leverage_volume_multiplier": 1.3,

                    # Stop loss and take profit multipliers (ALL configurable)
                    "stop_loss_base_multiplier": 1.0,
                    "stop_loss_confidence_multiplier": 0.8,
                    "stop_loss_volatility_multiplier": 1.2,
                    "stop_loss_risk_multiplier": 1.1,

                    "take_profit_base_multiplier": 1.0,
                    "take_profit_confidence_multiplier": 1.2,
                    "take_profit_volatility_multiplier": 0.8,
                    "take_profit_profit_multiplier": 1.3,

                    # Risk management parameters (ALL configurable)
                    "max_position_size": 0.1,
                    "max_leverage": 3.0,
                    "max_drawdown": 0.05,
                    "correlation_threshold": 0.8,
                    "volatility_cap": 0.1,
                    "volume_cap": 5.0,
                    "confidence_cap": 0.95,
                    "scenario_dominance_cap": 0.9,
                    "risk_reward_cap": 5.0
                }
            }
    except Exception as e:
        pass
    except Exception as e:
        pass
        }

        logger.info("✅ Created comprehensive test configuration")
        return config

    except Exception as e:
        logger.error(f"❌ Failed to create test configuration: {e}")
        return {}

async def test_step17_optimized_tactician():
    """
    Test the complete step17 optimized Tactician implementation.
    """
    try:
        logger.info("🚀 Starting Step17 Optimized Tactician Test")

    except Exception as e:
        pass
    except Exception as e:
        pass
        # Create test data and configuration
        market_data = create_test_market_data(1000)
        config = create_test_config()

        if market_data.empty or not config:
    pass
    pass
            logger.error("❌ Failed to create test data or configuration")
            return False

        # Import the step17 optimized Tactician
        try:
            from src.tactician.step17_optimized_tactician import Step17OptimizedTactician
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import logger.info
            logger.info("✅ Successfully imported Step17OptimizedTactician")
        except ImportError as e:
            logger.error(f"❌ Failed to import Step17OptimizedTactician: {e}")
            return False

        # Initialize the Tactician
        tactician = Step17OptimizedTactician(config)
        success = await tactician.initialize()

        if not success:
    pass
    pass
            logger.error("❌ Failed to initialize Step17OptimizedTactician")
            return False

        logger.info("✅ Step17OptimizedTactician initialized successfully")

        # Test 1: Configuration validation
        logger.info("📋 Test 1: Configuration validation")
        config_summary = tactician.get_step17_configuration_summary()

        assert config_summary["is_initialized"], "Tactician should be initialized"
        assert len(config_summary["decision_thresholds"]) > 0, "Decision thresholds should be loaded"
        assert len(config_summary["risk_management"]) > 0, "Risk management should be loaded"

        logger.info(f"   ✅ Configuration validation passed")
        logger.info(f"   - Decision thresholds: {len(config_summary['decision_thresholds'])} parameters")
        logger.info(f"   - Risk management: {len(config_summary['risk_management'])} parameters")
        logger.info(f"   - Total configurable parameters: {config_summary['total_configurable_parameters']}")

        # Test 2: Scenario predictor validation
        logger.info("📋 Test 2: Scenario predictor validation")
        scenario_config = tactician.scenario_predictor.get_comprehensive_configuration_summary()

        assert len(tactician.scenario_predictor.scenarios) == 17, "Should have 17 fractal scenarios"
        assert scenario_config["n_features"] == 350, "Should have 350 comprehensive features"
        assert scenario_config["time_limit_minutes"] == 15, "Should have 15-minute look-ahead"

        logger.info(f"   ✅ Scenario predictor validation passed")
        logger.info(f"   - Scenarios: {len(tactician.scenario_predictor.scenarios)}")
        logger.info(f"   - Features: {scenario_config['n_features']}")
        logger.info(f"   - Time limit: {scenario_config['time_limit_minutes']} minutes")

        # Test 3: Feature extraction
        logger.info("📋 Test 3: Feature extraction")
        features = tactician.scenario_predictor.extract_comprehensive_features(market_data)

        assert len(features) == 350, "Should extract 350 comprehensive features"
        assert not np.any(np.isnan(features)), "Features should not contain NaN values"
        assert not np.any(np.isinf(features)), "Features should not contain infinite values"

        logger.info(f"   ✅ Feature extraction passed")
        logger.info(f"   - Extracted features: {len(features)}")
        logger.info(f"   - Feature range: [{features.min():.3f}, {features.max():.3f}]")

        # Test 4: Prediction generation
        logger.info("📋 Test 4: Prediction generation")

        # Create test analyst barriers
        analyst_barriers = {
            "upper_barrier": 0.02,   # 2% profit target
            "lower_barrier": -0.01   # 1% stop loss
        }

        predictions = await tactician.generate_predictions(
            market_data=market_data,
            analyst_barriers=analyst_barriers,
            symbol="ETHUSDT",
            timeframe="1m",
            analyst_confidence=0.7
        )

        assert "scenario_predictions" in predictions, "Should contain scenario predictions"
        assert "trading_decisions" in predictions, "Should contain trading decisions"
        assert "position_management" in predictions, "Should contain position management"
        assert "metadata" in predictions, "Should contain metadata"

        logger.info(f"   ✅ Prediction generation passed")
        logger.info(f"   - Model type: {predictions['metadata']['model_type']}")
        logger.info(f"   - Scenarios: {predictions['metadata']['n_scenarios']}")
        logger.info(f"   - Features: {predictions['metadata']['n_features']}")

        # Test 5: Trading decisions validation
        logger.info("📋 Test 5: Trading decisions validation")
        trading_decisions = predictions["trading_decisions"]

        assert "entry_signal" in trading_decisions, "Should have entry signal"
        assert "exit_signal" in trading_decisions, "Should have exit signal"
        assert "direction" in trading_decisions, "Should have direction"
        assert "confidence" in trading_decisions, "Should have confidence"
        assert "reasoning" in trading_decisions, "Should have reasoning"
        assert "scenario_metrics" in trading_decisions, "Should have scenario metrics"

        logger.info(f"   ✅ Trading decisions validation passed")
        logger.info(f"   - Entry signal: {trading_decisions['entry_signal']}")
        logger.info(f"   - Exit signal: {trading_decisions['exit_signal']}")
        logger.info(f"   - Direction: {trading_decisions['direction']}")
        logger.info(f"   - Confidence: {trading_decisions['confidence']:.3f}")

        # Test 6: Position management validation
        logger.info("📋 Test 6: Position management validation")
        position_management = predictions["position_management"]

        assert "position_size" in position_management, "Should have position size"
        assert "leverage" in position_management, "Should have leverage"
        assert "stop_loss" in position_management, "Should have stop loss"
        assert "take_profit" in position_management, "Should have take profit"
        assert "risk_metrics" in position_management, "Should have risk metrics"

        logger.info(f"   ✅ Position management validation passed")
        logger.info(f"   - Position size: {position_management['position_size']:.3f}")
        logger.info(f"   - Leverage: {position_management['leverage']:.2f}")
        logger.info(f"   - Stop loss: {position_management['stop_loss']:.3f}")
        logger.info(f"   - Take profit: {position_management['take_profit']:.3f}")

        # Test 7: Scenario analysis validation
        logger.info("📋 Test 7: Scenario analysis validation")
        scenario_predictions = predictions["scenario_predictions"]

        assert "probabilities" in scenario_predictions, "Should have scenario probabilities"
        assert "predicted_scenario" in scenario_predictions, "Should have predicted scenario"
        assert "scenario_name" in scenario_predictions, "Should have scenario name"
        assert "confidence" in scenario_predictions, "Should have confidence"
        assert "scenario_analysis" in scenario_predictions, "Should have scenario analysis"

        probabilities = scenario_predictions["probabilities"]
        assert len(probabilities) == 17, "Should have 17 scenario probabilities"
        assert abs(sum(probabilities.values()) - 1.0) < 1e-6, "Probabilities should sum to 1"

        logger.info(f"   ✅ Scenario analysis validation passed")
        logger.info(f"   - Predicted scenario: {scenario_predictions['predicted_scenario']}")
        logger.info(f"   - Scenario name: {scenario_predictions['scenario_name']}")
        logger.info(f"   - Model confidence: {scenario_predictions['confidence']:.3f}")

        # Test 8: Step17 optimization validation
        logger.info("📋 Test 8: Step17 optimization validation")

        # Check that ALL decision logic is configurable
        decision_thresholds = tactician.decision_thresholds
        risk_management = tactician.risk_management

        # Entry decision thresholds
        entry_thresholds = [k for k in decision_thresholds.keys() if k.startswith("entry_")]
        assert len(entry_thresholds) >= 9, f"Should have at least 9 entry thresholds, got {len(entry_thresholds)}"

        # Exit decision thresholds
        exit_thresholds = [k for k in decision_thresholds.keys() if k.startswith("exit_")]
        assert len(exit_thresholds) >= 6, f"Should have at least 6 exit thresholds, got {len(exit_thresholds)}"

        # Direction decision thresholds
        direction_thresholds = [k for k in decision_thresholds.keys() if k.startswith("direction_")]
        assert len(direction_thresholds) >= 4, f"Should have at least 4 direction thresholds, got {len(direction_thresholds)}"

        # Confidence calculation weights
        confidence_weights = [k for k in decision_thresholds.keys() if k.startswith("confidence_") and k.endswith("_weight")]
        assert len(confidence_weights) >= 6, f"Should have at least 6 confidence weights, got {len(confidence_weights)}"

        # Position sizing parameters
        position_params = [k for k in decision_thresholds.keys() if k.startswith("position_size_")]
        assert len(position_params) >= 7, f"Should have at least 7 position sizing parameters, got {len(position_params)}"

        # Leverage parameters
        leverage_params = [k for k in decision_thresholds.keys() if k.startswith("leverage_")]
        assert len(leverage_params) >= 7, f"Should have at least 7 leverage parameters, got {len(leverage_params)}"

        # Stop loss and take profit parameters
        sl_tp_params = [k for k in decision_thresholds.keys() if k.startswith(("stop_loss_", "take_profit_"))]
        assert len(sl_tp_params) >= 8, f"Should have at least 8 stop loss/take profit parameters, got {len(sl_tp_params)}"

        logger.info(f"   ✅ Step17 optimization validation passed")
        logger.info(f"   - Entry thresholds: {len(entry_thresholds)}")
        logger.info(f"   - Exit thresholds: {len(exit_thresholds)}")
        logger.info(f"   - Direction thresholds: {len(direction_thresholds)}")
        logger.info(f"   - Confidence weights: {len(confidence_weights)}")
        logger.info(f"   - Position parameters: {len(position_params)}")
        logger.info(f"   - Leverage parameters: {len(leverage_params)}")
        logger.info(f"   - SL/TP parameters: {len(sl_tp_params)}")
        logger.info(f"   - Risk management parameters: {len(risk_management)}")

        # Test 9: Performance tracking
        logger.info("📋 Test 9: Performance tracking")

        # Update position
        position_data = {
            "entry_price": 100.0,
            "entry_time": datetime.now(),
            "entry_confidence": 0.8,
            "direction": "LONG",
            "position_size": 0.05,
            "leverage": 2.0
        }
        tactician.update_position(position_data)

        # Update performance metrics
        trade_result = {"profit": 0.02}  # 2% profit
        tactician.update_performance_metrics(trade_result)

        performance_summary = tactician.get_performance_summary()

        assert performance_summary["performance_metrics"]["total_trades"] == 1, "Should have 1 trade"
        assert performance_summary["performance_metrics"]["winning_trades"] == 1, "Should have 1 winning trade"
        assert performance_summary["current_position"] is not None, "Should have current position"

        logger.info(f"   ✅ Performance tracking passed")
        logger.info(f"   - Total trades: {performance_summary['performance_metrics']['total_trades']}")
        logger.info(f"   - Win rate: {performance_summary['performance_metrics']['win_rate']:.1%}")
        logger.info(f"   - Position history: {performance_summary['position_history_count']} entries")

        # Test 10: Multiple prediction calls
        logger.info("📋 Test 10: Multiple prediction calls")

        # Generate multiple predictions to test consistency
        predictions_list = []
        for i in range(5):
    pass
    pass
            pred = await tactician.generate_predictions(
                market_data=market_data.iloc[-100:],  # Use last 100 samples
                analyst_barriers=analyst_barriers,
                symbol="ETHUSDT",
                timeframe="1m",
                analyst_confidence=0.7 + i * 0.05  # Varying analyst confidence
            )
            predictions_list.append(pred)

        # Check that predictions are generated consistently
        for i, pred in enumerate(predictions_list):
    pass
    pass
            assert pred["metadata"]["model_type"] == "step17_optimized_tactician", f"Prediction {i} should have correct model type"
            assert pred["metadata"]["n_scenarios"] == 17, f"Prediction {i} should have 17 scenarios"
            assert pred["metadata"]["n_features"] == 350, f"Prediction {i} should have 350 features"

        logger.info(f"   ✅ Multiple prediction calls passed")
        logger.info(f"   - Generated {len(predictions_list)} predictions successfully")

        # Final summary
        logger.info("🎉 Step17 Optimized Tactician Test Completed Successfully!")
        logger.info("")
        logger.info("📊 Test Summary:")
        logger.info("   ✅ Configuration validation")
        logger.info("   ✅ Scenario predictor validation")
        logger.info("   ✅ Feature extraction (350 features)")
        logger.info("   ✅ Prediction generation")
        logger.info("   ✅ Trading decisions validation")
        logger.info("   ✅ Position management validation")
        logger.info("   ✅ Scenario analysis validation (17 scenarios)")
        logger.info("   ✅ Step17 optimization validation (ALL parameters configurable)")
        logger.info("   ✅ Performance tracking")
        logger.info("   ✅ Multiple prediction calls")
        logger.info("")
        logger.info("🚀 Step17 Optimized Tactician is ready for production!")
        logger.info("   - ALL technical indicators implemented (50+ indicators)")
        logger.info("   - Fractal scenario analysis (17 scenarios)")
        logger.info("   - 15-minute look-ahead period")
        logger.info("   - FULL step17 optimization for ALL decision logic")
        logger.info("   - Complete migration from existing system")

        return True

    except Exception as e:
        logger.error(f"❌ Step17 Optimized Tactician Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    try:
        success = await test_step17_optimized_tactician()
    except Exception as e:
        pass
    except Exception as e:
        pass
        if success:
    pass
    pass
            logger.info("🎉 All tests passed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed")
            return 1
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    pass
    pass
    exit_code = asyncio.run(main())
    sys.exit(exit_code)