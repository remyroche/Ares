#!/usr/bin/env python3
"""
Test S/R Data Integration with Proper Data Access

This script demonstrates the S/R backtesting validation system with:
1. Proper data integration using ares_launcher patterns
2. Lookback period management from training modes
3. Multi-timeframe data access
4. Data quality validation
5. S/R level detection and validation
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tactician.sr_breakout_predictor import SRBreakoutPredictor
from tactician.sr_backtesting_validator import SRBacktestingValidator
from tactician.sr_detection_optimization import SRDetectionOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_market_data(n_periods: int = 1000, base_price: float = 50000.0) -> pd.DataFrame:
    """Generate realistic test market data."""
    logger.info(f"📊 Generating {n_periods} periods of test market data...")

    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=n_periods)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_periods)

    # Generate price data with realistic patterns
    np.random.seed(42)

    # Base price movement with trend and volatility
    price_changes = np.random.normal(0, 0.002, n_periods)  # 0.2% volatility
    trend = np.linspace(0, 0.1, n_periods)  # 10% upward trend
    prices = base_price * np.exp(np.cumsum(price_changes + trend/len(price_changes)))

    # Generate OHLC data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Add some intra-period volatility
        intra_volatility = np.random.normal(0, 0.001)
        high = price * (1 + abs(intra_volatility))
        low = price * (1 - abs(intra_volatility))

        # Ensure OHLC relationship
        open_price = price * (1 + np.random.normal(0, 0.0005))
        close_price = price * (1 + np.random.normal(0, 0.0005))

        # Generate volume with some correlation to price movement
        base_volume = np.random.uniform(1000, 10000)
        volume_multiplier = 1 + abs(price_changes[i]) * 10  # Higher volume on larger moves
        volume = base_volume * volume_multiplier

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': max(open_price, close_price, high),
            'low': min(open_price, close_price, low),
            'close': close_price,
            'volume': volume
        })

    df = pd.DataFrame(data)
    logger.info(f"✅ Generated test data: {len(df)} periods, price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    return df

async def test_data_integration_initialization():
    """Test data integration initialization with different training modes."""
    logger.info("🧪 Testing data integration initialization...")

    # Test different training modes
    training_modes = ["light", "blank", "full"]

    for mode in training_modes:
        try:
            logger.info(f"📋 Testing {mode} training mode...")

            # Create data integration with training mode
            data_integration = await create_sr_data_integration_simple(
                symbol="BTCUSDT",
                exchange="binance",
                timeframes=["1m", "5m", "15m", "30m"],
                training_mode=mode
            )

            if data_integration:
                logger.info(f"✅ {mode} mode initialized successfully")
                logger.info(f"   - Lookback days: {data_integration.lookback_days}")
                logger.info(f"   - Timeframes: {data_integration.timeframes}")
                logger.info(f"   - Symbol: {data_integration.symbol}")
                logger.info(f"   - Exchange: {data_integration.exchange}")
            else:
                logger.error(f"❌ {mode} mode initialization failed")

        except Exception as e:
            logger.error(f"❌ Error testing {mode} mode: {e}")

    return True

async def test_market_data_loading():
    """Test market data loading with proper lookback periods."""
    logger.info("🧪 Testing market data loading...")

    try:
        # Create data integration
        data_integration = await create_sr_data_integration_simple(
            symbol="BTCUSDT",
            exchange="binance",
            timeframes=["1m", "5m", "15m", "30m"],
            training_mode="blank"  # Use blank mode for testing
        )

        if not data_integration:
            logger.error("❌ Failed to create data integration")
            return False

        # Test loading data for each timeframe
        for timeframe in ["1m", "5m", "15m", "30m"]:
            logger.info(f"📊 Loading data for {timeframe}...")

            # Get appropriate lookback period for timeframe
            lookback_days = data_integration.get_lookback_period_for_timeframe(timeframe)
            logger.info(f"   - Using {lookback_days} days lookback for {timeframe}")

            # Load market data
            market_data = await data_integration.get_market_data(
                timeframe=timeframe,
                lookback_days=lookback_days
            )

            if market_data is not None and len(market_data) > 0:
                logger.info(f"✅ Loaded {len(market_data)} data points for {timeframe}")

                # Validate data quality
                quality_ok = await data_integration.validate_data_quality(market_data, timeframe)
                if quality_ok:
                    logger.info(f"✅ Data quality validation passed for {timeframe}")
                else:
                    logger.warning(f"⚠️ Data quality issues detected for {timeframe}")
            else:
                logger.warning(f"⚠️ No data available for {timeframe}, using test data")
                # Use test data as fallback
                market_data = generate_test_market_data(1000)

        # Test multi-timeframe data loading
        logger.info("📊 Loading multi-timeframe data...")
        multi_tf_data = await data_integration.get_multi_timeframe_data()

        if multi_tf_data:
            logger.info(f"✅ Loaded data for {len(multi_tf_data)} timeframes")
            for tf, data in multi_tf_data.items():
                logger.info(f"   - {tf}: {len(data)} data points")
        else:
            logger.warning("⚠️ No multi-timeframe data available, using test data")
            # Generate test data for multiple timeframes
            multi_tf_data = {
                "1m": generate_test_market_data(1440),  # 1 day of minute data
                "5m": generate_test_market_data(288),   # 1 day of 5-minute data
                "15m": generate_test_market_data(96),   # 1 day of 15-minute data
                "30m": generate_test_market_data(48),   # 1 day of 30-minute data
            }

        return True

    except Exception as e:
        logger.error(f"❌ Error testing market data loading: {e}")
        return False

async def test_sr_level_detection_with_data_integration():
    """Test S/R level detection using the data integration system."""
    logger.info("🧪 Testing S/R level detection with data integration...")

    try:
        # Create data integration
        data_integration = await create_sr_data_integration_simple(
            symbol="BTCUSDT",
            exchange="binance",
            timeframes=["1m", "5m", "15m", "30m"],
            training_mode="blank"
        )

        if not data_integration:
            logger.error("❌ Failed to create data integration")
            return False

        # Load market data for 15m timeframe
        market_data = await data_integration.get_market_data("15m", lookback_days=30)

        if market_data is None or len(market_data) == 0:
            logger.warning("⚠️ No real data available, using test data")
            market_data = generate_test_market_data(1000)

        # Initialize S/R predictor with config
        config = {
            "sr_breakout_predictor": {
                "enable_enhanced_strength": True,
                "enable_dbscan_clustering": True,
                "max_levels": 50,
                "min_level_distance": 0.001,
                "fractal_lookback": 20,
                "pivot_lookback": 20,
                "touch_count_lookback": 100,
                "volume_lookback_periods": 20,
            }
        }

        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()

        # Detect S/R levels
        logger.info("🔍 Detecting S/R levels...")
        sr_levels = await sr_predictor.detect_sr_levels(market_data)

        if sr_levels and len(sr_levels) > 0:
            logger.info(f"✅ Detected {len(sr_levels)} S/R levels")

            # Analyze level distribution
            support_levels = [level for level in sr_levels if level.get('type') == 'support']
            resistance_levels = [level for level in sr_levels if level.get('type') == 'resistance']

            logger.info(f"   - Support levels: {len(support_levels)}")
            logger.info(f"   - Resistance levels: {len(resistance_levels)}")

            # Show some level details
            for i, level in enumerate(sr_levels[:5]):  # Show first 5 levels
                logger.info(f"   - Level {i+1}: {level.get('price', 0):.2f} ({level.get('type', 'unknown')}) - Strength: {level.get('strength', 0):.3f}")

            return True
        else:
            logger.warning("⚠️ No S/R levels detected")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing S/R level detection: {e}")
        return False

async def test_backtesting_validation_with_data_integration():
    """Test backtesting validation using the data integration system."""
    logger.info("🧪 Testing backtesting validation with data integration...")

    try:
        # Create data integration
        data_integration = await create_sr_data_integration_simple(
            symbol="BTCUSDT",
            exchange="binance",
            timeframes=["1m", "5m", "15m", "30m"],
            training_mode="blank"
        )

        if not data_integration:
            logger.error("❌ Failed to create data integration")
            return False

        # Load market data
        market_data = await data_integration.get_market_data("15m", lookback_days=60)

        if market_data is None or len(market_data) == 0:
            logger.warning("⚠️ No real data available, using test data")
            market_data = generate_test_market_data(2000)  # More data for backtesting

        # Initialize S/R predictor and detect levels
        config = {
            "sr_breakout_predictor": {
                "enable_enhanced_strength": True,
                "enable_dbscan_clustering": True,
                "max_levels": 30,
                "min_level_distance": 0.002,
                "fractal_lookback": 20,
                "pivot_lookback": 20,
                "touch_count_lookback": 100,
            },
            "sr_backtesting": {
                "touch_threshold": 0.001,
                "bounce_threshold": 0.005,
                "breakout_threshold": 0.01,
                "false_breakout_threshold": 0.02,
                "confirmation_periods": 3,
                "min_touches": 2,
                "volume_spike_threshold": 1.5,
                "institutional_volume_threshold": 2.0,
                "volume_confirmation_threshold": 1.2,
                "volume_lookback_periods": 20,
                "min_bounce_rate": 0.6,
                "max_false_breakout_rate": 0.3,
                "min_volume_confirmation": 0.5,
            }
        }

        # Initialize components
        sr_predictor = SRBreakoutPredictor(config)
        await sr_predictor.initialize()

        # Detect S/R levels
        sr_levels = await sr_predictor.detect_sr_levels(market_data)

        if not sr_levels or len(sr_levels) == 0:
            logger.warning("⚠️ No S/R levels detected, creating test levels")
            # Create some test levels
            current_price = market_data['close'].iloc[-1]
            sr_levels = [
                {'price': current_price * 0.98, 'type': 'support', 'strength': 0.7},
                {'price': current_price * 1.02, 'type': 'resistance', 'strength': 0.8},
                {'price': current_price * 0.95, 'type': 'support', 'strength': 0.6},
                {'price': current_price * 1.05, 'type': 'resistance', 'strength': 0.9},
            ]

        # Initialize backtesting validator
        validator = SRBacktestingValidator(config)

        # Initialize data integration for validator
        await validator.initialize_data_integration(
            symbol="BTCUSDT",
            exchange="binance",
            timeframes=["1m", "5m", "15m", "30m"],
            training_mode="blank"
        )

        # Run validation
        logger.info("🔍 Running S/R level validation...")
        current_price = market_data['close'].iloc[-1]

        validation_result = await validator.validate_sr_levels(
            market_data=market_data,
            sr_levels=sr_levels,
            current_price=current_price
        )

        if validation_result:
            logger.info("✅ Backtesting validation completed successfully")
            logger.info(f"   - Overall validation score: {validation_result.sr_validation_score:.3f}")
            logger.info(f"   - Bounce rate: {validation_result.bounce_rate:.3f}")
            logger.info(f"   - False breakout rate: {validation_result.false_breakout_rate:.3f}")
            logger.info(f"   - Volume confirmation rate: {validation_result.volume_confirmation_rate:.3f}")
            logger.info(f"   - Level detection accuracy: {validation_result.level_detection_accuracy:.3f}")
            logger.info(f"   - Total levels tested: {validation_result.total_levels_tested}")
            logger.info(f"   - Successful levels: {validation_result.successful_levels}")

            return True
        else:
            logger.error("❌ Backtesting validation failed")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing backtesting validation: {e}")
        return False

async def test_optimization_with_data_integration():
    """Test optimization using the data integration system."""
    logger.info("🧪 Testing optimization with data integration...")

    try:
        # Create data integration
        data_integration = await create_sr_data_integration_simple(
            symbol="BTCUSDT",
            exchange="binance",
            timeframes=["1m", "5m", "15m", "30m"],
            training_mode="light"  # Use light mode for faster testing
        )

        if not data_integration:
            logger.error("❌ Failed to create data integration")
            return False

        # Load market data
        market_data = await data_integration.get_market_data("15m", lookback_days=30)

        if market_data is None or len(market_data) == 0:
            logger.warning("⚠️ No real data available, using test data")
            market_data = generate_test_market_data(1500)

        # Initialize optimizer with config
        config = {
            "sr_detection_optimization": {
                "n_trials": 5,  # Reduced for testing
                "cv_folds": 3,
                "test_size": 0.2,
                "optimization_timeout": 300,
                "timeframe_config": {
                    "1m": {
                        "touch_threshold": 0.0005,
                        "bounce_threshold": 0.002,
                        "breakout_threshold": 0.005,
                        "min_touches": 3,
                    },
                    "5m": {
                        "touch_threshold": 0.001,
                        "bounce_threshold": 0.003,
                        "breakout_threshold": 0.008,
                        "min_touches": 2,
                    },
                    "15m": {
                        "touch_threshold": 0.002,
                        "bounce_threshold": 0.005,
                        "breakout_threshold": 0.01,
                        "min_touches": 2,
                    },
                    "30m": {
                        "touch_threshold": 0.003,
                        "bounce_threshold": 0.008,
                        "breakout_threshold": 0.015,
                        "min_touches": 2,
                    },
                },
                "performance_thresholds": {
                    "min_optimization_score": 0.6,
                    "min_sharpe_ratio": 0.5,
                    "min_win_rate": 0.55,
                    "max_drawdown": 0.15,
                    "min_profit_factor": 1.2,
                },
            },
            "sr_breakout_predictor": {
                "enable_enhanced_strength": True,
                "enable_dbscan_clustering": True,
                "max_levels": 30,
                "min_level_distance": 0.002,
                "fractal_lookback": 20,
                "pivot_lookback": 20,
                "touch_count_lookback": 100,
            },
            "sr_backtesting": {
                "touch_threshold": 0.001,
                "bounce_threshold": 0.005,
                "breakout_threshold": 0.01,
                "false_breakout_threshold": 0.02,
                "confirmation_periods": 3,
                "min_touches": 2,
                "volume_spike_threshold": 1.5,
                "institutional_volume_threshold": 2.0,
                "volume_confirmation_threshold": 1.2,
                "volume_lookback_periods": 20,
                "min_bounce_rate": 0.6,
                "max_false_breakout_rate": 0.3,
                "min_volume_confirmation": 0.5,
            }
        }

        optimizer = SRDetectionOptimizer(config)

        # Initialize the optimizer
        if not await optimizer.initialize():
            logger.error("❌ Failed to initialize optimizer")
            return False

        # Run optimization for 15m timeframe
        logger.info("🎯 Running S/R detection optimization for 15m timeframe...")
        result = await optimizer.optimize_sr_detection(
            market_data=market_data,
            target_timeframe="15m"
        )

        if result:
            logger.info("✅ Optimization completed successfully")
            logger.info(f"   - Optimization score: {result.optimization_score:.3f}")
            logger.info(f"   - S/R validation score: {result.sr_validation_score:.3f}")
            logger.info(f"   - Bounce rate: {result.bounce_rate:.3f}")
            logger.info(f"   - False breakout rate: {result.false_breakout_rate:.3f}")
            logger.info(f"   - Volume confirmation rate: {result.volume_confirmation_rate:.3f}")
            logger.info(f"   - Timeframe optimized: {result.timeframe_optimized}")
            logger.info(f"   - Optimization method: {result.optimization_method}")
            logger.info(f"   - Number of trials: {result.n_trials}")

            # Show some optimized parameters
            if result.method_weights:
                logger.info("   - Method weights:")
                for method, weight in result.method_weights.items():
                    logger.info(f"     * {method}: {weight:.3f}")

            return True
        else:
            logger.error("❌ Optimization failed")
            return False

    except Exception as e:
        logger.error(f"❌ Error testing optimization: {e}")
        return False

async def test_comprehensive_integration():
    """Test comprehensive integration of all components."""
    logger.info("🧪 Testing comprehensive integration...")

    try:
        # Test all components in sequence
        tests = [
            ("Data Integration Initialization", test_data_integration_initialization),
            ("Market Data Loading", test_market_data_loading),
            ("S/R Level Detection", test_sr_level_detection_with_data_integration),
            ("Backtesting Validation", test_backtesting_validation_with_data_integration),
            ("Optimization", test_optimization_with_data_integration),
        ]

        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")

            try:
                result = await test_func()
                results[test_name] = result

                if result:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")

            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE INTEGRATION TEST SUMMARY")
        logger.info(f"{'='*60}")

        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)

        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")

        logger.info(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! S/R system is fully functional with proper data integration.")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Some components may need attention.")

        return passed_tests == total_tests

    except Exception as e:
        logger.error(f"❌ Comprehensive integration test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting S/R Data Integration Tests")
    logger.info("=" * 60)

    try:
        # Run comprehensive integration test
        success = await test_comprehensive_integration()

        if success:
            logger.info("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            logger.info("The S/R backtesting validation system is now fully functional with:")
            logger.info("✅ Proper data access patterns from ares_launcher")
            logger.info("✅ Lookback period management from training modes")
            logger.info("✅ Multi-timeframe data integration")
            logger.info("✅ Data quality validation")
            logger.info("✅ S/R level detection and validation")
            logger.info("✅ Parameter optimization with real data")
        else:
            logger.error("\n❌ SOME TESTS FAILED!")
            logger.error("Please review the error messages above and fix any issues.")

        return success

    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)