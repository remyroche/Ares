#!/usr/bin/env python3
"""
Test Script for Regime-Specific Triple Barrier Optimization

This script demonstrates the regime-specific optimization for the triple barrier method,
showing how different HMM regimes can have different barrier parameters optimized
for their specific market conditions.
"""

import asyncio
import logging
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the regime-specific optimizer
try:
        RegimeSpecificTripleBarrierOptimizer,
    except Exception as e:
        pass
    except Exception as e:
        pass
        create_regime_specific_triple_barrier_optimizer
    )
    logger.info("✅ Successfully imported regime-specific triple barrier optimizer")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.info("Please ensure the regime-specific optimizer module is in your Python path")
    exit(1)


def create_regime_specific_test_data(periods: int = 1000) -> Dict[str, pd.DataFrame]:
    pass
    pass
    """Create test data for different market regimes."""

    dates = pd.date_range(start="2024-01-01", periods=periods, freq="1min")

    # Create regime-specific data
    regime_data = {}

    # Bull regime data - upward trending
    np.random.seed(42)
    bull_returns = np.random.normal(0.0002, 0.015, periods)  # Positive trend
    bull_prices = [100.0]
    for i in range(1, periods):
    pass
    pass
        new_price = bull_prices[-1] * (1 + bull_returns[i])
        bull_prices.append(new_price)

    bull_data = pd.DataFrame({
        'open': np.array(bull_prices) * (1 + np.random.normal(0, 0.0005, periods)),
        'high': np.array(bull_prices) * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': np.array(bull_prices) * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': np.array(bull_prices),
        'volume': np.random.uniform(1000, 10000, periods),
        'regime': 'bull_regime'
    }, index=dates)

    # Ensure OHLC relationships are valid
    bull_data['high'] = bull_data[['open', 'high', 'close']].max(axis=1)
    bull_data['low'] = bull_data[['open', 'low', 'close']].min(axis=1)
    regime_data['bull_regime'] = bull_data

    # Bear regime data - downward trending
    np.random.seed(43)
    bear_returns = np.random.normal(-0.0001, 0.018, periods)  # Negative trend
    bear_prices = [100.0]
    for i in range(1, periods):
    pass
    pass
        new_price = bear_prices[-1] * (1 + bear_returns[i])
        bear_prices.append(new_price)

    bear_data = pd.DataFrame({
        'open': np.array(bear_prices) * (1 + np.random.normal(0, 0.0005, periods)),
        'high': np.array(bear_prices) * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': np.array(bear_prices) * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': np.array(bear_prices),
        'volume': np.random.uniform(1000, 10000, periods),
        'regime': 'bear_regime'
    }, index=dates)

    bear_data['high'] = bear_data[['open', 'high', 'close']].max(axis=1)
    bear_data['low'] = bear_data[['open', 'low', 'close']].min(axis=1)
    regime_data['bear_regime'] = bear_data

    # Sideways regime data - range-bound
    np.random.seed(44)
    sideways_returns = np.random.normal(0, 0.012, periods)  # No trend
    sideways_prices = [100.0]
    for i in range(1, periods):
    pass
    pass
        # Add mean reversion
        mean_reversion = -0.001 * (sideways_prices[-1] - 100.0) / 100.0
        new_price = sideways_prices[-1] * (1 + sideways_returns[i] + mean_reversion)
        sideways_prices.append(new_price)

    sideways_data = pd.DataFrame({
        'open': np.array(sideways_prices) * (1 + np.random.normal(0, 0.0005, periods)),
        'high': np.array(sideways_prices) * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': np.array(sideways_prices) * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': np.array(sideways_prices),
        'volume': np.random.uniform(1000, 10000, periods),
        'regime': 'sideways_regime'
    }, index=dates)

    sideways_data['high'] = sideways_data[['open', 'high', 'close']].max(axis=1)
    sideways_data['low'] = sideways_data[['open', 'low', 'close']].min(axis=1)
    regime_data['sideways_regime'] = sideways_data

    # Volatile regime data - choppy, unpredictable
    np.random.seed(45)
    volatile_returns = np.random.normal(0, 0.025, periods)  # High volatility
    volatile_prices = [100.0]
    for i in range(1, periods):
    pass
    pass
        # Add volatility clustering
        vol_multiplier = 1 + 0.8 * np.sin(i / 50)  # High volatility variation
        new_price = volatile_prices[-1] * (1 + volatile_returns[i] * vol_multiplier)
        volatile_prices.append(new_price)

    volatile_data = pd.DataFrame({
        'open': np.array(volatile_prices) * (1 + np.random.normal(0, 0.001, periods)),
        'high': np.array(volatile_prices) * (1 + np.abs(np.random.normal(0, 0.002, periods))),
        'low': np.array(volatile_prices) * (1 - np.abs(np.random.normal(0, 0.002, periods))),
        'close': np.array(volatile_prices),
        'volume': np.random.uniform(1000, 10000, periods),
        'regime': 'volatile_regime'
    }, index=dates)

    volatile_data['high'] = volatile_data[['open', 'high', 'close']].max(axis=1)
    volatile_data['low'] = volatile_data[['open', 'low', 'close']].min(axis=1)
    regime_data['volatile_regime'] = volatile_data

    # Trending regime data - sustained directional moves
    np.random.seed(46)
    trending_returns = np.random.normal(0.0003, 0.014, periods)  # Strong trend
    trending_prices = [100.0]
    for i in range(1, periods):
    pass
    pass
        # Add trend persistence
        trend_strength = 0.8 + 0.2 * np.sin(i / 100)  # Varying trend strength
        new_price = trending_prices[-1] * (1 + trending_returns[i] * trend_strength)
        trending_prices.append(new_price)

    trending_data = pd.DataFrame({
        'open': np.array(trending_prices) * (1 + np.random.normal(0, 0.0005, periods)),
        'high': np.array(trending_prices) * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': np.array(trending_prices) * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': np.array(trending_prices),
        'volume': np.random.uniform(1000, 10000, periods),
        'regime': 'trending_regime'
    }, index=dates)

    trending_data['high'] = trending_data[['open', 'high', 'close']].max(axis=1)
    trending_data['low'] = trending_data[['open', 'low', 'close']].min(axis=1)
    regime_data['trending_regime'] = trending_data

    return regime_data


async def test_regime_specific_optimization():
    """Test the regime-specific triple barrier optimization."""

    logger.info("🧪 Testing Regime-Specific Triple Barrier Optimization")
    logger.info("=" * 80)

    # Create test configuration
    config = {
        "regime_optimization": {
            "n_trials": 20,  # Reduced for testing
            "timeout": 1800,  # 30 minutes for testing
            "early_stopping_patience": 10
        }
    }

    # Create regime-specific optimizer
    optimizer = create_regime_specific_triple_barrier_optimizer(config)

    # Show supported regimes
    logger.info(f"✅ Optimizer created successfully!")
    logger.info(f"Total regimes supported: {len(optimizer.regime_configs)}")

    for regime_name, regime_config in optimizer.regime_configs.items():
    pass
    pass
        logger.info(f"\\\n{regime_name}:")
        logger.info(f"  Description: {regime_config['description']}")
        total_params = sum(len(category) for category in regime_config.values() if isinstance(category, dict))
        logger.info(f"  Total parameters: {total_params}")

    # Create test data for different regimes
    logger.info("\\\n📊 Creating test data for different regimes...")
    regime_data = create_regime_specific_test_data(500)  # Reduced for testing

    for regime_name, data in regime_data.items():
    pass
    pass
        logger.info(f"  {regime_name}: {len(data)} data points")

    # Run regime-specific optimization
    logger.info("\\\n🚀 Running regime-specific optimization...")
    try:
        optimization_results = await optimizer.optimize_regime_specific_parameters(
            regime_data,
            config["regime_optimization"]
    except Exception as e:
        pass
    except Exception as e:
        pass
        )

        logger.info("✅ Regime-specific optimization completed successfully!")

        # Show results
        for regime_name, result in optimization_results.items():
    pass
    pass
            if "error" not in result:
    pass
    pass
                logger.info(f"\\\n📈 {regime_name} results:")
                logger.info(f"  Best value: {result.get('best_value', 0):.4f}")
                logger.info(f"  Total trials: {result.get('total_trials', 0)}")
                logger.info(f"  Best trial: {result.get('best_trial', 0)}")

                # Show some key parameters
                best_params = result.get('best_params', {})
                if 'barrier_settings' in best_params:
    pass
    pass
                    barrier_params = best_params['barrier_settings']
                    logger.info(f"  Upper barrier: {barrier_params.get('upper_barrier_multiplier', 'N/A')}")
                    logger.info(f"  Lower barrier: {barrier_params.get('lower_barrier_multiplier', 'N/A')}")
                    logger.info(f"  Timeout: {barrier_params.get('barrier_timeout', 'N/A')}")
            else:
                logger.error(f"❌ {regime_name} failed: {result['error']}")

        return optimization_results

    except Exception as e:
        logger.error(f"❌ Regime-specific optimization failed: {e}")
        return None


async def test_regime_parameter_application():
    """Test applying regime-specific parameters."""

    logger.info("\\\n🧪 Testing Regime Parameter Application")
    logger.info("=" * 80)

    # Create optimizer
    config = {"regime_optimization": {"n_trials": 10}}
    optimizer = create_regime_specific_triple_barrier_optimizer(config)

    # Create minimal test data
    regime_data = create_regime_specific_test_data(100)

    try:
        # Run quick optimization
    except Exception as e:
        pass
    except Exception as e:
        pass
        results = await optimizer.optimize_regime_specific_parameters(
            regime_data,
            {"n_trials": 5, "timeout": 300}
        )

        # Test parameter application
        for regime_name in results.keys():
    pass
    pass
            if "error" not in results[regime_name]:
    pass
    pass
                logger.info(f"\\\n🔧 Applying parameters for {regime_name}...")

                application_result = await optimizer.apply_regime_parameters(regime_name)

                if "error" not in application_result:
    pass
    pass
                    logger.info(f"  ✅ Parameters applied successfully")
                    logger.info(f"  Parameters applied: {application_result.get('parameters_applied', 0)}")
                else:
                    logger.error(f"  ❌ Failed to apply parameters: {application_result['error']}")

        return True

    except Exception as e:
        logger.error(f"❌ Parameter application test failed: {e}")
        return False


async def test_optimization_recommendations():
    """Test optimization recommendations."""

    logger.info("\\\n🧪 Testing Optimization Recommendations")
    logger.info("=" * 80)

    # Create optimizer
    config = {"regime_optimization": {"n_trials": 10}}
    optimizer = create_regime_specific_triple_barrier_optimizer(config)

    # Create minimal test data
    regime_data = create_regime_specific_test_data(100)

    try:
        # Run quick optimization
    except Exception as e:
        pass
    except Exception as e:
        pass
        await optimizer.optimize_regime_specific_parameters(
            regime_data,
            {"n_trials": 5, "timeout": 300}
        )

        # Get recommendations
        recommendations = await optimizer.get_optimization_recommendations()

        logger.info("💡 Optimization recommendations:")
        for i, recommendation in enumerate(recommendations, 1):
    pass
    pass
            logger.info(f"  {i}. {recommendation}")

        return True

    except Exception as e:
        logger.error(f"❌ Recommendations test failed: {e}")
        return False


async def run_comprehensive_regime_test():
    """Run comprehensive regime-specific optimization test."""

    logger.info("🚀 Starting Comprehensive Regime-Specific Optimization Test")
    logger.info("=" * 100)

    test_results = {}

    # Test 1: Basic regime-specific optimization
    logger.info("\\\n" + "="*50)
    logger.info("TEST 1: Regime-Specific Optimization")
    logger.info("="*50)

    optimization_results = await test_regime_specific_optimization()
    test_results["regime_optimization"] = optimization_results is not None

    # Test 2: Parameter application
    logger.info("\\\n" + "="*50)
    logger.info("TEST 2: Parameter Application")
    logger.info("="*50)

    parameter_application = await test_regime_parameter_application()
    test_results["parameter_application"] = parameter_application

    # Test 3: Optimization recommendations
    logger.info("\\\n" + "="*50)
    logger.info("TEST 3: Optimization Recommendations")
    logger.info("="*50)

    recommendations = await test_optimization_recommendations()
    test_results["recommendations"] = recommendations

    # Summary
    logger.info("\\\n" + "="*100)
    logger.info("🎯 REGIME-SPECIFIC OPTIMIZATION TEST SUMMARY")
    logger.info("="*100)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

    # Show individual test results
    for test_name, result in test_results.items():
    pass
    pass
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")

    # Generate recommendations
    logger.info("\\\n💡 REGIME-SPECIFIC OPTIMIZATION FEATURES:")
    logger.info("  🎯 5 different market regimes supported")
    logger.info("  🔧 Regime-specific parameter optimization")
    logger.info("  📊 Separate optimization for each regime")
    logger.info("  🚀 MLflow integration for tracking")
    logger.info("  📈 Performance-based recommendations")

    logger.info("\\\n🔮 NEXT STEPS:")
    logger.info("  1. Integrate with your actual HMM regime detection")
    logger.info("  2. Connect with your triple barrier implementation")
    logger.info("  3. Run full optimization for all regimes")
    logger.info("  4. Monitor regime-specific performance")
    logger.info("  5. Use MLflow for experiment tracking")

    return test_results


async def main():
    """Main test function."""

    try:
        results = await run_comprehensive_regime_test()

    except Exception as e:
        pass
    except Exception as e:
        pass
        if all(results.values()):
    pass
    pass
            logger.info("\\\n🎉 REGIME-SPECIFIC OPTIMIZATION TEST COMPLETED SUCCESSFULLY!")
            logger.info("Your triple barrier method now has regime-specific optimization!")
        else:
            logger.info("\\\n⚠️ SOME TESTS FAILED - Review and fix issues before production use")

    except Exception as e:
        logger.error(f"❌ Comprehensive regime test failed: {e}")
        raise


if __name__ == "__main__":
    pass
    pass
    # Run the comprehensive regime test
    asyncio.run(main())