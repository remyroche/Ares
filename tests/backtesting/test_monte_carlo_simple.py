"""
Simple Unit Tests for Monte Carlo Simulation
=============================================

Lightweight tests that verify the Monte Carlo simulation functionality.

Author: Ares Trading System
Date: 2025-10-31
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_simulation_parameters_import():
    """Test that we can import SimulationParameters."""
    try:
        # Direct module import
        from src.utils.common_ml.backtesting import monte_carlo_engine
        assert hasattr(monte_carlo_engine, 'SimulationParameters')
        print("✅ Successfully imported SimulationParameters")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_simulation_parameters_creation():
    """Test creating simulation parameters."""
    from src.utils.common_ml.backtesting import monte_carlo_engine
    
    params = monte_carlo_engine.SimulationParameters()
    
    assert params.n_simulations == 10000
    assert params.n_periods == 252
    assert params.initial_value == 100.0
    assert params.drift == 0.05
    assert params.volatility == 0.2
    
    print("✅ SimulationParameters created successfully")
    print(f"   n_simulations: {params.n_simulations}")
    print(f"   n_periods: {params.n_periods}")
    print(f"   initial_value: {params.initial_value}")
    return True


def test_monte_carlo_config_creation():
    """Test creating Monte Carlo configuration."""
    from src.utils.common_ml.backtesting import monte_carlo_engine
    
    config = monte_carlo_engine.MonteCarloConfig(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",
        data_dir="test_data",
        enable_gpu_acceleration=False,
        enable_parallel_processing=False,
        enable_memory_optimization=False
    )
    
    assert config.symbol == "BTCUSDT"
    assert config.exchange == "binance"
    assert config.timeframe == "1h"
    assert config.enable_gpu_acceleration is False
    
    print("✅ MonteCarloConfig created successfully")
    print(f"   symbol: {config.symbol}")
    print(f"   exchange: {config.exchange}")
    print(f"   timeframe: {config.timeframe}")
    return True


def test_monte_carlo_engine_initialization():
    """Test Monte Carlo engine initialization."""
    from src.utils.common_ml.backtesting import monte_carlo_engine
    
    config = monte_carlo_engine.MonteCarloConfig(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",
        data_dir="test_data",
        enable_gpu_acceleration=False,
        enable_parallel_processing=False,
        enable_memory_optimization=False
    )
    
    engine = monte_carlo_engine.MonteCarloEngine(config)
    
    assert engine.config == config
    assert engine.logger is not None
    assert engine.parquet_utils is not None
    
    print("✅ MonteCarloEngine initialized successfully")
    print(f"   GPU acceleration: {config.enable_gpu_acceleration}")
    print(f"   Parallel processing: {config.enable_parallel_processing}")
    return True


def test_parameter_validation():
    """Test parameter validation."""
    from src.utils.common_ml.backtesting import monte_carlo_engine
    
    config = monte_carlo_engine.MonteCarloConfig(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",
        data_dir="test_data",
        enable_gpu_acceleration=False,
        enable_parallel_processing=False,
        enable_memory_optimization=False
    )
    
    engine = monte_carlo_engine.MonteCarloEngine(config)
    
    # Test valid parameters
    params = monte_carlo_engine.SimulationParameters(
        n_simulations=1000,
        n_periods=50
    )
    
    try:
        engine._validate_parameters(params)
        print("✅ Valid parameters accepted")
        passed = True
    except Exception as e:
        print(f"❌ Valid parameters rejected: {e}")
        passed = False
    
    # Test invalid parameters (too few simulations)
    invalid_params = monte_carlo_engine.SimulationParameters(
        n_simulations=500,  # Less than minimum
        n_periods=50
    )
    
    try:
        engine._validate_parameters(invalid_params)
        print("❌ Invalid parameters accepted (should have failed)")
        passed = False
    except Exception:
        print("✅ Invalid parameters correctly rejected")
    
    return passed


def test_basic_simulation():
    """Test basic simulation execution."""
    from src.utils.common_ml.backtesting import monte_carlo_engine
    
    config = monte_carlo_engine.MonteCarloConfig(
        symbol="BTCUSDT",
        exchange="binance",
        timeframe="1h",
        data_dir="test_data",
        enable_gpu_acceleration=False,
        enable_parallel_processing=False,
        enable_memory_optimization=False
    )
    
    engine = monte_carlo_engine.MonteCarloEngine(config)
    
    params = monte_carlo_engine.SimulationParameters(
        n_simulations=100,
        n_periods=50,
        initial_value=100.0,
        random_seed=42
    )
    
    # Test chunk simulation (synchronous)
    chunk = engine._simulate_chunk(100, params, None, 0)
    
    assert chunk.shape == (100, 51)  # n_simulations x (n_periods + 1)
    assert np.all(chunk[:, 0] == 100.0)  # Initial values
    assert np.all(chunk > 0)  # All values positive
    
    print("✅ Basic simulation completed successfully")
    print(f"   Shape: {chunk.shape}")
    print(f"   Final value range: [{chunk[:, -1].min():.2f}, {chunk[:, -1].max():.2f}]")
    print(f"   Mean final value: {chunk[:, -1].mean():.2f}")
    return True


def test_simulation_types():
    """Test simulation type enum."""
    from src.utils.common_ml.backtesting import monte_carlo_engine
    
    assert hasattr(monte_carlo_engine.SimulationType, 'PRICE_SIMULATION')
    assert hasattr(monte_carlo_engine.SimulationType, 'PORTFOLIO_SIMULATION')
    assert hasattr(monte_carlo_engine.SimulationType, 'RISK_SIMULATION')
    
    assert monte_carlo_engine.SimulationType.PRICE_SIMULATION.value == "price_simulation"
    
    print("✅ Simulation types verified")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Monte Carlo Simulation Unit Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Import SimulationParameters", test_simulation_parameters_import),
        ("Create SimulationParameters", test_simulation_parameters_creation),
        ("Create MonteCarloConfig", test_monte_carlo_config_creation),
        ("Initialize MonteCarloEngine", test_monte_carlo_engine_initialization),
        ("Parameter Validation", test_parameter_validation),
        ("Basic Simulation", test_basic_simulation),
        ("Simulation Types", test_simulation_types),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 Test: {test_name}")
        print("-" * 60)
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ PASSED: {test_name}")
            else:
                failed += 1
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    """Run tests directly."""
    success = run_all_tests()
    sys.exit(0 if success else 1)

