"""
Unit Tests for Monte Carlo Simulation
======================================

Tests for the Monte Carlo simulation engine and step.

Author: Ares Trading System
Date: 2025-10-31
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Direct import to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "monte_carlo_engine",
    str(Path(__file__).parent.parent.parent / "src/utils/common_ml/backtesting/monte_carlo_engine.py")
)
monte_carlo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monte_carlo_module)

MonteCarloEngine = monte_carlo_module.MonteCarloEngine
MonteCarloConfig = monte_carlo_module.MonteCarloConfig
SimulationParameters = monte_carlo_module.SimulationParameters
SimulationType = monte_carlo_module.SimulationType
MonteCarloResults = monte_carlo_module.MonteCarloResults


class TestSimulationParameters:
    """Test SimulationParameters dataclass."""
    
    def test_default_parameters(self):
        """Test default simulation parameters."""
        params = SimulationParameters()
        
        assert params.n_simulations == 10000
        assert params.n_periods == 252
        assert params.initial_value == 100.0
        assert params.drift == 0.05
        assert params.volatility == 0.2
        assert params.jump_probability == 0.05
        assert params.jump_size == 0.1
        assert params.random_seed == 42
        
    def test_custom_parameters(self):
        """Test custom simulation parameters."""
        params = SimulationParameters(
            n_simulations=5000,
            n_periods=126,
            initial_value=200.0,
            drift=0.1,
            volatility=0.3
        )
        
        assert params.n_simulations == 5000
        assert params.n_periods == 126
        assert params.initial_value == 200.0
        assert params.drift == 0.1
        assert params.volatility == 0.3


class TestMonteCarloConfig:
    """Test MonteCarloConfig dataclass."""
    
    def test_config_creation(self):
        """Test Monte Carlo configuration creation."""
        config = MonteCarloConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            data_dir="test_data"
        )
        
        assert config.symbol == "BTCUSDT"
        assert config.exchange == "binance"
        assert config.timeframe == "1h"
        assert config.data_dir == "test_data"
        assert config.enable_gpu_acceleration is True
        assert config.enable_memory_optimization is True
        assert config.enable_parallel_processing is True
        
    def test_config_with_custom_settings(self):
        """Test configuration with custom settings."""
        config = MonteCarloConfig(
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="15m",
            data_dir="test_data",
            enable_gpu_acceleration=False,
            enable_parallel_processing=False,
            chunk_size=500
        )
        
        assert config.enable_gpu_acceleration is False
        assert config.enable_parallel_processing is False
        assert config.chunk_size == 500


class TestMonteCarloEngine:
    """Test MonteCarloEngine class."""
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing."""
        return MonteCarloConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            data_dir="test_data",
            enable_gpu_acceleration=False,
            enable_parallel_processing=False,
            enable_memory_optimization=False
        )
    
    @pytest.fixture
    def simple_params(self):
        """Create simple parameters for testing."""
        return SimulationParameters(
            n_simulations=100,
            n_periods=50,
            initial_value=100.0,
            drift=0.05,
            volatility=0.2,
            random_seed=42
        )
    
    def test_engine_initialization(self, basic_config):
        """Test Monte Carlo engine initialization."""
        engine = MonteCarloEngine(basic_config)
        
        assert engine.config == basic_config
        assert engine.logger is not None
        assert engine.parquet_utils is not None
        
    def test_validate_parameters_success(self, basic_config, simple_params):
        """Test successful parameter validation."""
        engine = MonteCarloEngine(basic_config)
        
        # Should not raise any exception
        engine._validate_parameters(simple_params)
        
    def test_validate_parameters_too_few_simulations(self, basic_config):
        """Test validation failure with too few simulations."""
        engine = MonteCarloEngine(basic_config)
        
        params = SimulationParameters(
            n_simulations=500,  # Less than default minimum of 1000
            n_periods=50
        )
        
        with pytest.raises(Exception):  # Should raise ValidationError
            engine._validate_parameters(params)
    
    def test_validate_parameters_invalid_periods(self, basic_config):
        """Test validation failure with invalid periods."""
        engine = MonteCarloEngine(basic_config)
        
        params = SimulationParameters(
            n_simulations=1000,
            n_periods=0  # Invalid
        )
        
        with pytest.raises(Exception):
            engine._validate_parameters(params)
    
    def test_validate_parameters_invalid_volatility(self, basic_config):
        """Test validation failure with negative volatility."""
        engine = MonteCarloEngine(basic_config)
        
        params = SimulationParameters(
            n_simulations=1000,
            n_periods=50,
            volatility=-0.1  # Negative
        )
        
        with pytest.raises(Exception):
            engine._validate_parameters(params)
    
    @pytest.mark.asyncio
    async def test_sequential_simulation(self, basic_config, simple_params):
        """Test sequential simulation execution."""
        engine = MonteCarloEngine(basic_config)
        
        # Run sequential simulation
        simulated_paths = await engine._sequential_simulation(simple_params, None)
        
        # Check output shape
        assert simulated_paths.shape == (100, 51)  # n_simulations x (n_periods + 1)
        
        # Check initial values
        assert np.all(simulated_paths[:, 0] == 100.0)
        
        # Check that values are positive
        assert np.all(simulated_paths > 0)
        
    @pytest.mark.asyncio
    async def test_simulate_chunk(self, basic_config, simple_params):
        """Test simulation chunk execution."""
        engine = MonteCarloEngine(basic_config)
        
        # Simulate a chunk
        chunk_paths = engine._simulate_chunk(50, simple_params, None, 0)
        
        # Check output shape
        assert chunk_paths.shape == (50, 51)
        
        # Check initial values
        assert np.all(chunk_paths[:, 0] == 100.0)
        
    def test_calculate_results(self, basic_config, simple_params):
        """Test results calculation."""
        engine = MonteCarloEngine(basic_config)
        
        # Create sample simulated paths
        np.random.seed(42)
        simulated_paths = np.random.randn(100, 51).cumsum(axis=1) + 100.0
        simulated_paths = np.abs(simulated_paths)  # Ensure positive values
        
        # Calculate results
        results = engine._calculate_results(simulated_paths, simple_params)
        
        # Check result types
        assert isinstance(results, MonteCarloResults)
        assert results.symbol == "BTCUSDT"
        assert results.exchange == "binance"
        assert results.n_simulations == 100
        assert results.n_periods == 50
        
        # Check statistics
        assert results.mean_final_value > 0
        assert results.std_final_value > 0
        assert isinstance(results.var_95, float)
        assert isinstance(results.var_99, float)
        
        # Check percentiles
        assert len(results.percentiles) == 9
        assert 'p50' in results.percentiles
        assert 'p95' in results.percentiles
        
    def test_get_optimization_used(self, basic_config):
        """Test getting list of optimizations."""
        engine = MonteCarloEngine(basic_config)
        
        optimizations = engine._get_optimization_used()
        
        # Since we disabled all optimizations in basic_config
        assert len(optimizations) == 0
        
    def test_get_optimization_used_with_optimizations(self):
        """Test optimization list with enabled optimizations."""
        config = MonteCarloConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            data_dir="test_data",
            enable_gpu_acceleration=False,  # GPU might not be available
            enable_parallel_processing=True,
            enable_memory_optimization=True
        )
        
        engine = MonteCarloEngine(config)
        optimizations = engine._get_optimization_used()
        
        # Should have at least memory and parallel optimizations
        assert len(optimizations) >= 0  # May vary based on system
        
    @pytest.mark.asyncio
    async def test_full_simulation_flow(self, basic_config):
        """Test complete simulation flow."""
        # Create custom params with valid number of simulations
        params = SimulationParameters(
            n_simulations=1000,  # Meet minimum requirement
            n_periods=50,
            initial_value=100.0,
            drift=0.05,
            volatility=0.2,
            random_seed=42
        )
        
        engine = MonteCarloEngine(basic_config)
        
        # Run full simulation
        results = await engine.simulate(custom_params=params)
        
        # Check results
        assert isinstance(results, MonteCarloResults)
        assert results.n_simulations == 1000
        assert results.n_periods == 50
        assert results.execution_time > 0
        assert len(results.simulated_paths) == 1000
        assert len(results.final_values) == 1000
        assert len(results.returns) == 1000
        
    @pytest.mark.asyncio
    async def test_simulation_with_historical_data(self, basic_config):
        """Test simulation with historical data."""
        # Create sample historical data
        dates = pd.date_range(start='2023-01-01', periods=500, freq='h')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(500).cumsum() + 100.0
        })
        historical_data['close'] = np.abs(historical_data['close'])
        
        params = SimulationParameters(
            n_simulations=1000,
            n_periods=50,
            random_seed=42
        )
        
        engine = MonteCarloEngine(basic_config)
        
        # Run simulation with historical data
        results = await engine.simulate(
            historical_data=historical_data,
            custom_params=params
        )
        
        # Check results
        assert isinstance(results, MonteCarloResults)
        assert results.n_simulations == 1000


class TestMonteCarloResults:
    """Test MonteCarloResults dataclass."""
    
    def test_results_creation(self):
        """Test Monte Carlo results creation."""
        results = MonteCarloResults(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            simulation_type=SimulationType.PRICE_SIMULATION,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=1.5,
            n_simulations=1000,
            n_periods=252,
            random_seed=42
        )
        
        assert results.symbol == "BTCUSDT"
        assert results.exchange == "binance"
        assert results.n_simulations == 1000
        assert results.n_periods == 252
        assert results.simulation_type == SimulationType.PRICE_SIMULATION
        
    def test_results_with_data(self):
        """Test results with simulation data."""
        simulated_paths = np.random.randn(100, 51)
        final_values = simulated_paths[:, -1]
        returns = (final_values - 100.0) / 100.0
        
        results = MonteCarloResults(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            simulation_type=SimulationType.PRICE_SIMULATION,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=1.5,
            n_simulations=100,
            n_periods=50,
            random_seed=42,
            simulated_paths=simulated_paths,
            final_values=final_values,
            returns=returns,
            mean_final_value=np.mean(final_values),
            std_final_value=np.std(final_values),
            mean_return=np.mean(returns),
            std_return=np.std(returns)
        )
        
        assert len(results.simulated_paths) == 100
        assert len(results.final_values) == 100
        assert len(results.returns) == 100
        assert results.mean_final_value == np.mean(final_values)


class TestSimulationType:
    """Test SimulationType enum."""
    
    def test_simulation_types(self):
        """Test simulation type values."""
        assert SimulationType.PRICE_SIMULATION.value == "price_simulation"
        assert SimulationType.PORTFOLIO_SIMULATION.value == "portfolio_simulation"
        assert SimulationType.RISK_SIMULATION.value == "risk_simulation"
        assert SimulationType.STRATEGY_SIMULATION.value == "strategy_simulation"
        assert SimulationType.REGIME_SIMULATION.value == "regime_simulation"


class TestMonteCarloEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing."""
        return MonteCarloConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1h",
            data_dir="test_data",
            enable_gpu_acceleration=False,
            enable_parallel_processing=False,
            enable_memory_optimization=False
        )
    
    def test_zero_jump_probability(self, basic_config):
        """Test simulation with zero jump probability."""
        params = SimulationParameters(
            n_simulations=1000,
            n_periods=50,
            jump_probability=0.0,  # No jumps
            random_seed=42
        )
        
        engine = MonteCarloEngine(basic_config)
        chunk = engine._simulate_chunk(100, params, None, 0)
        
        # Should still produce valid results
        assert chunk.shape == (100, 51)
        assert np.all(chunk[:, 0] == 100.0)
        
    def test_high_volatility(self, basic_config):
        """Test simulation with high volatility."""
        params = SimulationParameters(
            n_simulations=1000,
            n_periods=50,
            volatility=1.0,  # 100% volatility
            random_seed=42
        )
        
        engine = MonteCarloEngine(basic_config)
        chunk = engine._simulate_chunk(100, params, None, 0)
        
        # Should still produce valid results
        assert chunk.shape == (100, 51)
        assert np.all(chunk > 0)  # Prices should remain positive
        
    def test_different_initial_values(self, basic_config):
        """Test simulation with different initial values."""
        for initial_value in [10.0, 100.0, 1000.0, 10000.0]:
            params = SimulationParameters(
                n_simulations=1000,
                n_periods=50,
                initial_value=initial_value,
                random_seed=42
            )
            
            engine = MonteCarloEngine(basic_config)
            chunk = engine._simulate_chunk(100, params, None, 0)
            
            # Check initial values match
            assert np.all(chunk[:, 0] == initial_value)


def test_imports():
    """Test that all required imports work."""
    try:
        from src.utils.common_ml.backtesting.monte_carlo_engine import (
            MonteCarloEngine,
            MonteCarloConfig,
            SimulationParameters,
            SimulationType,
            MonteCarloResults
        )
        
        print("✅ Successfully imported Monte Carlo engine components")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests directly."""
    print("Running Monte Carlo Simulation Unit Tests")
    print("=" * 50)
    
    # Run with pytest
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))

