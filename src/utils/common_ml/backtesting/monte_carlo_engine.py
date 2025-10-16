"""
Monte Carlo Engine with M1 Hardware Optimizations

This module provides a comprehensive Monte Carlo simulation engine with GPU acceleration,
memory optimization, and parallel processing for M1/M2/M3 Macs.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# M1 Optimization imports
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, m1_monte_carlo_simulate
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, parallel_monte_carlo_simulation

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose, validate_data_quality,
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)

class SimulationType(Enum):
    """Types of Monte Carlo simulations."""
    PRICE_SIMULATION = "price_simulation"
    PORTFOLIO_SIMULATION = "portfolio_simulation"
    RISK_SIMULATION = "risk_simulation"
    STRATEGY_SIMULATION = "strategy_simulation"
    REGIME_SIMULATION = "regime_simulation"

@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation."""
    # Basic parameters
    n_simulations: int = 10000
    n_periods: int = 252  # Trading days in a year
    initial_value: float = 100.0

    # Price simulation parameters
    drift: float = 0.05  # Annual drift
    volatility: float = 0.2  # Annual volatility
    jump_probability: float = 0.05  # Probability of jumps
    jump_size: float = 0.1  # Size of jumps

    # Portfolio parameters
    initial_capital: float = 100000.0
    rebalancing_frequency: int = 21  # Days between rebalancing
    transaction_costs: float = 0.001

    # Risk parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_horizon: int = 1  # Days for VaR calculation

    # Regime parameters
    n_regimes: int = 3
    regime_transition_matrix: Optional[np.ndarray] = None
    regime_parameters: Optional[Dict[str, Dict[str, float]]] = None

    # Random seed
    random_seed: int = 42

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo engine."""
    # Basic configuration
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str

    # Simulation parameters
    simulation_type: SimulationType = SimulationType.PRICE_SIMULATION
    simulation_params: SimulationParameters = field(default_factory=SimulationParameters)

    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None

    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False
    chunk_size: int = 1000  # Simulations per chunk

    # Output settings
    save_detailed_results: bool = True
    generate_plots: bool = True
    output_format: str = "parquet"  # parquet, csv, json

    # Validation settings
    min_simulations: int = 1000
    max_simulations: int = 100000
    convergence_threshold: float = 0.01

@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    # Basic info
    symbol: str
    exchange: str
    timeframe: str
    simulation_type: SimulationType
    start_time: datetime
    end_time: datetime
    total_duration: float

    # Simulation parameters
    n_simulations: int
    n_periods: int
    random_seed: int

    # Results
    simulated_paths: np.ndarray = field(default_factory=lambda: np.array([]))
    final_values: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    # Statistics
    mean_final_value: float = 0.0
    std_final_value: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    expected_shortfall: float = 0.0

    # Percentiles
    percentiles: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)

    # Convergence metrics
    convergence_achieved: bool = False
    convergence_iterations: int = 0
    convergence_error: float = 0.0

class MonteCarloEngine:
    """Monte Carlo simulation engine with M1 optimizations."""

    def __init__(self, config: MonteCarloConfig):
        """Initialize Monte Carlo engine."""
        self.config = config
        self.logger = logger.getChild('MonteCarloEngine')

        # Initialize M1 optimizers
        self.m1_gpu = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_m1_cpu_optimizer(
            max_workers=config.max_workers
        ) if config.enable_parallel_processing else None

        # Initialize utilities
        self.parquet_utils = get_parquet_utils()

        # Set random seed
        np.random.seed(config.simulation_params.random_seed)

        self.logger.info(f"🚀 MonteCarloEngine initialized for {config.symbol}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"🎲 Simulation type: {config.simulation_type.value}")
        self.logger.info(f"📊 Simulations: {config.simulation_params.n_simulations}")

    @traced(span_name='monte_carlo_simulation')
    async def simulate(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        custom_params: Optional[SimulationParameters] = None
    ) -> MonteCarloResults:
        """Execute Monte Carlo simulation with M1 optimizations."""

        self.logger.info("🚀 Starting Monte Carlo simulation...")
        start_time = time.time()

        # Use custom parameters if provided
        params = custom_params or self.config.simulation_params

        # Validate parameters
        self._validate_parameters(params)

        # Memory optimization context
        if self.m1_memory:
            with self.m1_memory.optimization_context():
                results = await self._execute_simulation(params, historical_data)
        else:
            results = await self._execute_simulation(params, historical_data)

        execution_time = time.time() - start_time
        results.execution_time = execution_time

        # Log memory usage
        if self.m1_memory:
            results.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()

        self.logger.info(f"✅ Monte Carlo simulation completed in {execution_time:.2f}s")
        self.logger.info(f"📊 Mean final value: {results.mean_final_value:.2f}")
        self.logger.info(f"📈 Mean return: {results.mean_return:.2%}")
        self.logger.info(f"⚠️ VaR 95%: {results.var_95:.2%}")

        return results

    def _validate_parameters(self, params: SimulationParameters) -> None:
        """Validate simulation parameters."""
        self.logger.info("🔍 Validating simulation parameters...")

        if params.n_simulations < self.config.min_simulations:
            self.logger.error(f"❌ Too few simulations: {params.n_simulations}. Minimum: {self.config.min_simulations}")
            raise ValidationError(f"Too few simulations: {params.n_simulations}. Minimum: {self.config.min_simulations}")

        if params.n_simulations > self.config.max_simulations:
            self.logger.error(f"❌ Too many simulations: {params.n_simulations}. Maximum: {self.config.max_simulations}")
            raise ValidationError(f"Too many simulations: {params.n_simulations}. Maximum: {self.config.max_simulations}")

        if params.n_periods <= 0:
            self.logger.error(f"❌ Invalid number of periods: {params.n_periods}")
            raise ValidationError(f"Invalid number of periods: {params.n_periods}")

        if params.initial_value <= 0:
            self.logger.error(f"❌ Invalid initial value: {params.initial_value}")
            raise ValidationError(f"Invalid initial value: {params.initial_value}")

        if params.volatility < 0:
            self.logger.error(f"❌ Invalid volatility: {params.volatility}")
            raise ValidationError(f"Invalid volatility: {params.volatility}")

        self.logger.info("✅ Simulation parameters validated successfully")
        self.logger.info(f"📊 Simulations: {params.n_simulations:,}")
        self.logger.info(f"📅 Periods: {params.n_periods}")
        self.logger.info(f"💰 Initial value: {params.initial_value}")
        self.logger.info(f"📈 Volatility: {params.volatility:.2%}")
        self.logger.info(f"📊 Drift: {params.drift:.2%}")

    async def _execute_simulation(
        self,
        params: SimulationParameters,
        historical_data: Optional[pd.DataFrame]
    ) -> MonteCarloResults:
        """Execute the actual simulation logic."""

        # Choose execution method based on configuration
        if (self.config.enable_gpu_acceleration and
            self.m1_gpu and
            self.m1_gpu.should_use_gpu(params.n_simulations, "monte_carlo")):

            self.logger.info("🚀 Using GPU acceleration for Monte Carlo simulation")
            simulated_paths = await self._gpu_simulation(params, historical_data)

        elif (self.config.enable_parallel_processing and
              self.m1_cpu and
              params.n_simulations > 1000):

            self.logger.info("⚡ Using parallel processing for Monte Carlo simulation")
            simulated_paths = await self._parallel_simulation(params, historical_data)

        else:
            self.logger.info("💻 Using sequential processing for Monte Carlo simulation")
            simulated_paths = await self._sequential_simulation(params, historical_data)

        # Calculate results
        results = self._calculate_results(simulated_paths, params)

        return results

    async def _gpu_simulation(
        self,
        params: SimulationParameters,
        historical_data: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Execute simulation using GPU acceleration."""

        # Prepare data for GPU
        if historical_data is not None:
            # Extract parameters from historical data
            returns = historical_data['close'].pct_change().dropna()
            drift = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            drift = params.drift
            volatility = params.volatility

        # Execute GPU simulation
        simulated_paths = await m1_monte_carlo_simulate(
            n_simulations=params.n_simulations,
            n_periods=params.n_periods,
            initial_value=params.initial_value,
            drift=drift,
            volatility=volatility,
            jump_probability=params.jump_probability,
            jump_size=params.jump_size
        )

        return simulated_paths

    async def _parallel_simulation(
        self,
        params: SimulationParameters,
        historical_data: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Execute simulation using parallel processing."""

        # Split simulations into chunks
        chunk_size = self.config.chunk_size
        n_chunks = (params.n_simulations + chunk_size - 1) // chunk_size

        self.logger.info(f"📊 Splitting {params.n_simulations:,} simulations into {n_chunks} chunks")
        self.logger.info(f"🔧 Chunk size: {chunk_size:,} simulations per chunk")
        self.logger.info(f"⚡ Using {self.m1_cpu.max_workers} parallel workers")

        # Create tasks for parallel execution
        tasks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, params.n_simulations)
            chunk_simulations = end_idx - start_idx

            self.logger.debug(f"🔄 Creating task for chunk {i+1}/{n_chunks}: {chunk_simulations:,} simulations")

            task = self.m1_cpu.submit_task(
                self._simulate_chunk,
                chunk_simulations, params, historical_data, start_idx
            )
            tasks.append(task)

        self.logger.info(f"🚀 Executing {len(tasks)} parallel simulation tasks...")
        start_time = time.time()

        # Execute all tasks
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        execution_time = time.time() - start_time
        self.logger.info(f"⏱️ Parallel execution completed in {execution_time:.2f}s")

        # Combine results
        valid_chunks = [chunk for chunk in chunk_results if not isinstance(chunk, Exception)]
        failed_chunks = [chunk for chunk in chunk_results if isinstance(chunk, Exception)]

        if failed_chunks:
            self.logger.warning(f"⚠️ {len(failed_chunks)} chunks failed out of {len(chunk_results)}")
            for i, error in enumerate(failed_chunks):
                self.logger.error(f"❌ Chunk {i} failed: {error}")

        if not valid_chunks:
            self.logger.error("❌ All simulation chunks failed")
            raise RuntimeError("All simulation chunks failed")

        self.logger.info(f"✅ Successfully processed {len(valid_chunks)} chunks")
        simulated_paths = np.concatenate(valid_chunks, axis=0)

        self.logger.info(f"📊 Combined results: {simulated_paths.shape[0]:,} simulations, {simulated_paths.shape[1]} periods")

        return simulated_paths

    async def _sequential_simulation(
        self,
        params: SimulationParameters,
        historical_data: Optional[pd.DataFrame]
    ) -> np.ndarray:
        """Execute simulation sequentially."""

        # Prepare parameters
        if historical_data is not None:
            returns = historical_data['close'].pct_change().dropna()
            drift = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
        else:
            drift = params.drift
            volatility = params.volatility

        # Generate random numbers
        dt = 1.0 / 252  # Daily time step
        random_shocks = np.random.normal(0, 1, (params.n_simulations, params.n_periods))

        # Initialize paths
        paths = np.zeros((params.n_simulations, params.n_periods + 1))
        paths[:, 0] = params.initial_value

        # Simulate paths
        for t in range(params.n_periods):
            # Geometric Brownian Motion
            drift_term = (drift - 0.5 * volatility**2) * dt
            diffusion_term = volatility * np.sqrt(dt) * random_shocks[:, t]

            # Add jumps if specified
            if params.jump_probability > 0:
                jump_mask = np.random.random(params.n_simulations) < params.jump_probability
                jump_sizes = np.random.normal(0, params.jump_size, params.n_simulations)
                jump_term = jump_mask * jump_sizes
            else:
                jump_term = 0

            # Update prices
            paths[:, t + 1] = paths[:, t] * np.exp(drift_term + diffusion_term + jump_term)

        return paths

    def _simulate_chunk(
        self,
        n_simulations: int,
        params: SimulationParameters,
        historical_data: Optional[pd.DataFrame],
        start_idx: int
    ) -> np.ndarray:
        """Simulate a chunk of paths."""

        # Set random seed for reproducibility
        np.random.seed(params.random_seed + start_idx)

        # Prepare parameters
        if historical_data is not None:
            returns = historical_data['close'].pct_change().dropna()
            drift = returns.mean() * 252
            volatility = returns.std() * np.sqrt(252)
        else:
            drift = params.drift
            volatility = params.volatility

        # Generate random numbers
        dt = 1.0 / 252
        random_shocks = np.random.normal(0, 1, (n_simulations, params.n_periods))

        # Initialize paths
        paths = np.zeros((n_simulations, params.n_periods + 1))
        paths[:, 0] = params.initial_value

        # Simulate paths
        for t in range(params.n_periods):
            drift_term = (drift - 0.5 * volatility**2) * dt
            diffusion_term = volatility * np.sqrt(dt) * random_shocks[:, t]

            if params.jump_probability > 0:
                jump_mask = np.random.random(n_simulations) < params.jump_probability
                jump_sizes = np.random.normal(0, params.jump_size, n_simulations)
                jump_term = jump_mask * jump_sizes
            else:
                jump_term = 0

            paths[:, t + 1] = paths[:, t] * np.exp(drift_term + diffusion_term + jump_term)

        return paths

    def _calculate_results(
        self,
        simulated_paths: np.ndarray,
        params: SimulationParameters
    ) -> MonteCarloResults:
        """Calculate comprehensive results from simulated paths."""

        # Extract final values and returns
        final_values = simulated_paths[:, -1]
        returns = (final_values - params.initial_value) / params.initial_value

        # Calculate basic statistics
        mean_final_value = np.mean(final_values)
        std_final_value = np.std(final_values)
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Calculate risk metrics
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        expected_shortfall = cvar_95

        # Calculate percentiles
        percentiles = {
            'p1': np.percentile(final_values, 1),
            'p5': np.percentile(final_values, 5),
            'p10': np.percentile(final_values, 10),
            'p25': np.percentile(final_values, 25),
            'p50': np.percentile(final_values, 50),
            'p75': np.percentile(final_values, 75),
            'p90': np.percentile(final_values, 90),
            'p95': np.percentile(final_values, 95),
            'p99': np.percentile(final_values, 99)
        }

        # Check convergence (simplified)
        convergence_achieved = std_return < self.config.convergence_threshold

        return MonteCarloResults(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            timeframe=self.config.timeframe,
            simulation_type=self.config.simulation_type,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            n_simulations=params.n_simulations,
            n_periods=params.n_periods,
            random_seed=params.random_seed,
            simulated_paths=simulated_paths,
            final_values=final_values,
            returns=returns,
            mean_final_value=mean_final_value,
            std_final_value=std_final_value,
            mean_return=mean_return,
            std_return=std_return,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            expected_shortfall=expected_shortfall,
            percentiles=percentiles,
            optimization_used=self._get_optimization_used(),
            convergence_achieved=convergence_achieved,
            convergence_iterations=params.n_simulations,
            convergence_error=std_return
        )

    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        optimizations = []

        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")

        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")

        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")

        return optimizations

    async def save_results(self, results: MonteCarloResults, output_dir: str) -> None:
        """Save Monte Carlo results to disk."""
        ensure_directory(output_dir)

        # Save detailed results
        if self.config.save_detailed_results:
            results_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_monte_carlo_results.json"
            await safe_json_dump(results_file, results.__dict__)
            self.logger.info(f"💾 Results saved to {results_file}")

        # Save simulated paths
        if results.simulated_paths.size > 0:
            paths_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_simulated_paths.parquet"
            paths_df = pd.DataFrame(results.simulated_paths)
            await self.parquet_utils.save_dataframe(paths_df, paths_file)
            self.logger.info(f"💾 Simulated paths saved to {paths_file}")

        # Save final values
        if results.final_values.size > 0:
            values_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_final_values.parquet"
            values_df = pd.DataFrame({'final_value': results.final_values})
            await self.parquet_utils.save_dataframe(values_df, values_file)
            self.logger.info(f"💾 Final values saved to {values_file}")

        # Save returns
        if results.returns.size > 0:
            returns_file = f"{output_dir}/{self.config.symbol}_{self.config.exchange}_returns.parquet"
            returns_df = pd.DataFrame({'return': results.returns})
            await self.parquet_utils.save_dataframe(returns_df, returns_file)
            self.logger.info(f"💾 Returns saved to {returns_file}")
