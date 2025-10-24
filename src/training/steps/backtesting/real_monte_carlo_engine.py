"""
Real Monte Carlo Simulation Engine

Enhanced Monte Carlo simulation for backtesting with:
- Hardware-accelerated parallel processing (M1 optimization)
- Comprehensive validation and error handling
- Advanced metric calculations (VaR, ES, Sharpe, Sortino, Calmar)
- Data leakage prevention
- Multiple simulation methods (Bootstrap, Parametric, Historical, Hybrid)
- Cross-validation integration
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
from pathlib import Path
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Hardware optimization
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, M1GPUAccelerator
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor

# VectorBT optimization utilities
from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, VectorizationConfig
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer

# ML utilities
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.ml_common.cv_utils import TimeSeriesSplitValidator
from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import create_enhanced_oof_generator, OOFStrategy
from src.utils.ml_common.data_leakage_detector import DataLeakageDetector

# Math validation
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, validate_finite,
    validate_probability, validate_positive, validate_range,
    check_for_nans, check_for_infs
)

# Common operations
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor, calculate_calmar_ratio
)
from src.utils.common_utilities import ensure_list, ensure_array, flatten_dict

# Monte Carlo base engine
try:
    from src.utils.common_ml.backtesting.monte_carlo_engine import MonteCarloEngine, MonteCarloConfig
    BASE_ENGINE_AVAILABLE = True
except ImportError:
    BASE_ENGINE_AVAILABLE = False

# Import BaseStep for unified utility access
from src.training.steps.base_step import BaseStep

# Output utilities
from src.utils.tprint import tprint, tprint_data_preview

# Decorators
from src.core.decorators import handles_errors, traced, log_execution_time

logger = logging.getLogger(__name__)

class MonteCarloMode(Enum):
    """Monte Carlo simulation modes."""
    BOOTSTRAP = "bootstrap"
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    HYBRID = "hybrid"

@dataclass
class MonteCarloMetrics:
    """Comprehensive metrics from Monte Carlo simulation"""
    # Return metrics
    mean_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0
    median_return: float = 0.0

    # Risk metrics
    var_value: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    tail_risk: float = 0.0
    tail_ratio: float = 0.0

    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Confidence intervals
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    confidence_level: float = 0.95

    # Simulation metadata
    n_simulations: int = 0
    simulation_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'return_metrics': {
                'mean': self.mean_return,
                'std': self.std_return,
                'min': self.min_return,
                'max': self.max_return,
                'median': self.median_return
            },
            'risk_metrics': {
                'var': self.var_value,
                'expected_shortfall': self.expected_shortfall,
                'max_drawdown': self.max_drawdown,
                'tail_risk': self.tail_risk,
                'tail_ratio': self.tail_ratio
            },
            'performance_metrics': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor
            },
            'confidence_interval': {
                'lower': self.ci_lower,
                'upper': self.ci_upper,
                'level': self.confidence_level
            },
            'metadata': {
                'n_simulations': self.n_simulations,
                'mode': self.simulation_mode
            }
        }

@dataclass
class RealMonteCarloConfig:
    """Configuration for real Monte Carlo simulation."""
    # Basic configuration
    n_simulations: int = 1000
    confidence_level: float = 0.95
    simulation_horizon: int = 252  # Trading days

    # Hardware optimization
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunk_size_mb: int = 128

    # Simulation parameters
    mode: MonteCarloMode = MonteCarloMode.HYBRID
    bootstrap_sample_size: float = 0.8
    parametric_distribution: str = "normal"  # "normal", "t", "skewed_t"

    # Risk parameters
    var_confidence: float = 0.05
    expected_shortfall_confidence: float = 0.01
    max_drawdown_threshold: float = 0.2

    # Data validation
    enable_data_validation: bool = True
    enable_leakage_detection: bool = True
    min_samples: int = 30

    # Cross-validation
    enable_cv_validation: bool = True
    cv_folds: int = 5
    embargo_pct: float = 0.01

    # Output settings
    save_results: bool = True
    results_path: str = "monte_carlo_results"
    enable_detailed_logging: bool = True

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

class RealMonteCarloEngine(BaseStep):
    """
    Real Monte Carlo simulation engine using existing utilities.

    This engine provides comprehensive Monte Carlo simulation with:
    - GPU acceleration for M1/M2/M3 Macs
    - Memory optimization for large simulations
    - Multiple simulation methods (bootstrap, parametric, historical)
    - Risk metrics calculation (VaR, Expected Shortfall, etc.)
    """

    def __init__(self, config: RealMonteCarloConfig, logger: Optional[logging.Logger] = None):
        """Initialize the enhanced Monte Carlo engine with hardware acceleration."""
        super().__init__("real_monte_carlo_engine", config.__dict__, logger)
        self.config = config

        tprint("🚀 Initializing Enhanced Monte Carlo Simulation Engine", "header")

        # Initialize hardware optimizers using BaseStep utilities
        try:
            self.gpu_manager = self.get_m1_gpu_manager() if config.enable_gpu_acceleration else None
            if config.enable_gpu_acceleration:
                self.gpu_accelerator = self.get_m1_gpu_accelerator()
            else:
                self.gpu_accelerator = None
        except Exception as e:
            tprint(f"⚠️  GPU acceleration unavailable: {e}", "warning")
            self.gpu_manager = None
            self.gpu_accelerator = None

        try:
            self.memory_optimizer = self.get_m1_memory_optimizer() if config.enable_memory_optimization else None
            if config.enable_memory_optimization:
                self.m1_memory_optimizer = self.get_m1_memory_optimizer_instance()
                self.m1_memory_optimizer.optimize_memory_for_ml()
            else:
                self.m1_memory_optimizer = None
        except Exception as e:
            tprint(f"⚠️  Memory optimization unavailable: {e}", "warning")
            self.memory_optimizer = None
            self.m1_memory_optimizer = None

        try:
            self.cpu_optimizer = self.get_m1_cpu_optimizer() if config.enable_parallel_processing else None
            if config.enable_parallel_processing:
                self.m1_cpu_optimizer = self.get_m1_cpu_optimizer_instance()
            else:
                self.m1_cpu_optimizer = None
        except Exception as e:
            tprint(f"⚠️  CPU optimization unavailable: {e}", "warning")
            self.cpu_optimizer = None
            self.m1_cpu_optimizer = None

        # Initialize matrix operations using BaseStep utilities
        try:
            self.matrix_ops = self.get_unified_matrix_operations()
            self.matrix_processor = self.get_hardware_optimized_matrix_processor()
            self.batch_processor = self.get_batch_matrix_processor(
                chunk_size_mb=config.chunk_size_mb,
                enable_gpu=config.enable_gpu_acceleration,
                enable_parallel=config.enable_parallel_processing,
                max_workers=config.max_workers
            )
        except Exception as e:
            tprint(f"⚠️  Matrix operations unavailable: {e}", "warning")
            self.matrix_ops = None
            self.matrix_processor = None
            self.batch_processor = None

        # Initialize VectorBT optimization utilities
        try:
            # Create VectorBT configuration
            vectorbt_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=config.enable_gpu_acceleration,
                enable_parallel=config.enable_parallel_processing,
                memory_efficient=config.enable_memory_optimization,
                max_memory_gb=8.0,
                chunk_size=config.chunk_size_mb * 1024,  # Convert MB to KB
                enable_monitoring=True,
                enable_profiling=False,
                batch_size=10000,
                enable_batch_processing=True,
                rolling_optimization_threshold=1000,
                enable_rolling_optimization=True
            )

            self.vectorization_manager = self.get_unified_vectorization_manager(vectorbt_config)
            self.rolling_optimizer = self.get_vectorbt_rolling_optimizer(
                enable_gpu=config.enable_gpu_acceleration,
                enable_parallel=config.enable_parallel_processing,
                memory_efficient=config.enable_memory_optimization,
                chunk_size=config.chunk_size_mb * 1024,
                fast_fail=True,
                enable_logging=True
            )
            tprint("✅ VectorBT optimization utilities initialized", "success")
        except Exception as e:
            tprint(f"⚠️  VectorBT optimization unavailable: {e}", "warning")
            self.vectorization_manager = None
            self.rolling_optimizer = None

        # Initialize CV and validation utilities using BaseStep utilities
        if config.enable_cv_validation:
            self.cv_validator = self.get_time_series_split_validator(
                n_splits=config.cv_folds,
                test_size=1.0 / config.cv_folds,
                embargo_pct=config.embargo_pct
            )
            self.oof_generator = self.create_enhanced_oof_generator(
                strategy=OOFStrategy.MEAN,
                n_folds=5,
                enable_confidence_intervals=True,
                enable_diversity_metrics=True,
                enable_leakage_detection=True
            )
            tprint("✅ CV utilities initialized", "success")
        else:
            self.cv_validator = None
            self.oof_generator = None

        if config.enable_leakage_detection:
            self.leakage_detector = self.get_data_leakage_detector()
            tprint("✅ Data leakage detector initialized", "success")
        else:
            self.leakage_detector = None

        # Initialize ML utilities using BaseStep utilities
        try:
            self.hpo_optimizer = self.get_hyperparameter_optimizer()
        except Exception:
            self.hpo_optimizer = None

        # Initialize base Monte Carlo engine if available
        if BASE_ENGINE_AVAILABLE:
            try:
                self.monte_carlo_engine = MonteCarloEngine()
            except Exception:
                self.monte_carlo_engine = None
        else:
            self.monte_carlo_engine = None

        # Results storage
        self.simulation_results = []
        self.risk_metrics = {}
        self.simulation_paths = []  # Store individual simulation paths

        # Performance monitoring
        self.performance_stats = {
            'vectorbt_operations': 0,
            'matrix_operations': 0,
            'standard_operations': 0,
            'total_simulations': 0,
            'total_time': 0.0,
            'memory_optimizations': 0,
            'gpu_operations': 0,
            'parallel_operations': 0,
            'errors': 0,
            'fallbacks': 0
        }

        # Configuration summary
        tprint(f"📊 Monte Carlo Configuration:", "info")
        tprint(f"   Simulations: {config.n_simulations:,}", "info")
        tprint(f"   Confidence level: {config.confidence_level:.1%}", "info")
        tprint(f"   Simulation horizon: {config.simulation_horizon} days", "info")
        tprint(f"   Mode: {config.mode.value}", "info")
        tprint(f"   Parallel processing: {config.enable_parallel_processing} ({config.max_workers} workers)", "info")
        tprint(f"   Hardware optimization: GPU={config.enable_gpu_acceleration}, "
              f"Memory={config.enable_memory_optimization}", "info")
        tprint(f"   Data validation: {config.enable_data_validation}", "info")
        tprint(f"   Leakage detection: {config.enable_leakage_detection}", "info")
        tprint(f"   CV validation: {config.enable_cv_validation} ({config.cv_folds} folds)", "info")

        tprint("✅ Monte Carlo Engine initialization complete", "success")

    async def run_simulation(self, returns_data: pd.Series, portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """Run comprehensive Monte Carlo simulation with validation."""
        start_time = time.time()
        tprint(f"🎲 Running {self.config.n_simulations:,} Monte Carlo Simulations", "header")
        tprint(f"   Mode: {self.config.mode.value}", "info")
        tprint(f"   Portfolio value: ${portfolio_value:,.2f}", "info")

        # Preview input data for troubleshooting
        tprint_data_preview(returns_data, "monte_carlo_input_returns", level="DEBUG", include_metadata=True)

        try:
            # Validate and prepare data
            prepared_data = self._prepare_and_validate_data(returns_data)

            if not prepared_data['valid']:
                tprint(f"❌ Data validation failed: {prepared_data.get('error', 'Unknown error')}", "error")
                raise ValueError(f"Data validation failed: {prepared_data.get('error')}")

            returns = prepared_data['returns']
            tprint(f"✅ Data validated: {len(returns)} samples", "success")
            tprint(f"   Mean return: {np.mean(returns):.4f}, Std: {np.std(returns):.4f}", "info")
            
            # Preview validated data
            tprint_data_preview(returns, "monte_carlo_validated_returns", level="INFO", include_metadata=True)

            # Check for data leakage if enabled
            if self.leakage_detector and self.config.enable_leakage_detection:
                self._check_data_leakage(returns)

            # Run simulations based on mode
            tprint(f"🔄 Running simulations ({self.config.mode.value} mode)", "info")
            if self.config.mode == MonteCarloMode.BOOTSTRAP:
                simulation_results = await self._bootstrap_simulation(returns, portfolio_value)
            elif self.config.mode == MonteCarloMode.PARAMETRIC:
                simulation_results = await self._parametric_simulation(returns, portfolio_value)
            elif self.config.mode == MonteCarloMode.HISTORICAL:
                simulation_results = await self._historical_simulation(returns, portfolio_value)
            elif self.config.mode == MonteCarloMode.HYBRID:
                simulation_results = await self._hybrid_simulation(returns, portfolio_value)
            else:
                raise ValueError(f"Unknown simulation mode: {self.config.mode}")

            tprint(f"✅ Completed {len(simulation_results):,} simulation scenarios", "success")
            
            # Preview simulation results
            tprint_data_preview(simulation_results, "monte_carlo_simulation_results", level="INFO", max_rows=10)

            # Calculate comprehensive metrics
            tprint("📊 Calculating risk metrics", "info")
            metrics = self._calculate_comprehensive_metrics(simulation_results, portfolio_value, returns)

            # Store results
            self.simulation_results = simulation_results
            self.risk_metrics = metrics.to_dict()
            
            # Preview calculated metrics
            tprint_data_preview(metrics.to_dict(), "monte_carlo_calculated_metrics", level="DEBUG", include_metadata=True)

            execution_time = time.time() - start_time

            tprint(f"✅ Monte Carlo Simulation Complete", "success")
            tprint(f"   Execution time: {execution_time:.2f}s", "info")
            tprint(f"   Scenarios: {len(simulation_results):,}", "info")
            tprint(f"   Mean return: {metrics.mean_return:.2%}", "info")
            tprint(f"   Sharpe ratio: {metrics.sharpe_ratio:.3f}", "info")
            tprint(f"   VaR ({self.config.var_confidence:.1%}): {metrics.var_value:.2%}", "info")
            tprint(f"   Max drawdown: {metrics.max_drawdown:.2%}", "info")

            result = {
                'simulation_results': simulation_results,
                'metrics': metrics,
                'risk_metrics': metrics.to_dict(),
                'n_simulations': self.config.n_simulations,
                'confidence_level': self.config.confidence_level,
                'execution_time': execution_time,
                'data_statistics': prepared_data.get('statistics', {})
            }

            # Save results if requested
            if self.config.save_results:
                self._save_results(result)

            return result

        except Exception as e:
            self.logger.error(f"❌ Monte Carlo simulation failed: {e}")
            tprint(f"❌ Monte Carlo simulation failed: {e}", "error")
            raise

    def _prepare_and_validate_data(self, returns_data: pd.Series) -> Dict[str, Any]:
        """Prepare and validate returns data for simulation"""
        try:
            tprint("📊 Validating input data", "info")
            
            # Preview raw input data
            tprint_data_preview(returns_data, "monte_carlo_raw_input_data", level="DEBUG", include_metadata=True)

            # Convert to array and remove NaN using BaseStep utilities
            returns = self.ensure_array(returns_data)
            returns = returns[~self.check_for_nans(returns)]
            returns = returns[~self.check_for_infs(returns)]
            
            # Preview cleaned data
            tprint_data_preview(returns, "monte_carlo_cleaned_returns", level="DEBUG", include_metadata=True)

            if len(returns) < self.config.min_samples:
                return {
                    'valid': False,
                    'error': f'Insufficient data: {len(returns)} < {self.config.min_samples}'
                }

            # Calculate statistics using BaseStep utilities
            statistics = {
                'n_samples': len(returns),
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
                'skewness': float(pd.Series(returns).skew()),
                'kurtosis': float(pd.Series(returns).kurtosis())
            }
            
            # Preview calculated statistics
            tprint_data_preview(statistics, "monte_carlo_data_statistics", level="DEBUG", include_metadata=True)

            # Check for suspicious patterns
            if statistics['std'] == 0:
                tprint("⚠️  Zero variance in returns", "warning")
                return {'valid': False, 'error': 'Zero variance in returns'}

            if abs(statistics['skewness']) > 5:
                tprint(f"⚠️  High skewness detected: {statistics['skewness']:.2f}", "warning")

            if abs(statistics['kurtosis']) > 10:
                tprint(f"⚠️  High kurtosis detected: {statistics['kurtosis']:.2f}", "warning")

            tprint(f"✅ Data validation passed", "success")

            return {
                'valid': True,
                'returns': returns,
                'statistics': statistics
            }

        except Exception as e:
            tprint(f"❌ Data validation failed: {e}", "error")
            return {'valid': False, 'error': str(e)}

    def _check_data_leakage(self, returns: np.ndarray):
        """Check for data leakage in returns data"""
        try:
            tprint("🔍 Checking for data leakage", "info")

            # Create simple features for leakage check
            X = pd.DataFrame({
                'return': returns,
                'return_lag1': np.roll(returns, 1),
                'return_lag2': np.roll(returns, 2)
            }).iloc[2:]  # Remove first 2 rows with invalid lags

            y = pd.Series(returns[2:] > 0)  # Binary: positive return or not

            leakage_results = self.leakage_detector.detect_leakage(X.values, y.values)

            if leakage_results.get('has_leakage', False):
                leakage_score = leakage_results.get('leakage_score', 0)
                tprint(f"⚠️  Potential data leakage detected: score={leakage_score:.4f}", "warning")
            else:
                tprint("✅ No data leakage detected", "success")

        except Exception as e:
            tprint(f"⚠️  Leakage detection failed: {e}", "warning")

    async def _bootstrap_simulation(self, returns: pd.Series, portfolio_value: float) -> List[float]:
        """Bootstrap simulation using historical returns."""
        self.logger.info("🔄 Running bootstrap simulation")
        
        # Preview bootstrap parameters
        bootstrap_params = {
            'n_simulations': self.config.n_simulations,
            'horizon': self.config.simulation_horizon,
            'sample_size': int(len(returns) * self.config.bootstrap_sample_size),
            'portfolio_value': portfolio_value
        }
        tprint_data_preview(bootstrap_params, "bootstrap_simulation_params", level="DEBUG", include_metadata=True)

        try:
            n_simulations = self.config.n_simulations
            horizon = self.config.simulation_horizon
            sample_size = int(len(returns) * self.config.bootstrap_sample_size)

            # Use hardware optimization if available
            if self.memory_optimizer:
                with self.memory_optimizer.optimize_for_workload("monte_carlo"):
                    return await self._run_bootstrap_optimized(returns, portfolio_value, n_simulations, horizon, sample_size)
            else:
                return await self._run_bootstrap_standard(returns, portfolio_value, n_simulations, horizon, sample_size)

        except Exception as e:
            self.logger.error(f"❌ Bootstrap simulation failed: {e}")
            raise

    async def _run_bootstrap_optimized(self, returns: pd.Series, portfolio_value: float,
                                    n_simulations: int, horizon: int, sample_size: int) -> List[float]:
        """Optimized bootstrap simulation using VectorBT and matrix operations."""
        try:
            # Use VectorBT rolling optimizer if available for enhanced performance
            if self.rolling_optimizer and self.vectorization_manager:
                tprint("🎯 Using VectorBT-optimized bootstrap simulation", "info")

                # Use VectorBT for efficient sampling and calculations
                # Generate random indices for bootstrap sampling
                random_indices = np.random.randint(0, len(returns), size=(n_simulations, horizon))

                # Sample returns using VectorBT-optimized operations
                sampled_returns = returns.values[random_indices]

                # Use VectorBT rolling operations for portfolio value calculations
                # Calculate cumulative returns using VectorBT rolling sum for each simulation
                portfolio_values = []

                # Process simulations in batches for memory efficiency
                batch_size = min(1000, n_simulations)
                for i in range(0, n_simulations, batch_size):
                    batch_end = min(i + batch_size, n_simulations)
                    batch_returns = sampled_returns[i:batch_end]

                    # Use VectorBT rolling sum for cumulative returns
                    batch_df = pd.DataFrame(batch_returns)
                    cumulative_returns = self.rolling_optimizer.rolling_sum(
                        batch_df, window=horizon
                    ).iloc[:, -1]  # Get the final cumulative return

                    # Calculate portfolio values for this batch
                    batch_portfolio_values = portfolio_value * (1 + cumulative_returns.values)
                    portfolio_values.extend(batch_portfolio_values.tolist())

                # Update performance stats
                self.performance_stats['vectorbt_operations'] += n_simulations
                self.performance_stats['total_simulations'] += n_simulations
                
                # Preview bootstrap results
                tprint_data_preview(portfolio_values, "bootstrap_simulation_results", level="DEBUG", max_rows=5)

                return portfolio_values

            # Fallback to matrix operations
            elif self.matrix_ops:
                tprint("🎯 Using matrix operations for bootstrap simulation", "info")

                # Generate random indices for bootstrap sampling
                random_indices = np.random.randint(0, len(returns), size=(n_simulations, horizon))

                # Sample returns using matrix operations
                sampled_returns = returns.values[random_indices]

                # Calculate portfolio values using vectorized operations
                portfolio_values = portfolio_value * np.prod(1 + sampled_returns, axis=1)

                # Update performance stats
                self.performance_stats['matrix_operations'] += n_simulations
                self.performance_stats['total_simulations'] += n_simulations

                return portfolio_values.tolist()
            else:
                return await self._run_bootstrap_standard(returns, portfolio_value, n_simulations, horizon, sample_size)

        except Exception as e:
            self.logger.error(f"❌ Optimized bootstrap simulation failed: {e}")
            tprint(f"❌ VectorBT bootstrap simulation failed: {e}", "error")
            # Fallback to standard implementation
            return await self._run_bootstrap_standard(returns, portfolio_value, n_simulations, horizon, sample_size)

    async def _run_bootstrap_standard(self, returns: pd.Series, portfolio_value: float,
                                    n_simulations: int, horizon: int, sample_size: int) -> List[float]:
        """Standard bootstrap simulation."""
        try:
            portfolio_values = []

            for _ in range(n_simulations):
                # Bootstrap sample
                sample_returns = returns.sample(n=horizon, replace=True)

                # Calculate portfolio value
                portfolio_value_sim = portfolio_value * (1 + sample_returns).prod()
                portfolio_values.append(portfolio_value_sim)

            return portfolio_values

        except Exception as e:
            self.logger.error(f"❌ Standard bootstrap simulation failed: {e}")
            raise

    async def _parametric_simulation(self, returns: pd.Series, portfolio_value: float) -> List[float]:
        """Parametric simulation using fitted distributions."""
        self.logger.info("📊 Running parametric simulation")
        
        # Preview parametric simulation parameters
        param_params = {
            'distribution': self.config.parametric_distribution,
            'n_simulations': self.config.n_simulations,
            'horizon': self.config.simulation_horizon,
            'portfolio_value': portfolio_value
        }
        tprint_data_preview(param_params, "parametric_simulation_params", level="DEBUG", include_metadata=True)

        try:
            # Fit distribution parameters
            if self.config.parametric_distribution == "normal":
                mu, sigma = returns.mean(), returns.std()
                simulated_returns = np.random.normal(mu, sigma, (self.config.n_simulations, self.config.simulation_horizon))
                
                # Preview fitted parameters
                fitted_params = {'mu': mu, 'sigma': sigma}
                tprint_data_preview(fitted_params, "fitted_normal_params", level="DEBUG", include_metadata=True)
            elif self.config.parametric_distribution == "t":
                from scipy import stats
                df, loc, scale = stats.t.fit(returns)
                simulated_returns = stats.t.rvs(df, loc, scale, size=(self.config.n_simulations, self.config.simulation_horizon))
            else:
                raise ValueError(f"Unknown parametric distribution: {self.config.parametric_distribution}")

            # Calculate portfolio values
            portfolio_values = portfolio_value * np.prod(1 + simulated_returns, axis=1)
            
            # Preview parametric simulation results
            tprint_data_preview(portfolio_values.tolist(), "parametric_simulation_results", level="DEBUG", max_rows=5)

            return portfolio_values.tolist()

        except Exception as e:
            self.logger.error(f"❌ Parametric simulation failed: {e}")
            raise

    async def _historical_simulation(self, returns: pd.Series, portfolio_value: float) -> List[float]:
        """Historical simulation using historical scenarios."""
        self.logger.info("📈 Running historical simulation")
        
        # Preview historical simulation parameters
        hist_params = {
            'n_simulations': self.config.n_simulations,
            'horizon': self.config.simulation_horizon,
            'portfolio_value': portfolio_value
        }
        tprint_data_preview(hist_params, "historical_simulation_params", level="DEBUG", include_metadata=True)

        try:
            # Use historical returns as scenarios
            historical_returns = returns.values
            n_scenarios = len(historical_returns)
            
            # Preview historical data
            tprint_data_preview(historical_returns, "historical_returns_data", level="DEBUG", max_rows=10)

            portfolio_values = []

            for _ in range(self.config.n_simulations):
                # Randomly select historical scenario
                scenario_returns = np.random.choice(historical_returns, size=self.config.simulation_horizon, replace=True)

            # Calculate portfolio value
            portfolio_value_sim = portfolio_value * (1 + scenario_returns).prod()
            portfolio_values.append(portfolio_value_sim)

            # Preview historical simulation results
            tprint_data_preview(portfolio_values, "historical_simulation_results", level="DEBUG", max_rows=5)

            return portfolio_values

        except Exception as e:
            self.logger.error(f"❌ Historical simulation failed: {e}")
            raise

    async def _hybrid_simulation(self, returns: pd.Series, portfolio_value: float) -> List[float]:
        """Hybrid simulation combining multiple methods."""
        self.logger.info("🔀 Running hybrid simulation")

        try:
            # Combine bootstrap and parametric methods
            n_bootstrap = int(self.config.n_simulations * 0.6)
            n_parametric = self.config.n_simulations - n_bootstrap

            # Bootstrap simulation
            bootstrap_results = await self._bootstrap_simulation(returns, portfolio_value)
            bootstrap_values = bootstrap_results[:n_bootstrap]

            # Parametric simulation
            parametric_results = await self._parametric_simulation(returns, portfolio_value)
            parametric_values = parametric_results[:n_parametric]

            # Combine results
            combined_results = bootstrap_values + parametric_values

            return combined_results

        except Exception as e:
            self.logger.error(f"❌ Hybrid simulation failed: {e}")
            raise

    def _calculate_comprehensive_metrics(self, simulation_results: List[float],
                                        initial_value: float,
                                        original_returns: np.ndarray) -> MonteCarloMetrics:
        """Calculate comprehensive risk and performance metrics with VectorBT optimization."""
        try:
            if not simulation_results:
                tprint("⚠️  No simulation results to calculate metrics", "warning")
                return MonteCarloMetrics()

            # Preview input data for metrics calculation
            tprint_data_preview(simulation_results, "metrics_input_simulation_results", level="DEBUG", max_rows=5)
            tprint_data_preview(original_returns, "metrics_input_original_returns", level="DEBUG", max_rows=10)

            # Validate simulation results
            results_array = ensure_array(simulation_results)
            results_array = results_array[~check_for_nans(results_array)]
            results_array = results_array[~check_for_infs(results_array)]
            
            # Preview validated results array
            tprint_data_preview(results_array, "metrics_validated_results_array", level="DEBUG", max_rows=5)

            if len(results_array) == 0:
                tprint("⚠️  No valid simulation results after filtering", "warning")
                return MonteCarloMetrics()

            # Calculate returns from portfolio values
            returns = (results_array - initial_value) / initial_value
            returns = returns[~check_for_nans(returns)]
            
            # Preview calculated returns
            tprint_data_preview(returns, "metrics_calculated_returns", level="DEBUG", max_rows=10)

            if len(returns) == 0:
                return MonteCarloMetrics()

            # Use VectorBT rolling optimizer for enhanced statistical calculations
            if self.rolling_optimizer and self.vectorization_manager:
                tprint("🎯 Using VectorBT-optimized metrics calculation", "info")

                # Convert returns to pandas Series for VectorBT processing
                returns_series = pd.Series(returns)

                # Use VectorBT for rolling statistics
                try:
                    # Calculate rolling statistics using VectorBT with optimized window sizes
                    window_size = min(20, len(returns))
                    large_window_size = min(50, len(returns))

                    # Use VectorBT for efficient rolling calculations
                    rolling_mean = self.rolling_optimizer.rolling_mean(returns_series, window=window_size)
                    rolling_std = self.rolling_optimizer.rolling_std(returns_series, window=window_size)
                    rolling_min = self.rolling_optimizer.rolling_min(returns_series, window=window_size)
                    rolling_max = self.rolling_optimizer.rolling_max(returns_series, window=window_size)

                    # Use VectorBT for quantile calculations (VaR) with full window
                    var_confidence = validate_probability(self.config.var_confidence)
                    var_value = self.rolling_optimizer.rolling_quantile(
                        returns_series, window=len(returns), q=var_confidence
                    ).iloc[-1]  # Get the final quantile value

                    # Calculate rolling skewness and kurtosis for tail risk analysis
                    rolling_skew = self.rolling_optimizer.rolling_skew(returns_series, window=large_window_size)
                    rolling_kurt = self.rolling_optimizer.rolling_kurt(returns_series, window=large_window_size)

                    # Use the last values from rolling calculations with validation
                    mean_return = float(rolling_mean.iloc[-1]) if not rolling_mean.empty and not pd.isna(rolling_mean.iloc[-1]) else float(np.mean(returns))
                    std_return = float(rolling_std.iloc[-1]) if not rolling_std.empty and not pd.isna(rolling_std.iloc[-1]) else float(np.std(returns))
                    min_return = float(rolling_min.iloc[-1]) if not rolling_min.empty and not pd.isna(rolling_min.iloc[-1]) else float(np.min(returns))
                    max_return = float(rolling_max.iloc[-1]) if not rolling_max.empty and not pd.isna(rolling_max.iloc[-1]) else float(np.max(returns))

                    # Update performance stats
                    self.performance_stats['vectorbt_operations'] += 6  # 6 rolling operations

                    tprint("✅ VectorBT metrics calculation completed", "success")

                except Exception as e:
                    tprint(f"⚠️  VectorBT metrics calculation failed, using standard methods: {e}", "warning")
                    # Fallback to standard calculations
                    mean_return = float(np.mean(returns))
                    std_return = float(np.std(returns))
                    min_return = float(np.min(returns))
                    max_return = float(np.max(returns))
                    var_confidence = validate_probability(self.config.var_confidence)
                    var_value = float(np.percentile(returns, var_confidence * 100))
            else:
                # Standard calculations
                mean_return = float(np.mean(returns))
                std_return = float(np.std(returns))
                min_return = float(np.min(returns))
                max_return = float(np.max(returns))
                var_confidence = validate_probability(self.config.var_confidence)
                var_value = float(np.percentile(returns, var_confidence * 100))

            # Additional statistics
            median_return = float(np.median(returns))

            # Validate statistics
            mean_return = validate_finite(mean_return, default=0.0)
            std_return = validate_positive(std_return, default=0.01)
            var_value = validate_finite(var_value, default=0.0)

            # Expected Shortfall (Conditional VaR)
            es_confidence = validate_probability(self.config.expected_shortfall_confidence)
            es_threshold = np.percentile(returns, es_confidence * 100)
            tail_returns = returns[returns <= es_threshold]
            es_value = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_value
            es_value = validate_finite(es_value, default=0.0)

            # Confidence intervals
            confidence_level = validate_probability(self.config.confidence_level)
            alpha = 1 - confidence_level
            lower_bound = float(np.percentile(returns, (alpha / 2) * 100))
            upper_bound = float(np.percentile(returns, (1 - alpha / 2) * 100))

            # Drawdown analysis using common_operations
            cumulative_returns = np.cumsum(returns)
            max_dd = calculate_max_drawdown(cumulative_returns)
            max_dd = validate_finite(max_dd, default=0.0)

            # Performance metrics using BaseStep utilities
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculate_sortino_ratio(returns)
            win_rate = self.calculate_win_rate(returns)
            profit_factor = self.calculate_profit_factor(returns)
            calmar_ratio = self.calculate_calmar_ratio(returns, max_dd)

            # Validate performance metrics using BaseStep utilities
            sharpe_ratio = self.validate_finite(sharpe_ratio, default=0.0)
            sortino_ratio = self.validate_finite(sortino_ratio, default=0.0)
            win_rate = self.validate_probability(win_rate)
            profit_factor = self.validate_positive(profit_factor, default=0.0)
            calmar_ratio = self.validate_finite(calmar_ratio, default=0.0)

            # Tail risk metrics
            tail_returns = returns[returns < var_value]
            tail_risk = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_value
            tail_ratio = safe_divide(tail_risk, std_return, default=0.0)

            metrics = MonteCarloMetrics(
                mean_return=mean_return,
                std_return=std_return,
                min_return=min_return,
                max_return=max_return,
                median_return=median_return,
                var_value=var_value,
                expected_shortfall=es_value,
                max_drawdown=max_dd,
                tail_risk=tail_risk,
                tail_ratio=tail_ratio,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                ci_lower=lower_bound,
                ci_upper=upper_bound,
                confidence_level=confidence_level,
                n_simulations=len(simulation_results),
                simulation_mode=self.config.mode.value
            )
            
            # Preview final calculated metrics
            tprint_data_preview(metrics.to_dict(), "monte_carlo_final_metrics", level="INFO", include_metadata=True)

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to calculate metrics: {e}")
            tprint(f"❌ Metrics calculation failed: {e}", "error")
            return MonteCarloMetrics()

    def _save_results(self, result: Dict[str, Any]):
        """Save simulation results to disk"""
        try:
            # Preview results before saving
            tprint_data_preview(result, "monte_carlo_results_to_save", level="DEBUG", include_metadata=True)
            
            results_path = Path(self.config.results_path)
            ensure_directory(str(results_path))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save summary as JSON
            summary = {
                'timestamp': timestamp,
                'n_simulations': result['n_simulations'],
                'confidence_level': result['confidence_level'],
                'execution_time': result['execution_time'],
                'risk_metrics': result['risk_metrics'],
                'data_statistics': result['data_statistics']
            }

            json_path = results_path / f"monte_carlo_summary_{timestamp}.json"
            safe_json_dump(summary, str(json_path))

            # Save full results as pickle
            pkl_path = results_path / f"monte_carlo_results_{timestamp}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)

            tprint(f"💾 Results saved to {results_path}", "success")

        except Exception as e:
            tprint(f"⚠️  Failed to save results: {e}", "warning")

    async def run_stress_test(self, returns_data: pd.Series, stress_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """Run comprehensive stress testing with specific scenarios."""
        tprint(f"💥 Running {len(stress_scenarios)} Stress Test Scenarios", "header")

        try:
            stress_results = {}
            baseline_return = self.risk_metrics.get('return_metrics', {}).get('mean', 0.0)

            for idx, (scenario_name, stress_factor) in enumerate(stress_scenarios.items(), 1):
                tprint(f"🔄 Scenario {idx}/{len(stress_scenarios)}: {scenario_name} (factor={stress_factor:.2f})", "info")

                # Validate stress factor
                stress_factor = validate_positive(stress_factor, default=1.0)

                # Apply stress factor to returns
                stressed_returns = returns_data * stress_factor

                # Run simulation with stressed data
                scenario_results = await self.run_simulation(stressed_returns)

                # Calculate impact
                scenario_return = scenario_results['risk_metrics'].get('return_metrics', {}).get('mean', 0)
                impact = scenario_return - baseline_return

                stress_results[scenario_name] = {
                    'stress_factor': stress_factor,
                    'results': scenario_results,
                    'impact': impact,
                    'relative_impact': safe_divide(impact, baseline_return, default=0.0)
                }

                tprint(f"   Impact: {impact:.2%} ({impact/baseline_return:.1%} relative)", "info")

            tprint(f"✅ Stress testing complete", "success")

            return stress_results

        except Exception as e:
            self.logger.error(f"❌ Stress testing failed: {e}")
            tprint(f"❌ Stress testing failed: {e}", "error")
            raise

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics including VectorBT usage."""
        try:
            stats = self.performance_stats.copy()

            # Add VectorBT-specific stats if available
            if self.rolling_optimizer:
                rolling_stats = self.rolling_optimizer.get_performance_stats()
                stats.update({
                    'vectorbt_rolling_operations': rolling_stats.get('vectorbt_operations', 0),
                    'vectorbt_gpu_operations': rolling_stats.get('gpu_operations', 0),
                    'vectorbt_memory_optimizations': rolling_stats.get('memory_optimizations', 0),
                    'vectorbt_errors': rolling_stats.get('errors', 0),
                    'vectorbt_avg_time_per_operation': rolling_stats.get('avg_time_per_operation', 0.0)
                })

            if self.vectorization_manager:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
                stats.update({
                    'unified_vectorization_operations': vectorization_stats.get('total_operations', 0),
                    'unified_vectorization_time': vectorization_stats.get('total_time', 0.0),
                    'unified_vectorization_memory_savings': vectorization_stats.get('memory_savings', 0.0),
                    'unified_vectorization_cache_hit_rate': vectorization_stats.get('cache_hit_rate', 0.0)
                })

            # Calculate efficiency metrics
            if stats['total_simulations'] > 0:
                stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_simulations']
                stats['matrix_usage_rate'] = stats['matrix_operations'] / stats['total_simulations']
                stats['standard_usage_rate'] = stats['standard_operations'] / stats['total_simulations']
                stats['avg_time_per_simulation'] = stats['total_time'] / stats['total_simulations']
            else:
                stats['vectorbt_usage_rate'] = 0.0
                stats['matrix_usage_rate'] = 0.0
                stats['standard_usage_rate'] = 0.0
                stats['avg_time_per_simulation'] = 0.0

            return stats

        except Exception as e:
            self.logger.error(f"❌ Failed to get performance stats: {e}")
            return self.performance_stats.copy()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive Monte Carlo report with validated metrics."""
        try:
            tprint("📋 Generating Monte Carlo Report", "header")

            if not self.simulation_results:
                tprint("⚠️  No simulation results available", "warning")
                return {'error': 'No simulation results available'}

            # Validate simulation results using BaseStep utilities
            results_array = self.ensure_array(self.simulation_results)
            results_array = results_array[~self.check_for_nans(results_array)]
            results_array = results_array[~self.check_for_infs(results_array)]

            report = {
                'simulation_config': {
                    'n_simulations': self.config.n_simulations,
                    'mode': self.config.mode.value,
                    'confidence_level': self.config.confidence_level,
                    'simulation_horizon': self.config.simulation_horizon,
                    'parallel_processing': self.config.enable_parallel_processing,
                    'hardware_acceleration': self.config.enable_gpu_acceleration
                },
                'risk_metrics': self.risk_metrics,
                'simulation_summary': {
                    'total_simulations': len(self.simulation_results),
                    'valid_simulations': len(results_array),
                    'mean_result': float(np.mean(results_array)),
                    'std_result': float(np.std(results_array)),
                    'min_result': float(np.min(results_array)),
                    'max_result': float(np.max(results_array)),
                    'median_result': float(np.median(results_array))
                },
                'hardware_performance': {
                    'gpu_enabled': self.gpu_accelerator is not None,
                    'memory_optimized': self.m1_memory_optimizer is not None,
                    'parallel_workers': self.config.max_workers if self.config.enable_parallel_processing else 1
                },
                'vectorbt_performance': {
                    'vectorbt_enabled': self.rolling_optimizer is not None,
                    'unified_vectorization_enabled': self.vectorization_manager is not None,
                    'performance_stats': self.get_performance_stats()
                }
            }

            # Add percentile analysis
            report['percentile_analysis'] = {
                f'p{int(p*100)}': float(np.percentile(results_array, p*100))
                for p in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            }

            tprint("✅ Report generated successfully", "success")
            tprint("📊 Key metrics:", "info")
            tprint(f"   Mean: ${report['simulation_summary']['mean_result']:,.2f}", "info")
            tprint(f"   Std: ${report['simulation_summary']['std_result']:,.2f}", "info")
            tprint(f"   Valid simulations: {report['simulation_summary']['valid_simulations']:,} / "
                  f"{report['simulation_summary']['total_simulations']:,}", "info")

            # Display VectorBT performance stats
            perf_stats = report['vectorbt_performance']['performance_stats']
            if perf_stats['vectorbt_operations'] > 0:
                tprint("🎯 VectorBT Performance:", "info")
                tprint(f"   VectorBT operations: {perf_stats['vectorbt_operations']:,}", "info")
                tprint(f"   VectorBT usage rate: {perf_stats['vectorbt_usage_rate']:.1%}", "info")
                tprint(f"   GPU operations: {perf_stats.get('vectorbt_gpu_operations', 0):,}", "info")
                tprint(f"   Memory optimizations: {perf_stats.get('vectorbt_memory_optimizations', 0):,}", "info")

            return report

        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            tprint(f"❌ Report generation failed: {e}", "error")
            return {'error': str(e)}

    # BaseStep utility methods
    def get_m1_gpu_manager(self):
        """Get M1 GPU manager using BaseStep utilities."""
        return self.get_utility('m1_gpu_manager')

    def get_m1_gpu_accelerator(self):
        """Get M1 GPU accelerator using BaseStep utilities."""
        return self.get_utility('m1_gpu_accelerator')

    def get_m1_memory_optimizer(self):
        """Get M1 memory optimizer using BaseStep utilities."""
        return self.get_utility('m1_memory_optimizer')

    def get_m1_memory_optimizer_instance(self):
        """Get M1 memory optimizer instance using BaseStep utilities."""
        return self.get_utility('m1_memory_optimizer_instance')

    def get_m1_cpu_optimizer(self):
        """Get M1 CPU optimizer using BaseStep utilities."""
        return self.get_utility('m1_cpu_optimizer')

    def get_m1_cpu_optimizer_instance(self):
        """Get M1 CPU optimizer instance using BaseStep utilities."""
        return self.get_utility('m1_cpu_optimizer_instance')

    def get_unified_matrix_operations(self):
        """Get unified matrix operations using BaseStep utilities."""
        return self.get_utility('unified_matrix_operations')

    def get_hardware_optimized_matrix_processor(self):
        """Get hardware optimized matrix processor using BaseStep utilities."""
        return self.get_utility('hardware_optimized_matrix_processor')

    def get_batch_matrix_processor(self, **kwargs):
        """Get batch matrix processor using BaseStep utilities."""
        return self.get_utility('batch_matrix_processor', **kwargs)

    def get_unified_vectorization_manager(self, config):
        """Get unified vectorization manager using BaseStep utilities."""
        return self.get_utility('unified_vectorization_manager', config)

    def get_vectorbt_rolling_optimizer(self, **kwargs):
        """Get VectorBT rolling optimizer using BaseStep utilities."""
        return self.get_utility('vectorbt_rolling_optimizer', **kwargs)

    def get_time_series_split_validator(self, **kwargs):
        """Get time series split validator using BaseStep utilities."""
        return self.get_utility('time_series_split_validator', **kwargs)

    def create_enhanced_oof_generator(self, **kwargs):
        """Create enhanced OOF generator using BaseStep utilities."""
        return self.get_utility('enhanced_oof_generator', **kwargs)

    def get_data_leakage_detector(self):
        """Get data leakage detector using BaseStep utilities."""
        return self.get_utility('data_leakage_detector')

    def get_hyperparameter_optimizer(self):
        """Get hyperparameter optimizer using BaseStep utilities."""
        return self.get_utility('hyperparameter_optimizer')

    # Math validation utilities
    def validate_finite(self, value, default=0.0):
        """Validate finite value using BaseStep utilities."""
        return self.get_utility('validate_finite', value, default)

    def validate_probability(self, value, default=0.0):
        """Validate probability value using BaseStep utilities."""
        return self.get_utility('validate_probability', value, default)

    def validate_positive(self, value, default=0.0):
        """Validate positive value using BaseStep utilities."""
        return self.get_utility('validate_positive', value, default)

    def check_for_nans(self, array):
        """Check for NaNs using BaseStep utilities."""
        return self.get_utility('check_for_nans', array)

    def check_for_infs(self, array):
        """Check for infinities using BaseStep utilities."""
        return self.get_utility('check_for_infs', array)

    def ensure_array(self, data):
        """Ensure array format using BaseStep utilities."""
        return self.get_utility('ensure_array', data)

    # Common operations utilities
    def calculate_sharpe_ratio(self, returns):
        """Calculate Sharpe ratio using BaseStep utilities."""
        return self.get_utility('calculate_sharpe_ratio', returns)

    def calculate_sortino_ratio(self, returns):
        """Calculate Sortino ratio using BaseStep utilities."""
        return self.get_utility('calculate_sortino_ratio', returns)

    def calculate_max_drawdown(self, returns):
        """Calculate max drawdown using BaseStep utilities."""
        return self.get_utility('calculate_max_drawdown', returns)

    def calculate_win_rate(self, returns):
        """Calculate win rate using BaseStep utilities."""
        return self.get_utility('calculate_win_rate', returns)

    def calculate_profit_factor(self, returns):
        """Calculate profit factor using BaseStep utilities."""
        return self.get_utility('calculate_profit_factor', returns)

    def calculate_calmar_ratio(self, returns, max_dd):
        """Calculate Calmar ratio using BaseStep utilities."""
        return self.get_utility('calculate_calmar_ratio', returns, max_dd)

# Convenience functions
async def run_monte_carlo_simulation(
    returns_data: pd.Series,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
    mode: MonteCarloMode = MonteCarloMode.HYBRID,
    **kwargs
) -> Dict[str, Any]:
    """Run Monte Carlo simulation with the given parameters."""
    config = RealMonteCarloConfig(
        n_simulations=n_simulations,
        confidence_level=confidence_level,
        mode=mode,
        **kwargs
    )

    engine = RealMonteCarloEngine(config)
    results = await engine.run_simulation(returns_data)

    return results
    return results
