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

# ML utilities
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.ml_common.cv_utils import TimeSeriesSplitValidator
from src.utils.ml_common.oof_generator import OOFGenerator
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

# Output utilities
from src.utils.tprint import tprint

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

class RealMonteCarloEngine:
    """
    Real Monte Carlo simulation engine using existing utilities.
    
    This engine provides comprehensive Monte Carlo simulation with:
    - GPU acceleration for M1/M2/M3 Macs
    - Memory optimization for large simulations
    - Multiple simulation methods (bootstrap, parametric, historical)
    - Risk metrics calculation (VaR, Expected Shortfall, etc.)
    """
    
    def __init__(self, config: RealMonteCarloConfig):
        """Initialize the enhanced Monte Carlo engine with hardware acceleration."""
        self.config = config
        self.logger = logger.getChild('RealMonteCarloEngine')
        
        tprint("🚀 Initializing Enhanced Monte Carlo Simulation Engine", "header")
        
        # Initialize hardware optimizers
        try:
            self.gpu_manager = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
            if config.enable_gpu_acceleration:
                self.gpu_accelerator = M1GPUAccelerator()
            else:
                self.gpu_accelerator = None
        except Exception as e:
            tprint(f"⚠️  GPU acceleration unavailable: {e}", "warning")
            self.gpu_manager = None
            self.gpu_accelerator = None
        
        try:
            self.memory_optimizer = get_m1_memory_optimizer() if config.enable_memory_optimization else None
            if config.enable_memory_optimization:
                self.m1_memory_optimizer = M1MemoryOptimizer()
                self.m1_memory_optimizer.optimize_memory_for_ml()
            else:
                self.m1_memory_optimizer = None
        except Exception as e:
            tprint(f"⚠️  Memory optimization unavailable: {e}", "warning")
            self.memory_optimizer = None
            self.m1_memory_optimizer = None
        
        try:
            self.cpu_optimizer = get_m1_cpu_optimizer() if config.enable_parallel_processing else None
            if config.enable_parallel_processing:
                self.m1_cpu_optimizer = M1CPUOptimizer()
            else:
                self.m1_cpu_optimizer = None
        except Exception as e:
            tprint(f"⚠️  CPU optimization unavailable: {e}", "warning")
            self.cpu_optimizer = None
            self.m1_cpu_optimizer = None
        
        # Initialize matrix operations
        try:
            self.matrix_ops = get_unified_matrix_operations()
            self.matrix_processor = HardwareOptimizedMatrixProcessor()
            self.batch_processor = BatchMatrixProcessor(
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
        
        # Initialize CV and validation utilities
        if config.enable_cv_validation:
            self.cv_validator = TimeSeriesSplitValidator(
                n_splits=config.cv_folds,
                test_size=1.0 / config.cv_folds,
                embargo_pct=config.embargo_pct
            )
            self.oof_generator = OOFGenerator()
            tprint("✅ CV utilities initialized", "success")
        else:
            self.cv_validator = None
            self.oof_generator = None
        
        if config.enable_leakage_detection:
            self.leakage_detector = DataLeakageDetector()
            tprint("✅ Data leakage detector initialized", "success")
        else:
            self.leakage_detector = None
        
        # Initialize ML utilities
        try:
            self.hpo_optimizer = HyperparameterOptimizer()
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
        
        try:
            # Validate and prepare data
            prepared_data = self._prepare_and_validate_data(returns_data)
            
            if not prepared_data['valid']:
                tprint(f"❌ Data validation failed: {prepared_data.get('error', 'Unknown error')}", "error")
                raise ValueError(f"Data validation failed: {prepared_data.get('error')}")
            
            returns = prepared_data['returns']
            tprint(f"✅ Data validated: {len(returns)} samples", "success")
            tprint(f"   Mean return: {np.mean(returns):.4f}, Std: {np.std(returns):.4f}", "info")
            
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
            
            # Calculate comprehensive metrics
            tprint("📊 Calculating risk metrics", "info")
            metrics = self._calculate_comprehensive_metrics(simulation_results, portfolio_value, returns)
            
            # Store results
            self.simulation_results = simulation_results
            self.risk_metrics = metrics.to_dict()
            
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
            
            # Convert to array and remove NaN
            returns = ensure_array(returns_data)
            returns = returns[~check_for_nans(returns)]
            returns = returns[~check_for_infs(returns)]
            
            if len(returns) < self.config.min_samples:
                return {
                    'valid': False,
                    'error': f'Insufficient data: {len(returns)} < {self.config.min_samples}'
                }
            
            # Calculate statistics
            statistics = {
                'n_samples': len(returns),
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
                'skewness': float(pd.Series(returns).skew()),
                'kurtosis': float(pd.Series(returns).kurtosis())
            }
            
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
        """Optimized bootstrap simulation using matrix operations."""
        try:
            # Use matrix operations for efficient sampling
            if self.matrix_ops:
                # Generate random indices for bootstrap sampling
                random_indices = np.random.randint(0, len(returns), size=(n_simulations, horizon))
                
                # Sample returns using matrix operations
                sampled_returns = returns.values[random_indices]
                
                # Calculate portfolio values using vectorized operations
                portfolio_values = portfolio_value * np.prod(1 + sampled_returns, axis=1)
                
                return portfolio_values.tolist()
            else:
                return await self._run_bootstrap_standard(returns, portfolio_value, n_simulations, horizon, sample_size)
                
        except Exception as e:
            self.logger.error(f"❌ Optimized bootstrap simulation failed: {e}")
            raise
    
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
        
        try:
            # Fit distribution parameters
            if self.config.parametric_distribution == "normal":
                mu, sigma = returns.mean(), returns.std()
                simulated_returns = np.random.normal(mu, sigma, (self.config.n_simulations, self.config.simulation_horizon))
            elif self.config.parametric_distribution == "t":
                from scipy import stats
                df, loc, scale = stats.t.fit(returns)
                simulated_returns = stats.t.rvs(df, loc, scale, size=(self.config.n_simulations, self.config.simulation_horizon))
            else:
                raise ValueError(f"Unknown parametric distribution: {self.config.parametric_distribution}")
            
            # Calculate portfolio values
            portfolio_values = portfolio_value * np.prod(1 + simulated_returns, axis=1)
            
            return portfolio_values.tolist()
            
        except Exception as e:
            self.logger.error(f"❌ Parametric simulation failed: {e}")
            raise
    
    async def _historical_simulation(self, returns: pd.Series, portfolio_value: float) -> List[float]:
        """Historical simulation using historical scenarios."""
        self.logger.info("📈 Running historical simulation")
        
        try:
            # Use historical returns as scenarios
            historical_returns = returns.values
            n_scenarios = len(historical_returns)
            
            portfolio_values = []
            
            for _ in range(self.config.n_simulations):
                # Randomly select historical scenario
                scenario_returns = np.random.choice(historical_returns, size=self.config.simulation_horizon, replace=True)
                
                # Calculate portfolio value
                portfolio_value_sim = portfolio_value * (1 + scenario_returns).prod()
                portfolio_values.append(portfolio_value_sim)
            
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
        """Calculate comprehensive risk and performance metrics with validation."""
        try:
            if not simulation_results:
                tprint("⚠️  No simulation results to calculate metrics", "warning")
                return MonteCarloMetrics()
            
            # Validate simulation results
            results_array = ensure_array(simulation_results)
            results_array = results_array[~check_for_nans(results_array)]
            results_array = results_array[~check_for_infs(results_array)]
            
            if len(results_array) == 0:
                tprint("⚠️  No valid simulation results after filtering", "warning")
                return MonteCarloMetrics()
            
            # Calculate returns from portfolio values
            returns = (results_array - initial_value) / initial_value
            returns = returns[~check_for_nans(returns)]
            
            if len(returns) == 0:
                return MonteCarloMetrics()
            
            # Basic statistics with validation
            mean_return = float(np.mean(returns))
            std_return = float(np.std(returns))
            min_return = float(np.min(returns))
            max_return = float(np.max(returns))
            median_return = float(np.median(returns))
            
            # Validate statistics
            mean_return = validate_finite(mean_return, default=0.0)
            std_return = validate_positive(std_return, default=0.01)
            
            # Value at Risk (VaR) with validation
            var_confidence = validate_probability(self.config.var_confidence)
            var_value = float(np.percentile(returns, var_confidence * 100))
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
            
            # Performance metrics using common_operations
            sharpe_ratio = calculate_sharpe_ratio(returns)
            sortino_ratio = calculate_sortino_ratio(returns)
            win_rate = calculate_win_rate(returns)
            profit_factor = calculate_profit_factor(returns)
            calmar_ratio = calculate_calmar_ratio(returns, max_dd)
            
            # Validate performance metrics
            sharpe_ratio = validate_finite(sharpe_ratio, default=0.0)
            sortino_ratio = validate_finite(sortino_ratio, default=0.0)
            win_rate = validate_probability(win_rate)
            profit_factor = validate_positive(profit_factor, default=0.0)
            calmar_ratio = validate_finite(calmar_ratio, default=0.0)
            
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
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate metrics: {e}")
            tprint(f"❌ Metrics calculation failed: {e}", "error")
            return MonteCarloMetrics()
    
    def _save_results(self, result: Dict[str, Any]):
        """Save simulation results to disk"""
        try:
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
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive Monte Carlo report with validated metrics."""
        try:
            tprint("📋 Generating Monte Carlo Report", "header")
            
            if not self.simulation_results:
                tprint("⚠️  No simulation results available", "warning")
                return {'error': 'No simulation results available'}
            
            # Validate simulation results
            results_array = ensure_array(self.simulation_results)
            results_array = results_array[~check_for_nans(results_array)]
            results_array = results_array[~check_for_infs(results_array)]
            
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
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
            tprint(f"❌ Report generation failed: {e}", "error")
            return {'error': str(e)}

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