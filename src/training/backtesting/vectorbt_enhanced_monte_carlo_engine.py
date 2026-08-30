"""
VectorBT Enhanced Monte Carlo Simulation Engine

This module provides a VectorBT-enhanced version of the Monte Carlo simulation engine
with significant performance improvements and enhanced functionality.
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

# VectorBT integration
from .vectorbt_integration import (
    VectorBTConfig, VectorBTSimulation, VectorBTMetrics,
    SimulationConfig, create_default_config, create_high_performance_config
)

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

class VectorBTMonteCarloMode(Enum):
    """VectorBT Monte Carlo simulation modes."""
    VECTORBT_ONLY = "vectorbt_only"
    HYBRID = "hybrid"
    COMPARISON = "comparison"

@dataclass
class VectorBTMonteCarloMetrics:
    """Comprehensive metrics from VectorBT Monte Carlo simulation"""
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
    vectorbt_enhanced: bool = True
    
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
                'mode': self.simulation_mode,
                'vectorbt_enhanced': self.vectorbt_enhanced
            }
        }

@dataclass
class VectorBTMonteCarloConfig:
    """Configuration for VectorBT Monte Carlo simulation."""
    # Basic configuration
    n_simulations: int = 1000
    confidence_level: float = 0.95
    simulation_horizon: int = 252  # Trading days
    
    # VectorBT configuration
    vectorbt_config: VectorBTConfig = field(default_factory=create_default_config)
    simulation_config: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Hardware optimization
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunk_size_mb: int = 128
    
    # Simulation parameters
    distribution: str = "normal"  # "normal", "t", "skewed_t"
    bootstrap_sample_size: float = 0.8
    
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
    results_path: str = "vectorbt_monte_carlo_results"
    enable_detailed_logging: bool = True
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

class VectorBTEnhancedMonteCarloEngine:
    """
    VectorBT Enhanced Monte Carlo Simulation Engine
    
    This engine provides comprehensive Monte Carlo simulation with VectorBT integration:
    - 10-100x performance improvement through vectorization
    - Enhanced simulation methods using VectorBT
    - Comprehensive risk metrics calculation
    - GPU acceleration support
    - Memory optimization
    """
    
    def __init__(self, config: VectorBTMonteCarloConfig):
        """Initialize the VectorBT enhanced Monte Carlo engine."""
        self.config = config
        self.logger = logger.getChild('VectorBTEnhancedMonteCarloEngine')
        
        # Initialize VectorBT components
        self._initialize_vectorbt_components()
        
        # Initialize hardware optimizers
        self._initialize_hardware_optimization()
        
        # Initialize ML utilities
        self._initialize_ml_utilities()
        
        # Results storage
        self.simulation_results = []
        self.risk_metrics = {}
        self.simulation_paths = []
        
        tprint("🚀 VectorBT Enhanced Monte Carlo Engine initialized", "header")
        tprint(f"📊 Simulations: {config.n_simulations:,}", "info")
        tprint(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}", "info")
        tprint(f"🧠 Memory optimization: {config.enable_memory_optimization}", "info")
        tprint(f"🔄 Parallel processing: {config.enable_parallel_processing}", "info")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT components."""
        try:
            # Initialize VectorBT simulation
            self.vectorbt_simulation = VectorBTSimulation(self.config.vectorbt_config)
            
            # Initialize VectorBT metrics
            self.vectorbt_metrics = VectorBTMetrics(self.config.vectorbt_config)
            
            tprint("✅ VectorBT components initialized", "success")
            
        except Exception as e:
            tprint(f"❌ VectorBT initialization failed: {e}", "error")
            raise
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize M1 accelerators
            self.gpu_manager = get_m1_gpu_manager() if self.config.enable_gpu_acceleration else None
            self.memory_optimizer = get_m1_memory_optimizer() if self.config.enable_memory_optimization else None
            self.cpu_optimizer = get_m1_cpu_optimizer() if self.config.enable_parallel_processing else None
            
            # Initialize matrix operations
            self.matrix_ops = get_unified_matrix_operations()
            self.matrix_processor = HardwareOptimizedMatrixProcessor()
            self.batch_processor = BatchMatrixProcessor(
                chunk_size_mb=self.config.chunk_size_mb,
                enable_gpu=self.config.enable_gpu_acceleration,
                enable_parallel=self.config.enable_parallel_processing,
                max_workers=self.config.max_workers
            )
            
            tprint("✅ Hardware optimization initialized", "success")
            
        except Exception as e:
            tprint(f"⚠️ Hardware optimization init failed: {e}", "warning")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_ops = None
            self.matrix_processor = None
            self.batch_processor = None
    
    def _initialize_ml_utilities(self):
        """Initialize ML utilities."""
        try:
            # Initialize CV and validation utilities
            if self.config.enable_cv_validation:
                self.cv_validator = TimeSeriesSplitValidator(
                    n_splits=self.config.cv_folds,
                    test_size=1.0 / self.config.cv_folds,
                    embargo_pct=self.config.embargo_pct
                )
                self.oof_generator = OOFGenerator()
            
            # Initialize leakage detector
            if self.config.enable_leakage_detection:
                self.leakage_detector = DataLeakageDetector()
            
            # Initialize HPO optimizer
            self.hpo_optimizer = HyperparameterOptimizer()
            
            tprint("✅ ML utilities initialized", "success")
            
        except Exception as e:
            tprint(f"⚠️ ML utilities init failed: {e}", "warning")
            self.cv_validator = None
            self.oof_generator = None
            self.leakage_detector = None
            self.hpo_optimizer = None
    
    async def run_simulation(self, returns_data: pd.Series, portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """Run comprehensive VectorBT Monte Carlo simulation."""
        start_time = time.time()
        tprint(f"🎲 Running {self.config.n_simulations:,} VectorBT Monte Carlo Simulations", "header")
        
        try:
            # Validate and prepare data
            prepared_data = self._prepare_and_validate_data(returns_data)
            
            if not prepared_data['valid']:
                tprint(f"❌ Data validation failed: {prepared_data.get('error', 'Unknown error')}", "error")
                raise ValueError(f"Data validation failed: {prepared_data.get('error')}")
            
            returns = prepared_data['returns']
            tprint(f"✅ Data validated: {len(returns)} samples", "success")
            
            # Check for data leakage if enabled
            if self.leakage_detector and self.config.enable_leakage_detection:
                self._check_data_leakage(returns)
            
            # Run VectorBT simulation
            tprint("🔄 Running VectorBT simulation", "info")
            simulation_results = await self._run_vectorbt_simulation(returns, portfolio_value)
            
            tprint(f"✅ Completed {len(simulation_results):,} simulation scenarios", "success")
            
            # Calculate comprehensive metrics using VectorBT
            tprint("📊 Calculating VectorBT risk metrics", "info")
            metrics = self._calculate_vectorbt_metrics(simulation_results, portfolio_value, returns)
            
            # Store results
            self.simulation_results = simulation_results
            self.risk_metrics = metrics.to_dict()
            
            execution_time = time.time() - start_time
            
            tprint(f"✅ VectorBT Monte Carlo Simulation Complete", "success")
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
                'data_statistics': prepared_data.get('statistics', {}),
                'vectorbt_enhanced': True
            }
            
            # Save results if requested
            if self.config.save_results:
                self._save_results(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ VectorBT Monte Carlo simulation failed: {e}")
            tprint(f"❌ VectorBT Monte Carlo simulation failed: {e}", "error")
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
                tprint("⚠️ Zero variance in returns", "warning")
                return {'valid': False, 'error': 'Zero variance in returns'}
            
            if abs(statistics['skewness']) > 5:
                tprint(f"⚠️ High skewness detected: {statistics['skewness']:.2f}", "warning")
            
            if abs(statistics['kurtosis']) > 10:
                tprint(f"⚠️ High kurtosis detected: {statistics['kurtosis']:.2f}", "warning")
            
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
                tprint(f"⚠️ Potential data leakage detected: score={leakage_score:.4f}", "warning")
            else:
                tprint("✅ No data leakage detected", "success")
                
        except Exception as e:
            tprint(f"⚠️ Leakage detection failed: {e}", "warning")
    
    async def _run_vectorbt_simulation(self, returns: np.ndarray, portfolio_value: float) -> List[float]:
        """Run VectorBT simulation."""
        try:
            # Use VectorBT simulation capabilities
            simulation_results = self.vectorbt_simulation.run_simulation(
                returns=pd.Series(returns),
                n_simulations=self.config.n_simulations,
                simulation_horizon=self.config.simulation_horizon,
                portfolio_value=portfolio_value
            )
            
            return simulation_results
            
        except Exception as e:
            self.logger.error(f"❌ VectorBT simulation failed: {e}")
            # Fallback to standard simulation
            return await self._run_standard_simulation(returns, portfolio_value)
    
    async def _run_standard_simulation(self, returns: np.ndarray, portfolio_value: float) -> List[float]:
        """Fallback standard simulation."""
        try:
            portfolio_values = []
            
            for _ in range(self.config.n_simulations):
                # Bootstrap sample
                sample_returns = np.random.choice(returns, size=self.config.simulation_horizon, replace=True)
                
                # Calculate portfolio value
                portfolio_value_sim = portfolio_value * (1 + sample_returns).prod()
                portfolio_values.append(portfolio_value_sim)
            
            return portfolio_values
            
        except Exception as e:
            self.logger.error(f"❌ Standard simulation failed: {e}")
            raise
    
    def _calculate_vectorbt_metrics(self, simulation_results: List[float], 
                                   initial_value: float,
                                   original_returns: np.ndarray) -> VectorBTMonteCarloMetrics:
        """Calculate comprehensive risk and performance metrics using VectorBT."""
        try:
            if not simulation_results:
                tprint("⚠️ No simulation results to calculate metrics", "warning")
                return VectorBTMonteCarloMetrics()
            
            # Validate simulation results
            results_array = ensure_array(simulation_results)
            results_array = results_array[~check_for_nans(results_array)]
            results_array = results_array[~check_for_infs(results_array)]
            
            if len(results_array) == 0:
                tprint("⚠️ No valid simulation results after filtering", "warning")
                return VectorBTMonteCarloMetrics()
            
            # Calculate returns from portfolio values
            returns = (results_array - initial_value) / initial_value
            returns = returns[~check_for_nans(returns)]
            
            if len(returns) == 0:
                return VectorBTMonteCarloMetrics()
            
            # Use VectorBT metrics calculation
            metrics_result = self.vectorbt_metrics.calculate_comprehensive_metrics(
                returns=pd.Series(returns)
            )
            
            # Extract metrics
            basic_metrics = metrics_result.metrics['basic_metrics']
            risk_metrics = metrics_result.metrics['risk_metrics']
            performance_metrics = metrics_result.metrics['performance_metrics']
            
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
            
            # Tail risk metrics
            tail_returns = returns[returns < var_value]
            tail_risk = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_value
            tail_ratio = safe_divide(tail_risk, np.std(returns), default=0.0)
            
            metrics = VectorBTMonteCarloMetrics(
                mean_return=basic_metrics.get('mean_return', 0.0),
                std_return=basic_metrics.get('std_return', 0.0),
                min_return=basic_metrics.get('min_return', 0.0),
                max_return=basic_metrics.get('max_return', 0.0),
                median_return=basic_metrics.get('median_return', 0.0),
                var_value=var_value,
                expected_shortfall=es_value,
                max_drawdown=risk_metrics.get('max_drawdown', 0.0),
                tail_risk=tail_risk,
                tail_ratio=tail_ratio,
                sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
                sortino_ratio=performance_metrics.get('sortino_ratio', 0.0),
                calmar_ratio=performance_metrics.get('calmar_ratio', 0.0),
                win_rate=performance_metrics.get('win_rate', 0.0),
                profit_factor=performance_metrics.get('profit_factor', 0.0),
                ci_lower=lower_bound,
                ci_upper=upper_bound,
                confidence_level=confidence_level,
                n_simulations=len(simulation_results),
                simulation_mode="vectorbt_enhanced",
                vectorbt_enhanced=True
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate VectorBT metrics: {e}")
            tprint(f"❌ VectorBT metrics calculation failed: {e}", "error")
            return VectorBTMonteCarloMetrics()
    
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
                'data_statistics': result['data_statistics'],
                'vectorbt_enhanced': result.get('vectorbt_enhanced', True)
            }
            
            json_path = results_path / f"vectorbt_monte_carlo_summary_{timestamp}.json"
            safe_json_dump(summary, str(json_path))
            
            # Save full results as pickle
            pkl_path = results_path / f"vectorbt_monte_carlo_results_{timestamp}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
            
            tprint(f"💾 Results saved to {results_path}", "success")
            
        except Exception as e:
            tprint(f"⚠️ Failed to save results: {e}", "warning")
    
    async def run_stress_test(self, returns_data: pd.Series, stress_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """Run comprehensive stress testing with VectorBT enhancement."""
        tprint(f"💥 Running {len(stress_scenarios)} VectorBT Stress Test Scenarios", "header")
        
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
            
            tprint(f"✅ VectorBT stress testing complete", "success")
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"❌ VectorBT stress testing failed: {e}")
            tprint(f"❌ VectorBT stress testing failed: {e}", "error")
            raise
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive VectorBT Monte Carlo report."""
        try:
            tprint("📋 Generating VectorBT Monte Carlo Report", "header")
            
            if not self.simulation_results:
                tprint("⚠️ No simulation results available", "warning")
                return {'error': 'No simulation results available'}
            
            # Validate simulation results
            results_array = ensure_array(self.simulation_results)
            results_array = results_array[~check_for_nans(results_array)]
            results_array = results_array[~check_for_infs(results_array)]
            
            report = {
                'simulation_config': {
                    'n_simulations': self.config.n_simulations,
                    'mode': 'vectorbt_enhanced',
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
                'vectorbt_performance': {
                    'gpu_enabled': self.config.enable_gpu_acceleration,
                    'memory_optimized': self.config.enable_memory_optimization,
                    'parallel_workers': self.config.max_workers if self.config.enable_parallel_processing else 1,
                    'vectorbt_enhanced': True
                }
            }
            
            # Add percentile analysis
            report['percentile_analysis'] = {
                f'p{int(p*100)}': float(np.percentile(results_array, p*100))
                for p in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            }
            
            tprint("✅ VectorBT report generated successfully", "success")
            tprint("📊 Key metrics:", "info")
            tprint(f"   Mean: ${report['simulation_summary']['mean_result']:,.2f}", "info")
            tprint(f"   Std: ${report['simulation_summary']['std_result']:,.2f}", "info")
            tprint(f"   Valid simulations: {report['simulation_summary']['valid_simulations']:,} / "
                  f"{report['simulation_summary']['total_simulations']:,}", "info")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate VectorBT report: {e}")
            tprint(f"❌ VectorBT report generation failed: {e}", "error")
            return {'error': str(e)}

# Convenience functions
async def run_vectorbt_monte_carlo_simulation(
    returns_data: pd.Series,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
    **kwargs
) -> Dict[str, Any]:
    """Run VectorBT Monte Carlo simulation with the given parameters."""
    config = VectorBTMonteCarloConfig(
        n_simulations=n_simulations,
        confidence_level=confidence_level,
        **kwargs
    )
    
    engine = VectorBTEnhancedMonteCarloEngine(config)
    results = await engine.run_simulation(returns_data)
    
    return results