"""
Real Monte Carlo Simulation Engine

This module provides comprehensive Monte Carlo simulation for backtesting using
existing utilities from src/utils/ for hardware optimization and ML validation.
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import existing utilities
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.matrix_operations.unified_operations import get_unified_matrix_operations
from src.utils.ml_common.optimization import HyperparameterOptimizer
from src.utils.common_ml.backtesting.monte_carlo_engine import MonteCarloEngine, MonteCarloConfig
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import safe_divide, safe_log, safe_sqrt, validate_finite
from src.core.decorators import handles_errors, traced, log_execution_time

logger = logging.getLogger(__name__)

class MonteCarloMode(Enum):
    """Monte Carlo simulation modes."""
    BOOTSTRAP = "bootstrap"
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    HYBRID = "hybrid"

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
    max_workers: int = 4
    
    # Simulation parameters
    mode: MonteCarloMode = MonteCarloMode.HYBRID
    bootstrap_sample_size: float = 0.8
    parametric_distribution: str = "normal"  # "normal", "t", "skewed_t"
    
    # Risk parameters
    var_confidence: float = 0.05
    expected_shortfall_confidence: float = 0.01
    max_drawdown_threshold: float = 0.2
    
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
        """Initialize the real Monte Carlo engine."""
        self.config = config
        self.logger = logger.getChild('RealMonteCarloEngine')
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager() if config.enable_gpu_acceleration else None
        self.memory_optimizer = get_m1_memory_optimizer() if config.enable_memory_optimization else None
        self.cpu_optimizer = get_m1_cpu_optimizer() if config.enable_parallel_processing else None
        
        # Initialize matrix operations
        self.matrix_ops = get_unified_matrix_operations()
        
        # Initialize ML utilities
        self.hpo_optimizer = HyperparameterOptimizer()
        
        # Initialize Monte Carlo engine
        self.monte_carlo_engine = MonteCarloEngine()
        
        # Results storage
        self.simulation_results = []
        self.risk_metrics = {}
        
    async def run_simulation(self, returns_data: pd.Series, portfolio_value: float = 100000.0) -> Dict[str, Any]:
        """Run Monte Carlo simulation on returns data."""
        self.logger.info(f"🎲 Running {self.config.n_simulations} Monte Carlo simulations")
        
        try:
            # Prepare data
            returns = returns_data.dropna()
            if len(returns) < 30:
                raise ValueError("Insufficient data for Monte Carlo simulation")
            
            # Run simulations based on mode
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
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(simulation_results, portfolio_value)
            
            # Store results
            self.simulation_results = simulation_results
            self.risk_metrics = risk_metrics
            
            self.logger.info(f"✅ Monte Carlo simulation completed: {len(simulation_results)} scenarios")
            
            return {
                'simulation_results': simulation_results,
                'risk_metrics': risk_metrics,
                'n_simulations': self.config.n_simulations,
                'confidence_level': self.config.confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"❌ Monte Carlo simulation failed: {e}")
            raise
    
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
    
    def _calculate_risk_metrics(self, simulation_results: List[float], initial_value: float) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        try:
            if not simulation_results:
                return {}
            
            results_array = np.array(simulation_results)
            returns = (results_array - initial_value) / initial_value
            
            # Basic statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            min_return = np.min(returns)
            max_return = np.max(returns)
            
            # Value at Risk (VaR)
            var_confidence = self.config.var_confidence
            var_value = np.percentile(returns, var_confidence * 100)
            
            # Expected Shortfall (Conditional VaR)
            es_confidence = self.config.expected_shortfall_confidence
            es_threshold = np.percentile(returns, es_confidence * 100)
            es_value = np.mean(returns[returns <= es_threshold])
            
            # Confidence intervals
            confidence_level = self.config.confidence_level
            alpha = 1 - confidence_level
            lower_bound = np.percentile(returns, (alpha / 2) * 100)
            upper_bound = np.percentile(returns, (1 - alpha / 2) * 100)
            
            # Drawdown analysis
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Tail risk metrics
            tail_risk = np.mean(returns[returns < var_value])
            tail_ratio = tail_risk / std_return if std_return > 0 else 0
            
            return {
                'mean_return': mean_return,
                'std_return': std_return,
                'min_return': min_return,
                'max_return': max_return,
                'var_value': var_value,
                'expected_shortfall': es_value,
                'confidence_interval': (lower_bound, upper_bound),
                'max_drawdown': max_drawdown,
                'tail_risk': tail_risk,
                'tail_ratio': tail_ratio,
                'n_simulations': len(simulation_results),
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate risk metrics: {e}")
            return {}
    
    async def run_stress_test(self, returns_data: pd.Series, stress_scenarios: Dict[str, float]) -> Dict[str, Any]:
        """Run stress testing with specific scenarios."""
        self.logger.info("💥 Running stress tests")
        
        try:
            stress_results = {}
            
            for scenario_name, stress_factor in stress_scenarios.items():
                # Apply stress factor to returns
                stressed_returns = returns_data * stress_factor
                
                # Run simulation with stressed data
                scenario_results = await self.run_simulation(stressed_returns)
                stress_results[scenario_name] = {
                    'stress_factor': stress_factor,
                    'results': scenario_results,
                    'impact': scenario_results['risk_metrics'].get('mean_return', 0) - 
                             self.risk_metrics.get('mean_return', 0)
                }
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"❌ Stress testing failed: {e}")
            raise
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive Monte Carlo report."""
        try:
            if not self.simulation_results:
                return {'error': 'No simulation results available'}
            
            report = {
                'simulation_config': {
                    'n_simulations': self.config.n_simulations,
                    'mode': self.config.mode.value,
                    'confidence_level': self.config.confidence_level,
                    'simulation_horizon': self.config.simulation_horizon
                },
                'risk_metrics': self.risk_metrics,
                'simulation_summary': {
                    'total_simulations': len(self.simulation_results),
                    'mean_result': np.mean(self.simulation_results),
                    'std_result': np.std(self.simulation_results),
                    'min_result': np.min(self.simulation_results),
                    'max_result': np.max(self.simulation_results)
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate report: {e}")
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