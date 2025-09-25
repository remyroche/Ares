"""
Unified Monte Carlo Engine for Backtesting

This module provides unified Monte Carlo simulation functionality for
backtesting across TAS, NAS, and hybrid systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    
    # Simulation parameters
    n_simulations: int = 1000
    confidence_level: float = 0.95
    
    # Data parameters
    bootstrap_method: str = "block"  # "block", "simple", "stationary"
    block_size: int = 252  # 1 year for daily data
    
    # Model parameters
    model_type: str = "historical"  # "historical", "parametric", "garch"
    
    # Risk parameters
    enable_var_simulation: bool = True
    enable_stress_testing: bool = True
    
    # Output parameters
    enable_detailed_results: bool = True
    save_simulations: bool = False


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""
    
    # Simulation parameters
    n_simulations: int
    confidence_level: float
    
    # Results
    simulated_returns: np.ndarray
    simulated_final_values: np.ndarray
    
    # Statistics
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    percentile_5: float
    percentile_95: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    expected_value: float
    
    # Performance metrics
    probability_of_loss: float
    probability_of_target_return: float
    target_return: float = 0.05  # 5% target
    
    # Metadata
    execution_time: float
    model_type: str
    bootstrap_method: str


class MonteCarloEngine:
    """
    Unified Monte Carlo simulation engine for backtesting.
    
    Provides comprehensive Monte Carlo simulation for risk assessment,
    scenario analysis, and performance evaluation.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """Initialize the Monte Carlo engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_simulation(
        self,
        model: Any,
        data: pd.DataFrame,
        regime_info: Optional[Dict[str, Any]] = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for backtesting.
        
        Args:
            model: Trading model or strategy
            data: Historical market data
            regime_info: Regime information (optional)
            
        Returns:
            MonteCarloResult with simulation results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting Monte Carlo simulation with {self.config.n_simulations} iterations")
        
        try:
            # Prepare data for simulation
            returns_data = self._prepare_returns_data(data)
            
            # Run simulations
            simulated_returns = self._run_simulations(returns_data, regime_info)
            
            # Calculate final portfolio values
            initial_value = 100000  # Default initial capital
            simulated_final_values = initial_value * (1 + simulated_returns).prod(axis=1)
            
            # Calculate statistics
            statistics = self._calculate_statistics(simulated_returns, simulated_final_values)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(simulated_returns, simulated_final_values)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                simulated_returns, simulated_final_values
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = MonteCarloResult(
                n_simulations=self.config.n_simulations,
                confidence_level=self.config.confidence_level,
                simulated_returns=simulated_returns,
                simulated_final_values=simulated_final_values,
                mean_return=statistics['mean_return'],
                std_return=statistics['std_return'],
                min_return=statistics['min_return'],
                max_return=statistics['max_return'],
                percentile_5=statistics['percentile_5'],
                percentile_95=statistics['percentile_95'],
                var_95=risk_metrics['var_95'],
                cvar_95=risk_metrics['cvar_95'],
                expected_value=risk_metrics['expected_value'],
                probability_of_loss=performance_metrics['probability_of_loss'],
                probability_of_target_return=performance_metrics['probability_of_target_return'],
                target_return=self.config.confidence_level * 0.1,  # Default target
                execution_time=execution_time,
                model_type=self.config.model_type,
                bootstrap_method=self.config.bootstrap_method
            )
            
            self.logger.info(f"Monte Carlo simulation completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Monte Carlo simulation failed: {e}")
            raise
    
    def _prepare_returns_data(self, data: pd.DataFrame) -> pd.Series:
        """Prepare returns data for simulation."""
        # Extract returns from data
        if 'returns' in data.columns:
            returns = data['returns'].dropna()
        elif 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
        else:
            raise ValueError("No returns data found in input data")
        
        # Validate returns
        if len(returns) < 100:
            raise ValueError("Insufficient data for Monte Carlo simulation")
        
        # Remove extreme outliers
        q01 = returns.quantile(0.01)
        q99 = returns.quantile(0.99)
        returns = returns[(returns >= q01) & (returns <= q99)]
        
        return returns
    
    def _run_simulations(
        self,
        returns_data: pd.Series,
        regime_info: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Run Monte Carlo simulations."""
        n_periods = len(returns_data)
        simulated_returns = np.zeros((self.config.n_simulations, n_periods))
        
        for i in range(self.config.n_simulations):
            if self.config.bootstrap_method == "block":
                simulated_returns[i] = self._block_bootstrap(returns_data)
            elif self.config.bootstrap_method == "simple":
                simulated_returns[i] = self._simple_bootstrap(returns_data)
            elif self.config.bootstrap_method == "stationary":
                simulated_returns[i] = self._stationary_bootstrap(returns_data)
            else:
                simulated_returns[i] = self._parametric_simulation(returns_data)
        
        return simulated_returns
    
    def _block_bootstrap(self, returns_data: pd.Series) -> np.ndarray:
        """Block bootstrap simulation."""
        n_periods = len(returns_data)
        block_size = min(self.config.block_size, n_periods)
        n_blocks = int(np.ceil(n_periods / block_size))
        
        simulated_returns = np.zeros(n_periods)
        
        for i in range(n_blocks):
            start_idx = np.random.randint(0, n_periods - block_size + 1)
            end_idx = start_idx + block_size
            
            block_start = i * block_size
            block_end = min(block_start + block_size, n_periods)
            
            simulated_returns[block_start:block_end] = returns_data.iloc[start_idx:end_idx].values[
                :block_end - block_start
            ]
        
        return simulated_returns
    
    def _simple_bootstrap(self, returns_data: pd.Series) -> np.ndarray:
        """Simple bootstrap simulation."""
        n_periods = len(returns_data)
        return np.random.choice(returns_data.values, size=n_periods, replace=True)
    
    def _stationary_bootstrap(self, returns_data: pd.Series) -> np.ndarray:
        """Stationary bootstrap simulation."""
        n_periods = len(returns_data)
        returns_array = returns_data.values
        simulated_returns = np.zeros(n_periods)
        
        # Start with a random observation
        current_idx = np.random.randint(0, n_periods)
        simulated_returns[0] = returns_array[current_idx]
        
        # Generate subsequent observations
        for i in range(1, n_periods):
            # Decide whether to start a new block or continue current one
            if np.random.random() < (1 / self.config.block_size):
                # Start new block
                current_idx = np.random.randint(0, n_periods)
            else:
                # Continue current block
                current_idx = (current_idx + 1) % n_periods
            
            simulated_returns[i] = returns_array[current_idx]
        
        return simulated_returns
    
    def _parametric_simulation(self, returns_data: pd.Series) -> np.ndarray:
        """Parametric simulation using fitted distribution."""
        n_periods = len(returns_data)
        
        # Fit normal distribution
        mean_return = returns_data.mean()
        std_return = returns_data.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, n_periods)
        
        return simulated_returns
    
    def _calculate_statistics(
        self,
        simulated_returns: np.ndarray,
        simulated_final_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate simulation statistics."""
        total_returns = simulated_final_values / 100000 - 1  # Assuming initial value of 100k
        
        return {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'min_return': np.min(total_returns),
            'max_return': np.max(total_returns),
            'percentile_5': np.percentile(total_returns, 5),
            'percentile_95': np.percentile(total_returns, 95)
        }
    
    def _calculate_risk_metrics(
        self,
        simulated_returns: np.ndarray,
        simulated_final_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate risk metrics from simulations."""
        total_returns = simulated_final_values / 100000 - 1
        
        # Value at Risk
        var_95 = np.percentile(total_returns, 5)  # 5th percentile for 95% confidence
        
        # Conditional Value at Risk
        cvar_95 = total_returns[total_returns <= var_95].mean()
        
        # Expected value
        expected_value = np.mean(total_returns)
        
        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'expected_value': expected_value
        }
    
    def _calculate_performance_metrics(
        self,
        simulated_returns: np.ndarray,
        simulated_final_values: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics from simulations."""
        total_returns = simulated_final_values / 100000 - 1
        
        # Probability of loss
        probability_of_loss = np.mean(total_returns < 0)
        
        # Probability of achieving target return
        target_return = 0.05  # 5% target
        probability_of_target_return = np.mean(total_returns >= target_return)
        
        return {
            'probability_of_loss': probability_of_loss,
            'probability_of_target_return': probability_of_target_return
        }
    
    def run_stress_test(
        self,
        model: Any,
        data: pd.DataFrame,
        stress_scenarios: List[str]
    ) -> Dict[str, MonteCarloResult]:
        """Run stress test scenarios."""
        stress_results = {}
        
        for scenario in stress_scenarios:
            self.logger.info(f"Running stress test: {scenario}")
            
            # Modify data based on scenario
            stressed_data = self._apply_stress_scenario(data, scenario)
            
            # Run simulation on stressed data
            result = self.run_simulation(model, stressed_data)
            stress_results[scenario] = result
        
        return stress_results
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Apply stress scenario to data."""
        stressed_data = data.copy()
        
        if scenario == "market_crash":
            # Apply 50% market crash
            if 'close' in stressed_data.columns:
                stressed_data['close'] *= 0.5
                stressed_data['returns'] = stressed_data['close'].pct_change()
        
        elif scenario == "volatility_spike":
            # Increase volatility by 3x
            if 'returns' in stressed_data.columns:
                stressed_data['returns'] *= 3
        
        elif scenario == "correlation_breakdown":
            # Randomize returns (break correlations)
            if 'returns' in stressed_data.columns:
                np.random.seed(42)
                stressed_data['returns'] = np.random.normal(
                    stressed_data['returns'].mean(),
                    stressed_data['returns'].std(),
                    len(stressed_data['returns'])
                )
        
        return stressed_data
    
    def generate_report(self, result: MonteCarloResult) -> str:
        """Generate Monte Carlo simulation report."""
        report = []
        report.append("=" * 60)
        report.append("MONTE CARLO SIMULATION REPORT")
        report.append("=" * 60)
        
        # Simulation parameters
        report.append(f"\nSIMULATION PARAMETERS:")
        report.append(f"Number of Simulations: {result.n_simulations:,}")
        report.append(f"Confidence Level: {result.confidence_level:.1%}")
        report.append(f"Model Type: {result.model_type}")
        report.append(f"Bootstrap Method: {result.bootstrap_method}")
        
        # Results summary
        report.append(f"\nSIMULATION RESULTS:")
        report.append(f"Expected Return: {result.mean_return:.2%}")
        report.append(f"Return Std Dev: {result.std_return:.2%}")
        report.append(f"Min Return: {result.min_return:.2%}")
        report.append(f"Max Return: {result.max_return:.2%}")
        
        # Percentiles
        report.append(f"\nPERCENTILES:")
        report.append(f"5th Percentile: {result.percentile_5:.2%}")
        report.append(f"95th Percentile: {result.percentile_95:.2%}")
        
        # Risk metrics
        report.append(f"\nRISK METRICS:")
        report.append(f"VaR (95%): {result.var_95:.2%}")
        report.append(f"CVaR (95%): {result.cvar_95:.2%}")
        
        # Performance metrics
        report.append(f"\nPERFORMANCE METRICS:")
        report.append(f"Probability of Loss: {result.probability_of_loss:.2%}")
        report.append(f"Probability of {result.target_return:.1%} Return: {result.probability_of_target_return:.2%}")
        
        # Execution info
        report.append(f"\nEXECUTION INFO:")
        report.append(f"Execution Time: {result.execution_time:.2f} seconds")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)