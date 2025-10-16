"""
Monte Carlo Simulation for TAS

Comprehensive Monte Carlo simulation for tree architecture search including
portfolio simulation, risk analysis, and scenario generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MonteCarloMethod(Enum):
    """Monte Carlo simulation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    BOOTSTRAP = "bootstrap"
    REGIME_BASED = "regime_based"
    FACTOR_BASED = "factor_based"


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    
    # Simulation parameters
    n_simulations: int = 10000
    simulation_horizon: int = 252  # Trading days
    method: MonteCarloMethod = MonteCarloMethod.PARAMETRIC
    
    # Data parameters
    use_regime_data: bool = True
    use_factor_data: bool = True
    regime_transition_probability: float = 0.1
    
    # Risk parameters
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_methods: List[str] = field(default_factory=lambda: ['historical', 'parametric'])
    
    # Advanced parameters
    enable_copula_simulation: bool = False
    enable_garch_simulation: bool = False
    enable_jump_diffusion: bool = False
    
    # Output parameters
    save_simulation_results: bool = True
    simulation_directory: str = "monte_carlo_results"
    detailed_simulation: bool = True


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    
    # Simulation results
    simulated_returns: np.ndarray
    simulated_equity_curves: np.ndarray
    simulation_statistics: Dict[str, float]
    
    # Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_return: float
    expected_volatility: float
    
    # Percentiles
    percentiles: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Regime analysis
    regime_simulations: Dict[str, np.ndarray]
    regime_statistics: Dict[str, Dict[str, float]]
    
    # Factor analysis
    factor_simulations: Dict[str, np.ndarray]
    factor_statistics: Dict[str, Dict[str, float]]
    
    # Time series
    simulation_returns: pd.Series
    simulation_volatility: pd.Series
    simulation_drawdown: pd.Series
    
    # Metadata
    simulation_period: Tuple[datetime, datetime]
    execution_time: float
    config: MonteCarloConfig


class MonteCarloSimulator:
    """
    Comprehensive Monte Carlo simulator for TAS.
    
    Provides portfolio simulation, risk analysis, and scenario generation
    for tree architecture search using various Monte Carlo methods.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """Initialize Monte Carlo simulator.
        
        Args:
            config: Monte Carlo configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Simulation state
        self.results = None
        self.simulation_data = None
        
        self.logger.info("✅ Monte Carlo Simulator initialized")
        self.logger.info(f"📊 Simulations: {config.n_simulations}")
        self.logger.info(f"📊 Horizon: {config.simulation_horizon} days")
        self.logger.info(f"📊 Method: {config.method.value}")
    
    def run_simulation(self, 
                      returns_series: pd.Series,
                      regime_data: Optional[Dict[str, Any]] = None,
                      factor_data: Optional[Dict[str, pd.Series]] = None,
                      benchmark_returns: Optional[pd.Series] = None) -> MonteCarloResult:
        """
        Run comprehensive Monte Carlo simulation.
        
        Args:
            returns_series: Portfolio returns series
            regime_data: Optional regime information
            factor_data: Optional factor data
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Monte Carlo simulation result
        """
        self.logger.info("🚀 Starting Monte Carlo simulation")
        start_time = datetime.now()
        
        try:
            # Validate data
            self._validate_data(returns_series, regime_data, factor_data, benchmark_returns)
            
            # Prepare simulation data
            simulation_data = self._prepare_simulation_data(returns_series, regime_data, factor_data, benchmark_returns)
            
            # Run Monte Carlo simulation
            if self.config.method == MonteCarloMethod.HISTORICAL:
                simulated_returns = self._run_historical_simulation(simulation_data)
            elif self.config.method == MonteCarloMethod.PARAMETRIC:
                simulated_returns = self._run_parametric_simulation(simulation_data)
            elif self.config.method == MonteCarloMethod.BOOTSTRAP:
                simulated_returns = self._run_bootstrap_simulation(simulation_data)
            elif self.config.method == MonteCarloMethod.REGIME_BASED:
                simulated_returns = self._run_regime_based_simulation(simulation_data)
            elif self.config.method == MonteCarloMethod.FACTOR_BASED:
                simulated_returns = self._run_factor_based_simulation(simulation_data)
            else:
                raise ValueError(f"Unknown simulation method: {self.config.method}")
            
            # Calculate simulation statistics
            simulation_statistics = self._calculate_simulation_statistics(simulated_returns)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(simulated_returns)
            
            # Calculate percentiles and confidence intervals
            percentiles = self._calculate_percentiles(simulated_returns)
            confidence_intervals = self._calculate_confidence_intervals(simulated_returns)
            
            # Analyze regime simulations
            regime_analysis = self._analyze_regime_simulations(simulated_returns, simulation_data)
            
            # Analyze factor simulations
            factor_analysis = self._analyze_factor_simulations(simulated_returns, simulation_data)
            
            # Create time series
            time_series = self._create_time_series(simulated_returns)
            
            # Create comprehensive result
            result = MonteCarloResult(
                # Simulation results
                simulated_returns=simulated_returns,
                simulated_equity_curves=self._calculate_equity_curves(simulated_returns),
                simulation_statistics=simulation_statistics,
                
                # Risk metrics
                var_95=risk_metrics['var_95'],
                var_99=risk_metrics['var_99'],
                cvar_95=risk_metrics['cvar_95'],
                cvar_99=risk_metrics['cvar_99'],
                expected_return=risk_metrics['expected_return'],
                expected_volatility=risk_metrics['expected_volatility'],
                
                # Percentiles
                percentiles=percentiles,
                confidence_intervals=confidence_intervals,
                
                # Regime analysis
                regime_simulations=regime_analysis['regime_simulations'],
                regime_statistics=regime_analysis['regime_statistics'],
                
                # Factor analysis
                factor_simulations=factor_analysis['factor_simulations'],
                factor_statistics=factor_analysis['factor_statistics'],
                
                # Time series
                simulation_returns=time_series['simulation_returns'],
                simulation_volatility=time_series['simulation_volatility'],
                simulation_drawdown=time_series['simulation_drawdown'],
                
                # Metadata
                simulation_period=(returns_series.index[0], returns_series.index[-1]),
                execution_time=(datetime.now() - start_time).total_seconds(),
                config=self.config
            )
            
            # Save results if configured
            if self.config.save_simulation_results:
                self._save_results(result)
            
            self.results = result
            self.logger.info(f"✅ Monte Carlo simulation completed in {result.execution_time:.2f}s")
            self.logger.info(f"📊 Expected return: {result.expected_return:.2%}")
            self.logger.info(f"📊 Expected volatility: {result.expected_volatility:.2%}")
            self.logger.info(f"📊 VaR 95%: {result.var_95:.2%}")
            self.logger.info(f"📊 CVaR 95%: {result.cvar_95:.2%}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Monte Carlo simulation failed: {e}")
            raise
    
    def _validate_data(self, 
                      returns_series: pd.Series,
                      regime_data: Optional[Dict[str, Any]],
                      factor_data: Optional[Dict[str, pd.Series]],
                      benchmark_returns: Optional[pd.Series]):
        """Validate data for Monte Carlo simulation."""
        if len(returns_series) < 30:
            raise ValueError(f"Insufficient data: {len(returns_series)} < 30")
        
        if regime_data and len(regime_data) != len(returns_series):
            self.logger.warning("⚠️ Regime data length mismatch with returns series")
        
        if factor_data:
            for factor_name, factor_series in factor_data.items():
                if len(factor_series) != len(returns_series):
                    self.logger.warning(f"⚠️ Factor {factor_name} length mismatch with returns series")
        
        if benchmark_returns is not None and len(benchmark_returns) != len(returns_series):
            self.logger.warning("⚠️ Benchmark returns length mismatch with portfolio returns")
        
        self.logger.info(f"✅ Data validation passed: {len(returns_series)} observations")
    
    def _prepare_simulation_data(self, 
                                returns_series: pd.Series,
                                regime_data: Optional[Dict[str, Any]],
                                factor_data: Optional[Dict[str, pd.Series]],
                                benchmark_returns: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for Monte Carlo simulation."""
        simulation_data = {
            'returns_series': returns_series,
            'regime_data': regime_data,
            'factor_data': factor_data,
            'benchmark_returns': benchmark_returns,
            'mean_return': returns_series.mean(),
            'std_return': returns_series.std(),
            'skewness': returns_series.skew(),
            'kurtosis': returns_series.kurtosis(),
            'min_return': returns_series.min(),
            'max_return': returns_series.max()
        }
        
        return simulation_data
    
    def _run_historical_simulation(self, simulation_data: Dict[str, Any]) -> np.ndarray:
        """Run historical Monte Carlo simulation."""
        returns_series = simulation_data['returns_series']
        n_simulations = self.config.n_simulations
        horizon = self.config.simulation_horizon
        
        # Bootstrap from historical data
        simulated_returns = np.zeros((n_simulations, horizon))
        
        for i in range(n_simulations):
            # Randomly sample from historical returns
            bootstrap_indices = np.random.choice(len(returns_series), size=horizon, replace=True)
            simulated_returns[i] = returns_series.iloc[bootstrap_indices].values
        
        return simulated_returns
    
    def _run_parametric_simulation(self, simulation_data: Dict[str, Any]) -> np.ndarray:
        """Run parametric Monte Carlo simulation."""
        mean_return = simulation_data['mean_return']
        std_return = simulation_data['std_return']
        skewness = simulation_data['skewness']
        kurtosis = simulation_data['kurtosis']
        
        n_simulations = self.config.n_simulations
        horizon = self.config.simulation_horizon
        
        # Generate normal random variables
        simulated_returns = np.random.normal(mean_return, std_return, (n_simulations, horizon))
        
        # Adjust for skewness and kurtosis (simplified)
        if abs(skewness) > 0.1:
            # Apply skewness transformation
            simulated_returns = np.sign(simulated_returns) * np.power(np.abs(simulated_returns), 1 + skewness/3)
        
        if abs(kurtosis) > 0.1:
            # Apply kurtosis transformation
            simulated_returns = simulated_returns * (1 + kurtosis/4)
        
        return simulated_returns
    
    def _run_bootstrap_simulation(self, simulation_data: Dict[str, Any]) -> np.ndarray:
        """Run bootstrap Monte Carlo simulation."""
        returns_series = simulation_data['returns_series']
        n_simulations = self.config.n_simulations
        horizon = self.config.simulation_horizon
        
        # Block bootstrap to preserve autocorrelation
        block_size = min(20, len(returns_series) // 4)
        n_blocks = horizon // block_size + 1
        
        simulated_returns = np.zeros((n_simulations, horizon))
        
        for i in range(n_simulations):
            simulation = []
            for _ in range(n_blocks):
                # Randomly select starting point
                start_idx = np.random.randint(0, len(returns_series) - block_size + 1)
                block = returns_series.iloc[start_idx:start_idx + block_size].values
                simulation.extend(block)
            
            # Truncate to horizon
            simulated_returns[i] = simulation[:horizon]
        
        return simulated_returns
    
    def _run_regime_based_simulation(self, simulation_data: Dict[str, Any]) -> np.ndarray:
        """Run regime-based Monte Carlo simulation."""
        returns_series = simulation_data['returns_series']
        regime_data = simulation_data['regime_data']
        
        if not regime_data:
            # Fall back to parametric simulation
            return self._run_parametric_simulation(simulation_data)
        
        regime_labels = regime_data.get('regime_labels', [])
        qualified_regimes = regime_data.get('qualified_regimes', {})
        
        n_simulations = self.config.n_simulations
        horizon = self.config.simulation_horizon
        
        simulated_returns = np.zeros((n_simulations, horizon))
        
        for i in range(n_simulations):
            simulation = []
            current_regime = np.random.choice(list(qualified_regimes.keys()))
            
            for t in range(horizon):
                # Check for regime transition
                if np.random.random() < self.config.regime_transition_probability:
                    current_regime = np.random.choice(list(qualified_regimes.keys()))
                
                # Generate return based on current regime
                regime_info = qualified_regimes[current_regime]
                regime_mean = regime_info.get('mean_return', 0.0)
                regime_std = regime_info.get('price_volatility', 0.1)
                
                return_value = np.random.normal(regime_mean, regime_std)
                simulation.append(return_value)
            
            simulated_returns[i] = simulation
        
        return simulated_returns
    
    def _run_factor_based_simulation(self, simulation_data: Dict[str, Any]) -> np.ndarray:
        """Run factor-based Monte Carlo simulation."""
        returns_series = simulation_data['returns_series']
        factor_data = simulation_data['factor_data']
        
        if not factor_data:
            # Fall back to parametric simulation
            return self._run_parametric_simulation(simulation_data)
        
        n_simulations = self.config.n_simulations
        horizon = self.config.simulation_horizon
        
        # Calculate factor loadings
        factor_loadings = {}
        for factor_name, factor_series in factor_data.items():
            correlation = returns_series.corr(factor_series)
            factor_loadings[factor_name] = correlation
        
        simulated_returns = np.zeros((n_simulations, horizon))
        
        for i in range(n_simulations):
            simulation = []
            
            for t in range(horizon):
                # Generate factor returns
                factor_return = 0.0
                for factor_name, factor_series in factor_data.items():
                    factor_mean = factor_series.mean()
                    factor_std = factor_series.std()
                    factor_sim = np.random.normal(factor_mean, factor_std)
                    factor_return += factor_loadings[factor_name] * factor_sim
                
                simulation.append(factor_return)
            
            simulated_returns[i] = simulation
        
        return simulated_returns
    
    def _calculate_simulation_statistics(self, simulated_returns: np.ndarray) -> Dict[str, float]:
        """Calculate simulation statistics."""
        return {
            'mean_return': np.mean(simulated_returns),
            'std_return': np.std(simulated_returns),
            'min_return': np.min(simulated_returns),
            'max_return': np.max(simulated_returns),
            'skewness': np.mean([np.mean(sim) for sim in simulated_returns]),
            'kurtosis': np.mean([np.std(sim) for sim in simulated_returns]),
            'sharpe_ratio': np.mean(simulated_returns) / np.std(simulated_returns) if np.std(simulated_returns) > 0 else 0.0
        }
    
    def _calculate_risk_metrics(self, simulated_returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics from simulations."""
        # VaR and CVaR
        var_95 = np.percentile(simulated_returns, 5)
        var_99 = np.percentile(simulated_returns, 1)
        
        cvar_95 = np.mean(simulated_returns[simulated_returns <= var_95])
        cvar_99 = np.mean(simulated_returns[simulated_returns <= var_99])
        
        # Expected return and volatility
        expected_return = np.mean(simulated_returns)
        expected_volatility = np.std(simulated_returns)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'expected_return': expected_return,
            'expected_volatility': expected_volatility
        }
    
    def _calculate_percentiles(self, simulated_returns: np.ndarray) -> Dict[str, float]:
        """Calculate percentiles from simulations."""
        percentiles = {}
        
        for confidence_level in self.config.confidence_levels:
            percentile = (1 - confidence_level) * 100
            percentiles[f'percentile_{int(percentile)}'] = np.percentile(simulated_returns, percentile)
        
        return percentiles
    
    def _calculate_confidence_intervals(self, simulated_returns: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals from simulations."""
        confidence_intervals = {}
        
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            lower = np.percentile(simulated_returns, (alpha/2) * 100)
            upper = np.percentile(simulated_returns, (1 - alpha/2) * 100)
            confidence_intervals[f'ci_{int(confidence_level*100)}'] = (lower, upper)
        
        return confidence_intervals
    
    def _analyze_regime_simulations(self, simulated_returns: np.ndarray, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-specific simulations."""
        regime_data = simulation_data['regime_data']
        
        if not regime_data:
            return {'regime_simulations': {}, 'regime_statistics': {}}
        
        regime_simulations = {}
        regime_statistics = {}
        
        # This would be more sophisticated in practice
        # For now, return basic structure
        for regime_name in regime_data.get('qualified_regimes', {}).keys():
            regime_simulations[regime_name] = simulated_returns
            regime_statistics[regime_name] = {
                'mean_return': np.mean(simulated_returns),
                'std_return': np.std(simulated_returns)
            }
        
        return {
            'regime_simulations': regime_simulations,
            'regime_statistics': regime_statistics
        }
    
    def _analyze_factor_simulations(self, simulated_returns: np.ndarray, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factor-specific simulations."""
        factor_data = simulation_data['factor_data']
        
        if not factor_data:
            return {'factor_simulations': {}, 'factor_statistics': {}}
        
        factor_simulations = {}
        factor_statistics = {}
        
        # This would be more sophisticated in practice
        # For now, return basic structure
        for factor_name in factor_data.keys():
            factor_simulations[factor_name] = simulated_returns
            factor_statistics[factor_name] = {
                'mean_return': np.mean(simulated_returns),
                'std_return': np.std(simulated_returns)
            }
        
        return {
            'factor_simulations': factor_simulations,
            'factor_statistics': factor_statistics
        }
    
    def _create_time_series(self, simulated_returns: np.ndarray) -> Dict[str, pd.Series]:
        """Create time series from simulations."""
        # Calculate average simulation
        avg_simulation = np.mean(simulated_returns, axis=0)
        
        # Create time series
        simulation_returns = pd.Series(avg_simulation)
        simulation_volatility = pd.Series(np.std(simulated_returns, axis=0))
        
        # Calculate drawdown
        cumulative_returns = (1 + simulation_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        simulation_drawdown = (cumulative_returns - running_max) / running_max
        
        return {
            'simulation_returns': simulation_returns,
            'simulation_volatility': simulation_volatility,
            'simulation_drawdown': simulation_drawdown
        }
    
    def _calculate_equity_curves(self, simulated_returns: np.ndarray) -> np.ndarray:
        """Calculate equity curves from simulations."""
        equity_curves = np.zeros_like(simulated_returns)
        
        for i in range(simulated_returns.shape[0]):
            equity_curves[i] = np.cumprod(1 + simulated_returns[i])
        
        return equity_curves
    
    def _save_results(self, result: MonteCarloResult):
        """Save Monte Carlo simulation results."""
        try:
            results_dir = Path(self.config.simulation_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = results_dir / f"monte_carlo_summary_{timestamp}.json"
            
            summary = {
                'n_simulations': self.config.n_simulations,
                'simulation_horizon': self.config.simulation_horizon,
                'method': self.config.method.value,
                'expected_return': result.expected_return,
                'expected_volatility': result.expected_volatility,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'cvar_95': result.cvar_95,
                'cvar_99': result.cvar_99,
                'simulation_statistics': result.simulation_statistics,
                'execution_time': result.execution_time
            }
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results
            if self.config.detailed_simulation:
                details_file = results_dir / f"monte_carlo_details_{timestamp}.json"
                
                details = {
                    'percentiles': result.percentiles,
                    'confidence_intervals': result.confidence_intervals,
                    'regime_statistics': result.regime_statistics,
                    'factor_statistics': result.factor_statistics
                }
                
                with open(details_file, 'w') as f:
                    json.dump(details, f, indent=2, default=str)
            
            self.logger.info(f"📁 Monte Carlo simulation results saved to {results_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_results(self) -> Optional[MonteCarloResult]:
        """Get Monte Carlo simulation results."""
        return self.results
    
    def export_results(self, filepath: str):
        """Export results to file."""
        if self.results is None:
            self.logger.warning("⚠️ No results to export")
            return
        
        try:
            # Export simulation returns
            returns_file = filepath.replace('.csv', '_simulation_returns.csv')
            self.results.simulation_returns.to_csv(returns_file)
            
            # Export simulation volatility
            volatility_file = filepath.replace('.csv', '_simulation_volatility.csv')
            self.results.simulation_volatility.to_csv(volatility_file)
            
            # Export simulation drawdown
            drawdown_file = filepath.replace('.csv', '_simulation_drawdown.csv')
            self.results.simulation_drawdown.to_csv(drawdown_file)
            
            self.logger.info(f"📁 Monte Carlo simulation results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")