"""
Scenario Testing for TAS

Comprehensive scenario testing for tree architecture search including
stress scenarios, Monte Carlo simulation, and sensitivity analysis.
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


class ScenarioType(Enum):
    """Scenario types."""
    STRESS = "stress"
    MONTE_CARLO = "monte_carlo"
    SENSITIVITY = "sensitivity"
    REGIME_CHANGE = "regime_change"
    MARKET_CRASH = "market_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class ScenarioConfig:
    """Configuration for scenario testing."""
    
    # Scenario types
    scenario_types: List[ScenarioType] = field(default_factory=lambda: [
        ScenarioType.STRESS,
        ScenarioType.MONTE_CARLO,
        ScenarioType.SENSITIVITY
    ])
    
    # Stress testing
    stress_scenarios: List[str] = field(default_factory=lambda: [
        'market_crash', 'volatility_spike', 'liquidity_crisis', 'regime_change'
    ])
    stress_magnitudes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    stress_probabilities: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3])
    
    # Monte Carlo parameters
    n_simulations: int = 10000
    simulation_horizon: int = 252  # Trading days
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Sensitivity analysis
    sensitivity_factors: List[str] = field(default_factory=lambda: [
        'volatility', 'correlation', 'regime_stability', 'liquidity'
    ])
    sensitivity_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'volatility': (0.5, 2.0),
        'correlation': (0.0, 1.0),
        'regime_stability': (0.0, 1.0),
        'liquidity': (0.0, 1.0)
    })
    
    # Regime change scenarios
    regime_change_probability: float = 0.1
    regime_change_magnitude: float = 0.2
    
    # Market crash scenarios
    crash_probability: float = 0.05
    crash_magnitude: float = 0.3
    
    # Volatility spike scenarios
    volatility_spike_probability: float = 0.1
    volatility_spike_magnitude: float = 2.0
    
    # Liquidity crisis scenarios
    liquidity_crisis_probability: float = 0.05
    liquidity_crisis_magnitude: float = 0.5
    
    # Output parameters
    save_scenario_results: bool = True
    scenario_directory: str = "scenario_results"
    detailed_scenario_analysis: bool = True


@dataclass
class ScenarioResult:
    """Result of scenario testing."""
    
    # Stress testing results
    stress_test_results: Dict[str, Dict[str, float]]
    stress_scenarios: Dict[str, float]
    stress_probabilities: Dict[str, float]
    
    # Monte Carlo results
    monte_carlo_results: Dict[str, float]
    monte_carlo_percentiles: Dict[str, float]
    monte_carlo_confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Sensitivity analysis
    sensitivity_results: Dict[str, Dict[str, float]]
    sensitivity_impact: Dict[str, float]
    sensitivity_rankings: List[Tuple[str, float]]
    
    # Regime change scenarios
    regime_change_results: Dict[str, float]
    regime_change_impact: float
    regime_change_probability: float
    
    # Market crash scenarios
    market_crash_results: Dict[str, float]
    market_crash_impact: float
    market_crash_probability: float
    
    # Volatility spike scenarios
    volatility_spike_results: Dict[str, float]
    volatility_spike_impact: float
    volatility_spike_probability: float
    
    # Liquidity crisis scenarios
    liquidity_crisis_results: Dict[str, float]
    liquidity_crisis_impact: float
    liquidity_crisis_probability: float
    
    # Overall scenario metrics
    worst_case_scenario: str
    best_case_scenario: str
    expected_scenario: str
    scenario_risk_score: float
    
    # Time series
    scenario_returns: pd.Series
    scenario_volatility: pd.Series
    scenario_drawdown: pd.Series
    
    # Metadata
    analysis_period: Tuple[datetime, datetime]
    execution_time: float
    config: ScenarioConfig


class ScenarioTester:
    """
    Comprehensive scenario tester for TAS.
    
    Provides stress testing, Monte Carlo simulation, sensitivity analysis,
    and scenario-based risk assessment for tree architecture search.
    """
    
    def __init__(self, config: ScenarioConfig):
        """Initialize scenario tester.
        
        Args:
            config: Scenario testing configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Scenario testing state
        self.results = None
        self.scenario_data = None
        
        self.logger.info("✅ Scenario Tester initialized")
        self.logger.info(f"📊 Scenario types: {[scenario.value for scenario in config.scenario_types]}")
        self.logger.info(f"📊 Monte Carlo simulations: {config.n_simulations}")
    
    def run_scenario_testing(self, 
                           returns_series: pd.Series,
                           regime_data: Optional[Dict[str, Any]] = None,
                           factor_data: Optional[Dict[str, pd.Series]] = None,
                           benchmark_returns: Optional[pd.Series] = None) -> ScenarioResult:
        """
        Run comprehensive scenario testing.
        
        Args:
            returns_series: Portfolio returns series
            regime_data: Optional regime information
            factor_data: Optional factor data
            benchmark_returns: Optional benchmark returns
            
        Returns:
            Scenario testing result
        """
        self.logger.info("🚀 Starting comprehensive scenario testing")
        start_time = datetime.now()
        
        try:
            # Validate data
            self._validate_data(returns_series, regime_data, factor_data, benchmark_returns)
            
            # Prepare scenario data
            scenario_data = self._prepare_scenario_data(returns_series, regime_data, factor_data, benchmark_returns)
            
            # Run scenario analysis
            scenario_results = {}
            
            if ScenarioType.STRESS in self.config.scenario_types:
                self.logger.info("⚠️ Running stress testing...")
                scenario_results['stress'] = self._run_stress_testing(scenario_data)
            
            if ScenarioType.MONTE_CARLO in self.config.scenario_types:
                self.logger.info("🎲 Running Monte Carlo simulation...")
                scenario_results['monte_carlo'] = self._run_monte_carlo_simulation(scenario_data)
            
            if ScenarioType.SENSITIVITY in self.config.scenario_types:
                self.logger.info("🔍 Running sensitivity analysis...")
                scenario_results['sensitivity'] = self._run_sensitivity_analysis(scenario_data)
            
            if ScenarioType.REGIME_CHANGE in self.config.scenario_types:
                self.logger.info("🎯 Running regime change scenarios...")
                scenario_results['regime_change'] = self._run_regime_change_scenarios(scenario_data)
            
            if ScenarioType.MARKET_CRASH in self.config.scenario_types:
                self.logger.info("💥 Running market crash scenarios...")
                scenario_results['market_crash'] = self._run_market_crash_scenarios(scenario_data)
            
            if ScenarioType.VOLATILITY_SPIKE in self.config.scenario_types:
                self.logger.info("📈 Running volatility spike scenarios...")
                scenario_results['volatility_spike'] = self._run_volatility_spike_scenarios(scenario_data)
            
            if ScenarioType.LIQUIDITY_CRISIS in self.config.scenario_types:
                self.logger.info("💧 Running liquidity crisis scenarios...")
                scenario_results['liquidity_crisis'] = self._run_liquidity_crisis_scenarios(scenario_data)
            
            # Calculate overall scenario metrics
            overall_metrics = self._calculate_overall_scenario_metrics(scenario_results)
            
            # Create comprehensive result
            result = ScenarioResult(
                # Stress testing results
                stress_test_results=scenario_results.get('stress', {}).get('stress_test_results', {}),
                stress_scenarios=scenario_results.get('stress', {}).get('stress_scenarios', {}),
                stress_probabilities=scenario_results.get('stress', {}).get('stress_probabilities', {}),
                
                # Monte Carlo results
                monte_carlo_results=scenario_results.get('monte_carlo', {}).get('monte_carlo_results', {}),
                monte_carlo_percentiles=scenario_results.get('monte_carlo', {}).get('monte_carlo_percentiles', {}),
                monte_carlo_confidence_intervals=scenario_results.get('monte_carlo', {}).get('monte_carlo_confidence_intervals', {}),
                
                # Sensitivity analysis
                sensitivity_results=scenario_results.get('sensitivity', {}).get('sensitivity_results', {}),
                sensitivity_impact=scenario_results.get('sensitivity', {}).get('sensitivity_impact', {}),
                sensitivity_rankings=scenario_results.get('sensitivity', {}).get('sensitivity_rankings', []),
                
                # Regime change scenarios
                regime_change_results=scenario_results.get('regime_change', {}).get('regime_change_results', {}),
                regime_change_impact=scenario_results.get('regime_change', {}).get('regime_change_impact', 0.0),
                regime_change_probability=self.config.regime_change_probability,
                
                # Market crash scenarios
                market_crash_results=scenario_results.get('market_crash', {}).get('market_crash_results', {}),
                market_crash_impact=scenario_results.get('market_crash', {}).get('market_crash_impact', 0.0),
                market_crash_probability=self.config.crash_probability,
                
                # Volatility spike scenarios
                volatility_spike_results=scenario_results.get('volatility_spike', {}).get('volatility_spike_results', {}),
                volatility_spike_impact=scenario_results.get('volatility_spike', {}).get('volatility_spike_impact', 0.0),
                volatility_spike_probability=self.config.volatility_spike_probability,
                
                # Liquidity crisis scenarios
                liquidity_crisis_results=scenario_results.get('liquidity_crisis', {}).get('liquidity_crisis_results', {}),
                liquidity_crisis_impact=scenario_results.get('liquidity_crisis', {}).get('liquidity_crisis_impact', 0.0),
                liquidity_crisis_probability=self.config.liquidity_crisis_probability,
                
                # Overall scenario metrics
                worst_case_scenario=overall_metrics['worst_case_scenario'],
                best_case_scenario=overall_metrics['best_case_scenario'],
                expected_scenario=overall_metrics['expected_scenario'],
                scenario_risk_score=overall_metrics['scenario_risk_score'],
                
                # Time series
                scenario_returns=scenario_data['scenario_returns'],
                scenario_volatility=scenario_data['scenario_volatility'],
                scenario_drawdown=scenario_data['scenario_drawdown'],
                
                # Metadata
                analysis_period=(returns_series.index[0], returns_series.index[-1]),
                execution_time=(datetime.now() - start_time).total_seconds(),
                config=self.config
            )
            
            # Save results if configured
            if self.config.save_scenario_results:
                self._save_results(result)
            
            self.results = result
            self.logger.info(f"✅ Scenario testing completed in {result.execution_time:.2f}s")
            self.logger.info(f"📊 Scenario risk score: {result.scenario_risk_score:.3f}")
            self.logger.info(f"⚠️ Worst case scenario: {result.worst_case_scenario}")
            self.logger.info(f"📈 Best case scenario: {result.best_case_scenario}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Scenario testing failed: {e}")
            raise
    
    def _validate_data(self, 
                      returns_series: pd.Series,
                      regime_data: Optional[Dict[str, Any]],
                      factor_data: Optional[Dict[str, pd.Series]],
                      benchmark_returns: Optional[pd.Series]):
        """Validate data for scenario testing."""
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
    
    def _prepare_scenario_data(self, 
                              returns_series: pd.Series,
                              regime_data: Optional[Dict[str, Any]],
                              factor_data: Optional[Dict[str, pd.Series]],
                              benchmark_returns: Optional[pd.Series]) -> Dict[str, Any]:
        """Prepare data for scenario testing."""
        # Calculate scenario metrics
        scenario_returns = returns_series.copy()
        scenario_volatility = returns_series.rolling(window=20).std()
        scenario_drawdown = self._calculate_scenario_drawdown(returns_series)
        
        scenario_data = {
            'returns_series': returns_series,
            'regime_data': regime_data,
            'factor_data': factor_data,
            'benchmark_returns': benchmark_returns,
            'scenario_returns': scenario_returns,
            'scenario_volatility': scenario_volatility,
            'scenario_drawdown': scenario_drawdown,
            'mean_return': returns_series.mean(),
            'std_return': returns_series.std(),
            'skewness': returns_series.skew(),
            'kurtosis': returns_series.kurtosis()
        }
        
        return scenario_data
    
    def _calculate_scenario_drawdown(self, returns_series: pd.Series) -> pd.Series:
        """Calculate scenario drawdown series."""
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown_series = (cumulative_returns - running_max) / running_max
        
        return drawdown_series
    
    def _run_stress_testing(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress testing scenarios."""
        returns_series = scenario_data['returns_series']
        mean_return = scenario_data['mean_return']
        std_return = scenario_data['std_return']
        
        stress_test_results = {}
        stress_scenarios = {}
        stress_probabilities = {}
        
        for scenario in self.config.stress_scenarios:
            scenario_results = {}
            
            for magnitude, probability in zip(self.config.stress_magnitudes, self.config.stress_probabilities):
                if scenario == 'market_crash':
                    stress_return = -magnitude
                elif scenario == 'volatility_spike':
                    stress_return = mean_return - magnitude * std_return
                elif scenario == 'liquidity_crisis':
                    stress_return = mean_return - magnitude * std_return
                elif scenario == 'regime_change':
                    stress_return = mean_return - magnitude * std_return
                else:
                    stress_return = -magnitude
                
                scenario_results[f'magnitude_{magnitude}'] = stress_return
            
            stress_test_results[scenario] = scenario_results
            stress_scenarios[scenario] = min(scenario_results.values())
            stress_probabilities[scenario] = probability
        
        return {
            'stress_test_results': stress_test_results,
            'stress_scenarios': stress_scenarios,
            'stress_probabilities': stress_probabilities
        }
    
    def _run_monte_carlo_simulation(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        returns_series = scenario_data['returns_series']
        mean_return = scenario_data['mean_return']
        std_return = scenario_data['std_return']
        skewness = scenario_data['skewness']
        kurtosis = scenario_data['kurtosis']
        
        # Generate Monte Carlo simulations
        n_simulations = self.config.n_simulations
        horizon = self.config.simulation_horizon
        
        # Use historical parameters for simulation
        simulated_returns = np.random.normal(mean_return, std_return, (n_simulations, horizon))
        
        # Calculate simulation metrics
        monte_carlo_results = {
            'mean_return': np.mean(simulated_returns),
            'std_return': np.std(simulated_returns),
            'min_return': np.min(simulated_returns),
            'max_return': np.max(simulated_returns),
            'skewness': np.mean([np.mean(sim) for sim in simulated_returns]),
            'kurtosis': np.mean([np.std(sim) for sim in simulated_returns])
        }
        
        # Calculate percentiles
        monte_carlo_percentiles = {}
        for confidence_level in self.config.confidence_levels:
            percentile = (1 - confidence_level) * 100
            monte_carlo_percentiles[f'percentile_{int(percentile)}'] = np.percentile(simulated_returns, percentile)
        
        # Calculate confidence intervals
        monte_carlo_confidence_intervals = {}
        for confidence_level in self.config.confidence_levels:
            alpha = 1 - confidence_level
            lower = np.percentile(simulated_returns, (alpha/2) * 100)
            upper = np.percentile(simulated_returns, (1 - alpha/2) * 100)
            monte_carlo_confidence_intervals[f'ci_{int(confidence_level*100)}'] = (lower, upper)
        
        return {
            'monte_carlo_results': monte_carlo_results,
            'monte_carlo_percentiles': monte_carlo_percentiles,
            'monte_carlo_confidence_intervals': monte_carlo_confidence_intervals
        }
    
    def _run_sensitivity_analysis(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run sensitivity analysis with comprehensive error handling."""
        returns_series = scenario_data['returns_series']
        factor_data = scenario_data['factor_data']
        
        sensitivity_results = {}
        sensitivity_impact = {}
        sensitivity_rankings = []
        
        try:
            # Validate input data
            if len(returns_series) < 10:
                self.logger.warning("⚠️ Insufficient data for sensitivity analysis")
                return {
                    'sensitivity_results': {},
                    'sensitivity_impact': {},
                    'sensitivity_rankings': []
                }
            
            for factor in self.config.sensitivity_factors:
                try:
                    factor_results = {}
                    
                    if factor == 'volatility':
                        base_volatility = returns_series.std()
                        if base_volatility <= 0:
                            self.logger.warning(f"⚠️ Invalid base volatility for {factor}: {base_volatility}")
                            continue
                            
                        for multiplier in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                            try:
                                adjusted_returns = returns_series * (multiplier / base_volatility)
                                if len(adjusted_returns) > 0 and np.isfinite(adjusted_returns.mean()):
                                    factor_results[f'multiplier_{multiplier}'] = adjusted_returns.mean()
                                else:
                                    self.logger.warning(f"⚠️ Invalid adjusted returns for multiplier {multiplier}")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Error calculating volatility sensitivity for multiplier {multiplier}: {e}")
                    
                    elif factor == 'correlation' and factor_data:
                        for factor_name, factor_series in factor_data.items():
                            try:
                                if len(factor_series) == len(returns_series):
                                    correlation = returns_series.corr(factor_series)
                                    if np.isfinite(correlation):
                                        factor_results[f'correlation_{factor_name}'] = correlation
                                    else:
                                        self.logger.warning(f"⚠️ Invalid correlation for {factor_name}")
                                else:
                                    self.logger.warning(f"⚠️ Length mismatch for factor {factor_name}")
                            except Exception as e:
                                self.logger.warning(f"⚠️ Error calculating correlation for {factor_name}: {e}")
                    
                    elif factor == 'regime_stability':
                        # Enhanced regime stability sensitivity
                        base_return = returns_series.mean()
                        if not np.isfinite(base_return):
                            self.logger.warning(f"⚠️ Invalid base return for regime stability")
                            continue
                            
                        for stability in [0.0, 0.25, 0.5, 0.75, 1.0]:
                            try:
                                # Apply stability as a confidence factor
                                adjusted_return = base_return * stability
                                if np.isfinite(adjusted_return):
                                    factor_results[f'stability_{stability}'] = adjusted_return
                            except Exception as e:
                                self.logger.warning(f"⚠️ Error calculating regime stability for {stability}: {e}")
                    
                    elif factor == 'liquidity':
                        # Enhanced liquidity sensitivity
                        base_return = returns_series.mean()
                        if not np.isfinite(base_return):
                            self.logger.warning(f"⚠️ Invalid base return for liquidity")
                            continue
                            
                        for liquidity in [0.0, 0.25, 0.5, 0.75, 1.0]:
                            try:
                                # Apply liquidity as a market depth factor
                                adjusted_return = base_return * liquidity
                                if np.isfinite(adjusted_return):
                                    factor_results[f'liquidity_{liquidity}'] = adjusted_return
                            except Exception as e:
                                self.logger.warning(f"⚠️ Error calculating liquidity sensitivity for {liquidity}: {e}")
                    
                    sensitivity_results[factor] = factor_results
                    
                    # Calculate impact with validation
                    if factor_results:
                        try:
                            values = list(factor_results.values())
                            if values and all(np.isfinite(v) for v in values):
                                impact = max(values) - min(values)
                                if np.isfinite(impact):
                                    sensitivity_impact[factor] = impact
                                    sensitivity_rankings.append((factor, impact))
                                else:
                                    self.logger.warning(f"⚠️ Invalid impact calculated for {factor}")
                            else:
                                self.logger.warning(f"⚠️ Invalid values in factor results for {factor}")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error calculating impact for {factor}: {e}")
                    else:
                        self.logger.warning(f"⚠️ No valid results for factor {factor}")
                
                except Exception as e:
                    self.logger.error(f"❌ Error processing sensitivity factor {factor}: {e}")
                    continue
            
            # Sort rankings by impact with validation
            try:
                sensitivity_rankings.sort(key=lambda x: x[1], reverse=True)
            except Exception as e:
                self.logger.warning(f"⚠️ Error sorting sensitivity rankings: {e}")
            
            self.logger.info(f"📊 Sensitivity analysis completed: {len(sensitivity_results)} factors analyzed")
            
        except Exception as e:
            self.logger.error(f"❌ Sensitivity analysis failed: {e}")
            return {
                'sensitivity_results': {},
                'sensitivity_impact': {},
                'sensitivity_rankings': []
            }
        
        return {
            'sensitivity_results': sensitivity_results,
            'sensitivity_impact': sensitivity_impact,
            'sensitivity_rankings': sensitivity_rankings
        }
    
    def _run_regime_change_scenarios(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run regime change scenarios."""
        returns_series = scenario_data['returns_series']
        regime_data = scenario_data['regime_data']
        
        regime_change_results = {}
        regime_change_impact = 0.0
        
        if regime_data:
            regime_labels = regime_data.get('regime_labels', [])
            qualified_regimes = regime_data.get('qualified_regimes', {})
            
            # Calculate regime change impact
            for regime_name, regime_info in qualified_regimes.items():
                regime_id = regime_info.get('regime_id')
                regime_mask = np.array(regime_labels) == regime_id
                
                if np.any(regime_mask):
                    regime_returns = returns_series[regime_mask]
                    regime_change_results[regime_name] = regime_returns.mean()
            
            # Calculate overall impact
            if regime_change_results:
                regime_change_impact = min(regime_change_results.values()) * self.config.regime_change_magnitude
        
        return {
            'regime_change_results': regime_change_results,
            'regime_change_impact': regime_change_impact
        }
    
    def _run_market_crash_scenarios(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run market crash scenarios."""
        returns_series = scenario_data['returns_series']
        mean_return = scenario_data['mean_return']
        std_return = scenario_data['std_return']
        
        market_crash_results = {
            'crash_return': mean_return - self.config.crash_magnitude * std_return,
            'crash_probability': self.config.crash_probability,
            'crash_impact': self.config.crash_magnitude * std_return
        }
        
        market_crash_impact = market_crash_results['crash_impact']
        
        return {
            'market_crash_results': market_crash_results,
            'market_crash_impact': market_crash_impact
        }
    
    def _run_volatility_spike_scenarios(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run volatility spike scenarios."""
        returns_series = scenario_data['returns_series']
        mean_return = scenario_data['mean_return']
        std_return = scenario_data['std_return']
        
        volatility_spike_results = {
            'spike_volatility': std_return * self.config.volatility_spike_magnitude,
            'spike_probability': self.config.volatility_spike_probability,
            'spike_impact': std_return * (self.config.volatility_spike_magnitude - 1)
        }
        
        volatility_spike_impact = volatility_spike_results['spike_impact']
        
        return {
            'volatility_spike_results': volatility_spike_results,
            'volatility_spike_impact': volatility_spike_impact
        }
    
    def _run_liquidity_crisis_scenarios(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run liquidity crisis scenarios."""
        returns_series = scenario_data['returns_series']
        mean_return = scenario_data['mean_return']
        std_return = scenario_data['std_return']
        
        liquidity_crisis_results = {
            'crisis_return': mean_return - self.config.liquidity_crisis_magnitude * std_return,
            'crisis_probability': self.config.liquidity_crisis_probability,
            'crisis_impact': self.config.liquidity_crisis_magnitude * std_return
        }
        
        liquidity_crisis_impact = liquidity_crisis_results['crisis_impact']
        
        return {
            'liquidity_crisis_results': liquidity_crisis_results,
            'liquidity_crisis_impact': liquidity_crisis_impact
        }
    
    def _calculate_overall_scenario_metrics(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall scenario metrics."""
        # Find worst case scenario
        worst_case_scenario = "market_crash"
        worst_case_impact = float('inf')
        
        for scenario_type, results in scenario_results.items():
            if 'impact' in results:
                if results['impact'] < worst_case_impact:
                    worst_case_impact = results['impact']
                    worst_case_scenario = scenario_type
        
        # Find best case scenario
        best_case_scenario = "monte_carlo"
        best_case_impact = float('-inf')
        
        for scenario_type, results in scenario_results.items():
            if 'impact' in results:
                if results['impact'] > best_case_impact:
                    best_case_impact = results['impact']
                    best_case_scenario = scenario_type
        
        # Calculate scenario risk score
        scenario_risk_score = abs(worst_case_impact) / abs(best_case_impact) if best_case_impact != 0 else 1.0
        
        return {
            'worst_case_scenario': worst_case_scenario,
            'best_case_scenario': best_case_scenario,
            'expected_scenario': 'monte_carlo',
            'scenario_risk_score': scenario_risk_score
        }
    
    def _save_results(self, result: ScenarioResult):
        """Save scenario testing results."""
        try:
            results_dir = Path(self.config.scenario_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = results_dir / f"scenario_testing_summary_{timestamp}.json"
            
            summary = {
                'worst_case_scenario': result.worst_case_scenario,
                'best_case_scenario': result.best_case_scenario,
                'scenario_risk_score': result.scenario_risk_score,
                'stress_scenarios': result.stress_scenarios,
                'monte_carlo_results': result.monte_carlo_results,
                'sensitivity_rankings': result.sensitivity_rankings,
                'execution_time': result.execution_time
            }
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results
            if self.config.detailed_scenario_analysis:
                details_file = results_dir / f"scenario_testing_details_{timestamp}.json"
                
                details = {
                    'stress_test_results': result.stress_test_results,
                    'monte_carlo_percentiles': result.monte_carlo_percentiles,
                    'monte_carlo_confidence_intervals': result.monte_carlo_confidence_intervals,
                    'sensitivity_results': result.sensitivity_results,
                    'regime_change_results': result.regime_change_results,
                    'market_crash_results': result.market_crash_results,
                    'volatility_spike_results': result.volatility_spike_results,
                    'liquidity_crisis_results': result.liquidity_crisis_results
                }
                
                with open(details_file, 'w') as f:
                    json.dump(details, f, indent=2, default=str)
            
            self.logger.info(f"📁 Scenario testing results saved to {results_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_results(self) -> Optional[ScenarioResult]:
        """Get scenario testing results."""
        return self.results
    
    def export_results(self, filepath: str):
        """Export results to file."""
        if self.results is None:
            self.logger.warning("⚠️ No results to export")
            return
        
        try:
            # Export scenario returns
            returns_file = filepath.replace('.csv', '_scenario_returns.csv')
            self.results.scenario_returns.to_csv(returns_file)
            
            # Export scenario volatility
            volatility_file = filepath.replace('.csv', '_scenario_volatility.csv')
            self.results.scenario_volatility.to_csv(volatility_file)
            
            # Export scenario drawdown
            drawdown_file = filepath.replace('.csv', '_scenario_drawdown.csv')
            self.results.scenario_drawdown.to_csv(drawdown_file)
            
            self.logger.info(f"📁 Scenario testing results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")