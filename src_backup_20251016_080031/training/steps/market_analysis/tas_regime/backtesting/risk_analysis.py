"""
Risk Analysis for TAS

Comprehensive risk analysis for tree architecture search including
VaR, CVaR, stress testing, scenario analysis, and risk attribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class RiskMetric(Enum):
    """Risk metrics."""
    VAR = "var"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    VOLATILITY = "volatility"
    SKEWNESS = "skewness"
    KURTOSIS = "kurtosis"


@dataclass
class RiskConfig:
    """Configuration for risk analysis."""
    
    # VaR parameters
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_methods: List[str] = field(default_factory=lambda: ['historical', 'parametric', 'monte_carlo'])
    
    # CVaR parameters
    cvar_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Stress testing
    enable_stress_testing: bool = True
    stress_scenarios: List[str] = field(default_factory=lambda: [
        'market_crash', 'volatility_spike', 'liquidity_crisis', 'regime_change'
    ])
    stress_magnitudes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    
    # Scenario analysis
    enable_scenario_analysis: bool = True
    scenario_probabilities: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.2, 0.1])
    scenario_returns: List[float] = field(default_factory=lambda: [-0.2, -0.1, 0.0, 0.1, 0.2])
    
    # Risk attribution
    enable_risk_attribution: bool = True
    risk_factors: List[str] = field(default_factory=lambda: [
        'market', 'regime', 'liquidity', 'concentration', 'leverage'
    ])
    
    # Advanced parameters
    enable_regime_risk: bool = True
    enable_liquidity_risk: bool = True
    enable_concentration_risk: bool = True
    enable_leverage_risk: bool = True
    
    # Output parameters
    save_risk_analysis: bool = True
    risk_directory: str = "risk_analysis_results"
    detailed_risk_analysis: bool = True


@dataclass
class RiskResult:
    """Result of risk analysis."""
    
    # VaR metrics
    var_95: float
    var_99: float
    var_historical: Dict[str, float]
    var_parametric: Dict[str, float]
    var_monte_carlo: Dict[str, float]
    
    # CVaR metrics
    cvar_95: float
    cvar_99: float
    cvar_historical: Dict[str, float]
    cvar_parametric: Dict[str, float]
    cvar_monte_carlo: Dict[str, float]
    
    # Drawdown metrics
    max_drawdown: float
    average_drawdown: float
    drawdown_duration: int
    recovery_time: int
    
    # Risk ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Beta and Alpha
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    
    # Higher moments
    volatility: float
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Stress testing
    stress_test_results: Dict[str, Dict[str, float]]
    stress_scenarios: Dict[str, float]
    
    # Scenario analysis
    scenario_analysis: Dict[str, float]
    expected_return: float
    expected_volatility: float
    
    # Risk attribution
    risk_attribution: Dict[str, float]
    risk_contribution: Dict[str, float]
    risk_impact: Dict[str, float]
    
    # Regime risk
    regime_risk: Dict[str, float]
    regime_stability: Dict[str, float]
    regime_transition_risk: float
    
    # Time series
    returns_series: pd.Series
    drawdown_series: pd.Series
    risk_series: pd.Series
    
    # Metadata
    analysis_period: Tuple[datetime, datetime]
    execution_time: float
    config: RiskConfig


class RiskAnalyzer:
    """
    Comprehensive risk analyzer for TAS.
    
    Provides VaR, CVaR, stress testing, scenario analysis,
    and risk attribution for tree architecture search.
    """
    
    def __init__(self, config: RiskConfig):
        """Initialize risk analyzer.
        
        Args:
            config: Risk analysis configuration
        """
        tprint_info("🚀 Initializing Risk Analyzer")
        tprint_debug(f"Configuration: {config}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Risk analysis state
        tprint_debug("📊 Initializing risk analysis state...")
        self.results = None
        self.risk_data = None
        
        tprint_success("✅ Risk Analyzer initialized")
        tprint_info(f"📊 VaR confidence levels: {config.var_confidence_levels}")
        tprint_info(f"📊 CVaR confidence levels: {config.cvar_confidence_levels}")
        self.logger.info("✅ Risk Analyzer initialized")
        self.logger.info(f"📊 VaR confidence levels: {config.var_confidence_levels}")
        self.logger.info(f"📊 CVaR confidence levels: {config.cvar_confidence_levels}")
    
    def run_analysis(self, 
                    returns_series: pd.Series,
                    benchmark_returns: Optional[pd.Series] = None,
                    regime_data: Optional[Dict[str, Any]] = None,
                    factor_data: Optional[Dict[str, pd.Series]] = None) -> RiskResult:
        """
        Run comprehensive risk analysis.
        
        Args:
            returns_series: Portfolio returns series
            benchmark_returns: Optional benchmark returns
            regime_data: Optional regime information
            factor_data: Optional factor data
            
        Returns:
            Risk analysis result
        """
        tprint_info("🚀 Starting comprehensive risk analysis")
        tprint_debug(f"Returns series shape: {returns_series.shape}")
        tprint_debug(f"Benchmark returns: {'Yes' if benchmark_returns is not None else 'No'}")
        tprint_debug(f"Regime data: {'Yes' if regime_data is not None else 'No'}")
        tprint_debug(f"Factor data: {'Yes' if factor_data is not None else 'No'}")
        
        self.logger.info("🚀 Starting comprehensive risk analysis")
        start_time = datetime.now()
        
        try:
            # Validate data
            tprint_debug("🔍 Validating data...")
            self._validate_data(returns_series, benchmark_returns, regime_data, factor_data)
            tprint_success("✅ Data validation passed")
            
            # Prepare risk data
            tprint_debug("📊 Preparing risk data...")
            risk_data = self._prepare_risk_data(returns_series, benchmark_returns, regime_data, factor_data)
            tprint_success("✅ Risk data prepared")
            
            # Calculate VaR metrics
            tprint_info("📊 Calculating VaR metrics...")
            self.logger.info("📊 Calculating VaR metrics...")
            var_metrics = self._calculate_var_metrics(risk_data)
            tprint_success("✅ VaR metrics calculated")
            
            # Calculate CVaR metrics
            tprint_info("📊 Calculating CVaR metrics...")
            self.logger.info("📊 Calculating CVaR metrics...")
            cvar_metrics = self._calculate_cvar_metrics(risk_data)
            tprint_success("✅ CVaR metrics calculated")
            
            # Calculate drawdown metrics
            tprint_info("📉 Calculating drawdown metrics...")
            self.logger.info("📉 Calculating drawdown metrics...")
            drawdown_metrics = self._calculate_drawdown_metrics(risk_data)
            tprint_success("✅ Drawdown metrics calculated")
            
            # Calculate risk ratios
            tprint_info("📈 Calculating risk ratios...")
            self.logger.info("📈 Calculating risk ratios...")
            risk_ratios = self._calculate_risk_ratios(risk_data)
            tprint_success("✅ Risk ratios calculated")
            
            # Calculate beta and alpha
            tprint_info("🔍 Calculating beta and alpha...")
            self.logger.info("🔍 Calculating beta and alpha...")
            beta_alpha_metrics = self._calculate_beta_alpha_metrics(risk_data)
            tprint_success("✅ Beta and alpha calculated")
            
            # Calculate higher moments
            tprint_info("📊 Calculating higher moments...")
            self.logger.info("📊 Calculating higher moments...")
            higher_moments = self._calculate_higher_moments(risk_data)
            tprint_success("✅ Higher moments calculated")
            
            # Run stress testing
            tprint_info("⚠️ Running stress testing...")
            self.logger.info("⚠️ Running stress testing...")
            stress_test_results = self._run_stress_testing(risk_data)
            tprint_success("✅ Stress testing completed")
            
            # Run scenario analysis
            tprint_info("🎯 Running scenario analysis...")
            self.logger.info("🎯 Running scenario analysis...")
            scenario_analysis = self._run_scenario_analysis(risk_data)
            tprint_success("✅ Scenario analysis completed")
            
            # Run risk attribution
            tprint_info("🔍 Running risk attribution...")
            self.logger.info("🔍 Running risk attribution...")
            risk_attribution = self._run_risk_attribution(risk_data)
            tprint_success("✅ Risk attribution completed")
            
            # Analyze regime risk
            tprint_info("🎯 Analyzing regime risk...")
            self.logger.info("🎯 Analyzing regime risk...")
            regime_risk_analysis = self._analyze_regime_risk(risk_data)
            tprint_success("✅ Regime risk analysis completed")
            
            # Create comprehensive result
            tprint_info("📋 Creating comprehensive risk analysis result...")
            result = RiskResult(
                # VaR metrics
                var_95=var_metrics['var_95'],
                var_99=var_metrics['var_99'],
                var_historical=var_metrics['var_historical'],
                var_parametric=var_metrics['var_parametric'],
                var_monte_carlo=var_metrics['var_monte_carlo'],
                
                # CVaR metrics
                cvar_95=cvar_metrics['cvar_95'],
                cvar_99=cvar_metrics['cvar_99'],
                cvar_historical=cvar_metrics['cvar_historical'],
                cvar_parametric=cvar_metrics['cvar_parametric'],
                cvar_monte_carlo=cvar_metrics['cvar_monte_carlo'],
                
                # Drawdown metrics
                max_drawdown=drawdown_metrics['max_drawdown'],
                average_drawdown=drawdown_metrics['average_drawdown'],
                drawdown_duration=drawdown_metrics['drawdown_duration'],
                recovery_time=drawdown_metrics['recovery_time'],
                
                # Risk ratios
                sharpe_ratio=risk_ratios['sharpe_ratio'],
                sortino_ratio=risk_ratios['sortino_ratio'],
                calmar_ratio=risk_ratios['calmar_ratio'],
                omega_ratio=risk_ratios['omega_ratio'],
                
                # Beta and Alpha
                beta=beta_alpha_metrics['beta'],
                alpha=beta_alpha_metrics['alpha'],
                tracking_error=beta_alpha_metrics['tracking_error'],
                information_ratio=beta_alpha_metrics['information_ratio'],
                
                # Higher moments
                volatility=higher_moments['volatility'],
                skewness=higher_moments['skewness'],
                kurtosis=higher_moments['kurtosis'],
                jarque_bera_stat=higher_moments['jarque_bera_stat'],
                jarque_bera_pvalue=higher_moments['jarque_bera_pvalue'],
                
                # Stress testing
                stress_test_results=stress_test_results['stress_test_results'],
                stress_scenarios=stress_test_results['stress_scenarios'],
                
                # Scenario analysis
                scenario_analysis=scenario_analysis['scenario_analysis'],
                expected_return=scenario_analysis['expected_return'],
                expected_volatility=scenario_analysis['expected_volatility'],
                
                # Risk attribution
                risk_attribution=risk_attribution['risk_attribution'],
                risk_contribution=risk_attribution['risk_contribution'],
                risk_impact=risk_attribution['risk_impact'],
                
                # Regime risk
                regime_risk=regime_risk_analysis['regime_risk'],
                regime_stability=regime_risk_analysis['regime_stability'],
                regime_transition_risk=regime_risk_analysis['regime_transition_risk'],
                
                # Time series
                returns_series=risk_data['returns_series'],
                drawdown_series=risk_data['drawdown_series'],
                risk_series=risk_data['risk_series'],
                
                # Metadata
                analysis_period=(returns_series.index[0], returns_series.index[-1]),
                execution_time=(datetime.now() - start_time).total_seconds(),
                config=self.config
            )
            
            # Save results if configured
            if self.config.save_risk_analysis:
                tprint_debug("💾 Saving risk analysis results...")
                self._save_results(result)
                tprint_success("✅ Risk analysis results saved")
            
            self.results = result
            tprint_success(f"✅ Risk analysis completed in {result.execution_time:.2f}s")
            tprint_info(f"📊 VaR 95%: {result.var_95:.2%}")
            tprint_info(f"📊 CVaR 95%: {result.cvar_95:.2%}")
            tprint_info(f"📉 Max drawdown: {result.max_drawdown:.2%}")
            tprint_info(f"📈 Sharpe ratio: {result.sharpe_ratio:.3f}")
            self.logger.info(f"✅ Risk analysis completed in {result.execution_time:.2f}s")
            self.logger.info(f"📊 VaR 95%: {result.var_95:.2%}")
            self.logger.info(f"📊 CVaR 95%: {result.cvar_95:.2%}")
            self.logger.info(f"📉 Max drawdown: {result.max_drawdown:.2%}")
            self.logger.info(f"📈 Sharpe ratio: {result.sharpe_ratio:.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Risk analysis failed: {e}")
            self.logger.error(f"❌ Risk analysis failed: {e}")
            raise
    
    def _validate_data(self, 
                      returns_series: pd.Series,
                      benchmark_returns: Optional[pd.Series],
                      regime_data: Optional[Dict[str, Any]],
                      factor_data: Optional[Dict[str, pd.Series]]):
        """Validate data for risk analysis."""
        if len(returns_series) < 30:
            raise ValueError(f"Insufficient data: {len(returns_series)} < 30")
        
        if benchmark_returns is not None and len(benchmark_returns) != len(returns_series):
            self.logger.warning("⚠️ Benchmark returns length mismatch with portfolio returns")
        
        if regime_data and len(regime_data) != len(returns_series):
            self.logger.warning("⚠️ Regime data length mismatch with returns series")
        
        if factor_data:
            for factor_name, factor_series in factor_data.items():
                if len(factor_series) != len(returns_series):
                    self.logger.warning(f"⚠️ Factor {factor_name} length mismatch with returns series")
        
        self.logger.info(f"✅ Data validation passed: {len(returns_series)} observations")
    
    def _prepare_risk_data(self, 
                          returns_series: pd.Series,
                          benchmark_returns: Optional[pd.Series],
                          regime_data: Optional[Dict[str, Any]],
                          factor_data: Optional[Dict[str, pd.Series]]) -> Dict[str, Any]:
        """Prepare data for risk analysis."""
        # Calculate drawdown series
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown_series = (cumulative_returns - running_max) / running_max
        
        # Calculate risk series (volatility)
        risk_series = returns_series.rolling(window=20).std()
        
        risk_data = {
            'returns_series': returns_series,
            'benchmark_returns': benchmark_returns,
            'regime_data': regime_data,
            'factor_data': factor_data,
            'drawdown_series': drawdown_series,
            'risk_series': risk_series,
            'cumulative_returns': cumulative_returns
        }
        
        return risk_data
    
    def _calculate_var_metrics(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate VaR metrics using multiple methods."""
        returns_series = risk_data['returns_series']
        
        var_metrics = {}
        
        # Historical VaR
        var_historical = {}
        for confidence_level in self.config.var_confidence_levels:
            var_historical[f'var_{int(confidence_level*100)}'] = np.percentile(returns_series, (1-confidence_level)*100)
        
        var_metrics['var_historical'] = var_historical
        var_metrics['var_95'] = var_historical.get('var_95', 0.0)
        var_metrics['var_99'] = var_historical.get('var_99', 0.0)
        
        # Parametric VaR
        var_parametric = {}
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        
        for confidence_level in self.config.var_confidence_levels:
            z_score = -1.645 if confidence_level == 0.95 else -2.326  # Approximate
            var_parametric[f'var_{int(confidence_level*100)}'] = mean_return + z_score * std_return
        
        var_metrics['var_parametric'] = var_parametric
        
        # Monte Carlo VaR (simplified)
        var_monte_carlo = {}
        n_simulations = 10000
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        for confidence_level in self.config.var_confidence_levels:
            var_monte_carlo[f'var_{int(confidence_level*100)}'] = np.percentile(simulated_returns, (1-confidence_level)*100)
        
        var_metrics['var_monte_carlo'] = var_monte_carlo
        
        return var_metrics
    
    def _calculate_cvar_metrics(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CVaR metrics using multiple methods."""
        returns_series = risk_data['returns_series']
        
        cvar_metrics = {}
        
        # Historical CVaR
        cvar_historical = {}
        for confidence_level in self.config.cvar_confidence_levels:
            var_threshold = np.percentile(returns_series, (1-confidence_level)*100)
            tail_returns = returns_series[returns_series <= var_threshold]
            cvar_historical[f'cvar_{int(confidence_level*100)}'] = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
        
        cvar_metrics['cvar_historical'] = cvar_historical
        cvar_metrics['cvar_95'] = cvar_historical.get('cvar_95', 0.0)
        cvar_metrics['cvar_99'] = cvar_historical.get('cvar_99', 0.0)
        
        # Parametric CVaR (simplified)
        cvar_parametric = {}
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        
        for confidence_level in self.config.cvar_confidence_levels:
            z_score = -1.645 if confidence_level == 0.95 else -2.326
            cvar_parametric[f'cvar_{int(confidence_level*100)}'] = mean_return + z_score * std_return
        
        cvar_metrics['cvar_parametric'] = cvar_parametric
        
        # Monte Carlo CVaR (simplified)
        cvar_monte_carlo = {}
        n_simulations = 10000
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        for confidence_level in self.config.cvar_confidence_levels:
            var_threshold = np.percentile(simulated_returns, (1-confidence_level)*100)
            tail_returns = simulated_returns[simulated_returns <= var_threshold]
            cvar_monte_carlo[f'cvar_{int(confidence_level*100)}'] = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
        
        cvar_metrics['cvar_monte_carlo'] = cvar_monte_carlo
        
        return cvar_metrics
    
    def _calculate_drawdown_metrics(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate drawdown metrics."""
        drawdown_series = risk_data['drawdown_series']
        
        # Maximum drawdown
        max_drawdown = drawdown_series.min()
        
        # Average drawdown
        average_drawdown = drawdown_series[drawdown_series < 0].mean() if len(drawdown_series[drawdown_series < 0]) > 0 else 0.0
        
        # Drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(drawdown_series)
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(drawdown_series)
        
        return {
            'max_drawdown': max_drawdown,
            'average_drawdown': average_drawdown,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time
        }
    
    def _calculate_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        in_drawdown = drawdown_series < 0
        drawdown_periods = []
        current_period = 0
        
        for is_drawdown in in_drawdown:
            if is_drawdown:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_recovery_time(self, drawdown_series: pd.Series) -> int:
        """Calculate average recovery time."""
        # Simplified calculation
        return self._calculate_drawdown_duration(drawdown_series)
    
    def _calculate_risk_ratios(self, risk_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk ratios."""
        returns_series = risk_data['returns_series']
        drawdown_series = risk_data['drawdown_series']
        
        # Sharpe ratio
        mean_return = returns_series.mean()
        std_return = returns_series.std()
        risk_free_rate = 0.02  # 2% annual
        sharpe_ratio = (mean_return - risk_free_rate/252) / std_return if std_return > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        sortino_ratio = (mean_return - risk_free_rate/252) / downside_std if downside_std > 0 else 0.0
        
        # Calmar ratio
        max_drawdown = abs(drawdown_series.min())
        calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Omega ratio (simplified)
        positive_returns = returns_series[returns_series > 0]
        negative_returns = returns_series[returns_series < 0]
        omega_ratio = positive_returns.sum() / abs(negative_returns.sum()) if negative_returns.sum() != 0 else float('inf')
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio
        }
    
    def _calculate_beta_alpha_metrics(self, risk_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate beta and alpha metrics."""
        returns_series = risk_data['returns_series']
        benchmark_returns = risk_data['benchmark_returns']
        
        if benchmark_returns is None:
            return {
                'beta': 1.0,
                'alpha': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0
            }
        
        # Calculate beta
        covariance = np.cov(returns_series, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # Calculate alpha
        mean_return = returns_series.mean()
        mean_benchmark = benchmark_returns.mean()
        alpha = mean_return - beta * mean_benchmark
        
        # Calculate tracking error
        excess_returns = returns_series - benchmark_returns
        tracking_error = excess_returns.std()
        
        # Calculate information ratio
        information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }
    
    def _calculate_higher_moments(self, risk_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate higher moments and statistical tests."""
        returns_series = risk_data['returns_series']
        
        # Basic moments
        volatility = returns_series.std()
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis()
        
        # Jarque-Bera test
        from scipy.stats import jarque_bera
        jb_stat, jb_pvalue = jarque_bera(returns_series)
        
        return {
            'volatility': volatility,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue
        }
    
    def _run_stress_testing(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress testing scenarios."""
        returns_series = risk_data['returns_series']
        
        stress_test_results = {}
        stress_scenarios = {}
        
        for scenario in self.config.stress_scenarios:
            scenario_results = {}
            
            for magnitude in self.config.stress_magnitudes:
                if scenario == 'market_crash':
                    stress_return = -magnitude
                elif scenario == 'volatility_spike':
                    stress_return = returns_series.mean() - magnitude * returns_series.std()
                elif scenario == 'liquidity_crisis':
                    stress_return = returns_series.mean() - magnitude * returns_series.std()
                elif scenario == 'regime_change':
                    stress_return = returns_series.mean() - magnitude * returns_series.std()
                else:
                    stress_return = -magnitude
                
                scenario_results[f'magnitude_{magnitude}'] = stress_return
            
            stress_test_results[scenario] = scenario_results
            stress_scenarios[scenario] = min(scenario_results.values())
        
        return {
            'stress_test_results': stress_test_results,
            'stress_scenarios': stress_scenarios
        }
    
    def _run_scenario_analysis(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run scenario analysis."""
        scenario_analysis = {}
        
        for i, (probability, scenario_return) in enumerate(zip(self.config.scenario_probabilities, self.config.scenario_returns)):
            scenario_analysis[f'scenario_{i+1}'] = {
                'probability': probability,
                'return': scenario_return
            }
        
        # Calculate expected return and volatility
        expected_return = sum(p * r for p, r in zip(self.config.scenario_probabilities, self.config.scenario_returns))
        expected_volatility = np.sqrt(sum(p * (r - expected_return)**2 for p, r in zip(self.config.scenario_probabilities, self.config.scenario_returns)))
        
        return {
            'scenario_analysis': scenario_analysis,
            'expected_return': expected_return,
            'expected_volatility': expected_volatility
        }
    
    def _run_risk_attribution(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run risk attribution analysis."""
        returns_series = risk_data['returns_series']
        factor_data = risk_data['factor_data']
        
        risk_attribution = {}
        risk_contribution = {}
        risk_impact = {}
        
        if factor_data:
            for factor_name, factor_series in factor_data.items():
                # Calculate factor risk contribution
                factor_correlation = returns_series.corr(factor_series)
                factor_volatility = factor_series.std()
                
                risk_attribution[factor_name] = factor_correlation
                risk_contribution[factor_name] = factor_correlation * factor_volatility
                risk_impact[factor_name] = abs(factor_correlation) * factor_volatility
        else:
            # Default risk factors
            for risk_factor in self.config.risk_factors:
                risk_attribution[risk_factor] = 0.0
                risk_contribution[risk_factor] = 0.0
                risk_impact[risk_factor] = 0.0
        
        return {
            'risk_attribution': risk_attribution,
            'risk_contribution': risk_contribution,
            'risk_impact': risk_impact
        }
    
    def _analyze_regime_risk(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-specific risk."""
        returns_series = risk_data['returns_series']
        regime_data = risk_data['regime_data']
        
        regime_risk = {}
        regime_stability = {}
        regime_transition_risk = 0.0
        
        if regime_data:
            regime_labels = regime_data.get('regime_labels', [])
            qualified_regimes = regime_data.get('qualified_regimes', {})
            
            for regime_name, regime_info in qualified_regimes.items():
                regime_id = regime_info.get('regime_id')
                regime_mask = np.array(regime_labels) == regime_id
                
                if np.any(regime_mask):
                    regime_returns = returns_series[regime_mask]
                    regime_risk[regime_name] = regime_returns.std()
                    regime_stability[regime_name] = 1.0 - regime_returns.std() / abs(regime_returns.mean()) if regime_returns.mean() != 0 else 0.0
        
        return {
            'regime_risk': regime_risk,
            'regime_stability': regime_stability,
            'regime_transition_risk': regime_transition_risk
        }
    
    def _save_results(self, result: RiskResult):
        """Save risk analysis results."""
        try:
            results_dir = Path(self.config.risk_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = results_dir / f"risk_analysis_summary_{timestamp}.json"
            
            summary = {
                'var_95': result.var_95,
                'var_99': result.var_99,
                'cvar_95': result.cvar_95,
                'cvar_99': result.cvar_99,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'volatility': result.volatility,
                'skewness': result.skewness,
                'kurtosis': result.kurtosis,
                'execution_time': result.execution_time
            }
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed results
            if self.config.detailed_risk_analysis:
                details_file = results_dir / f"risk_analysis_details_{timestamp}.json"
                
                details = {
                    'var_historical': result.var_historical,
                    'var_parametric': result.var_parametric,
                    'var_monte_carlo': result.var_monte_carlo,
                    'cvar_historical': result.cvar_historical,
                    'cvar_parametric': result.cvar_parametric,
                    'cvar_monte_carlo': result.cvar_monte_carlo,
                    'stress_test_results': result.stress_test_results,
                    'scenario_analysis': result.scenario_analysis,
                    'risk_attribution': result.risk_attribution,
                    'regime_risk': result.regime_risk
                }
                
                with open(details_file, 'w') as f:
                    json.dump(details, f, indent=2, default=str)
            
            self.logger.info(f"📁 Risk analysis results saved to {results_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save results: {e}")
    
    def get_results(self) -> Optional[RiskResult]:
        """Get risk analysis results."""
        return self.results
    
    def export_results(self, filepath: str):
        """Export results to file."""
        if self.results is None:
            self.logger.warning("⚠️ No results to export")
            return
        
        try:
            # Export returns series
            returns_file = filepath.replace('.csv', '_returns.csv')
            self.results.returns_series.to_csv(returns_file)
            
            # Export drawdown series
            drawdown_file = filepath.replace('.csv', '_drawdown.csv')
            self.results.drawdown_series.to_csv(drawdown_file)
            
            # Export risk series
            risk_file = filepath.replace('.csv', '_risk.csv')
            self.results.risk_series.to_csv(risk_file)
            
            self.logger.info(f"📁 Risk analysis results exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export results: {e}")