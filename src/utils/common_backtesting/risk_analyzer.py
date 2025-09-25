"""
Unified Risk Analyzer for Backtesting

This module provides unified risk analysis functionality for backtesting
results across TAS, NAS, and hybrid systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class RiskAnalysisConfig:
    """Configuration for risk analysis."""
    
    # Risk metrics parameters
    confidence_level: float = 0.95  # 95% confidence level
    enable_var: bool = True
    enable_cvar: bool = True
    enable_stress_testing: bool = True
    
    # Stress testing parameters
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash", "volatility_spike", "correlation_breakdown"
    ])
    
    # Tail risk parameters
    enable_tail_risk: bool = True
    extreme_percentile: float = 0.01  # 1% extreme events
    
    # Correlation analysis
    enable_correlation_analysis: bool = True
    rolling_correlation_window: int = 252
    
    # Liquidity risk
    enable_liquidity_analysis: bool = True
    
    # Output parameters
    enable_detailed_report: bool = True


@dataclass
class RiskMetrics:
    """Risk metrics for analysis."""
    
    # Value at Risk (VaR)
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Tail risk metrics
    expected_shortfall: float
    tail_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    recovery_time: int
    
    # Volatility metrics
    realized_volatility: float
    implied_volatility: Optional[float] = None
    volatility_of_volatility: float
    
    # Correlation metrics
    correlation_to_market: float
    correlation_stability: float
    
    # Stress test results
    stress_test_results: Optional[Dict[str, float]] = None
    
    # Liquidity metrics
    liquidity_metrics: Optional[Dict[str, float]] = None
    
    # Time series data
    returns_series: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    volatility_series: Optional[pd.Series] = None


class RiskAnalyzer:
    """
    Unified risk analyzer for backtesting results.
    
    Provides comprehensive risk analysis including VaR, CVaR, stress testing,
    tail risk analysis, and correlation analysis.
    """
    
    def __init__(self, config: RiskAnalysisConfig):
        """Initialize the risk analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Analyze risk metrics for returns series.
        
        Args:
            returns: Strategy returns time series
            market_returns: Market returns for correlation analysis (optional)
            
        Returns:
            Dictionary of risk metrics
        """
        self.logger.info("Starting risk analysis")
        
        # Validate inputs
        returns = self._validate_returns(returns)
        
        # Calculate VaR and CVaR
        var_metrics = self._calculate_var_metrics(returns)
        
        # Calculate tail risk metrics
        tail_metrics = self._calculate_tail_risk(returns)
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        
        # Calculate volatility metrics
        volatility_metrics = self._calculate_volatility_metrics(returns)
        
        # Calculate correlation metrics
        correlation_metrics = {}
        if market_returns is not None:
            correlation_metrics = self._calculate_correlation_metrics(returns, market_returns)
        
        # Perform stress testing
        stress_results = {}
        if self.config.enable_stress_testing:
            stress_results = self._perform_stress_testing(returns)
        
        # Combine all metrics
        risk_metrics = {
            **var_metrics,
            **tail_metrics,
            **drawdown_metrics,
            **volatility_metrics,
            **correlation_metrics,
            'stress_test_results': stress_results
        }
        
        self.logger.info("Risk analysis completed")
        return risk_metrics
    
    def _validate_returns(self, returns: pd.Series) -> pd.Series:
        """Validate and clean returns series."""
        # Remove NaN values
        returns = returns.dropna()
        
        # Check for infinite values
        if np.isinf(returns).any():
            self.logger.warning("Found infinite values in returns, replacing with NaN")
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) < 30:
            raise ValueError("Insufficient data for risk analysis (minimum 30 observations)")
        
        return returns
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Conditional VaR (CVaR) metrics."""
        # Historical VaR
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% confidence
        var_99 = np.percentile(returns, 1)  # 1st percentile for 99% confidence
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def _calculate_tail_risk(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate tail risk metrics."""
        # Expected shortfall (same as CVaR)
        var_threshold = np.percentile(returns, self.config.extreme_percentile * 100)
        expected_shortfall = returns[returns <= var_threshold].mean()
        
        # Tail ratio (ratio of extreme losses to extreme gains)
        extreme_losses = returns[returns <= var_threshold]
        extreme_gains = returns[returns >= np.percentile(returns, (1 - self.config.extreme_percentile) * 100)]
        
        if len(extreme_gains) > 0:
            tail_ratio = abs(extreme_losses.mean()) / extreme_gains.mean()
        else:
            tail_ratio = np.inf
        
        return {
            'expected_shortfall': expected_shortfall,
            'tail_ratio': tail_ratio
        }
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate drawdown-related risk metrics."""
        # Calculate cumulative returns and drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown (only negative values)
        negative_drawdowns = drawdown[drawdown < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
        
        # Drawdown duration
        drawdown_periods = drawdown < -0.01  # 1% threshold
        drawdown_duration = self._calculate_drawdown_duration(drawdown_periods)
        
        # Recovery time
        recovery_time = self._calculate_recovery_time(drawdown)
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_duration': drawdown_duration,
            'recovery_time': recovery_time
        }
    
    def _calculate_drawdown_duration(self, drawdown_periods: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        if not drawdown_periods.any():
            return 0
        
        # Find consecutive drawdown periods
        groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
        drawdown_lengths = drawdown_periods.groupby(groups).sum()
        
        return int(drawdown_lengths.max())
    
    def _calculate_recovery_time(self, drawdown: pd.Series) -> int:
        """Calculate average recovery time from drawdowns."""
        # Find drawdown periods
        drawdown_periods = drawdown < -0.01  # 1% threshold
        
        if not drawdown_periods.any():
            return 0
        
        # Find recovery periods (when drawdown returns to zero)
        recovery_periods = (drawdown >= 0) & drawdown_periods.shift()
        
        if not recovery_periods.any():
            return len(drawdown)  # Never recovered
        
        # Calculate recovery times
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, (is_drawdown, is_recovery) in enumerate(zip(drawdown_periods, recovery_periods)):
            if is_drawdown and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif is_recovery and in_drawdown:
                # Recovery
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
                in_drawdown = False
        
        return np.mean(recovery_times) if recovery_times else len(drawdown)
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-related risk metrics."""
        # Realized volatility (annualized)
        realized_volatility = returns.std() * np.sqrt(252)
        
        # Volatility of volatility (vol of vol)
        rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
        volatility_of_volatility = rolling_vol.std()
        
        return {
            'realized_volatility': realized_volatility,
            'volatility_of_volatility': volatility_of_volatility
        }
    
    def _calculate_correlation_metrics(
        self,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate correlation-related risk metrics."""
        # Align series
        common_index = returns.index.intersection(market_returns.index)
        aligned_returns = returns.loc[common_index]
        aligned_market = market_returns.loc[common_index]
        
        # Correlation to market
        correlation_to_market = aligned_returns.corr(aligned_market)
        
        # Correlation stability (rolling correlation)
        if len(aligned_returns) >= self.config.rolling_correlation_window:
            rolling_corr = aligned_returns.rolling(window=self.config.rolling_correlation_window).corr(aligned_market)
            correlation_stability = rolling_corr.std()
        else:
            correlation_stability = 0
        
        return {
            'correlation_to_market': correlation_to_market,
            'correlation_stability': correlation_stability
        }
    
    def _perform_stress_testing(self, returns: pd.Series) -> Dict[str, float]:
        """Perform stress testing scenarios."""
        stress_results = {}
        
        for scenario in self.config.stress_scenarios:
            if scenario == "market_crash":
                # Simulate market crash (large negative returns)
                crash_returns = returns.copy()
                crash_returns.iloc[-30:] *= -2  # Double the negative impact in last 30 periods
                stress_results[scenario] = crash_returns.sum()
                
            elif scenario == "volatility_spike":
                # Simulate volatility spike
                spike_returns = returns.copy()
                spike_returns.iloc[-21:] *= 3  # Triple volatility in last 21 periods
                stress_results[scenario] = spike_returns.sum()
                
            elif scenario == "correlation_breakdown":
                # Simulate correlation breakdown (randomize correlations)
                breakdown_returns = returns.copy()
                breakdown_returns.iloc[-60:] = np.random.normal(
                    breakdown_returns.iloc[-60:].mean(),
                    breakdown_returns.iloc[-60:].std() * 2,
                    len(breakdown_returns.iloc[-60:])
                )
                stress_results[scenario] = breakdown_returns.sum()
        
        return stress_results
    
    def calculate_liquidity_risk(self, volume_data: pd.Series) -> Dict[str, float]:
        """Calculate liquidity risk metrics."""
        if volume_data is None or len(volume_data) == 0:
            return {}
        
        # Volume volatility
        volume_volatility = volume_data.pct_change().std()
        
        # Volume trend
        volume_trend = volume_data.rolling(window=21).mean().pct_change().mean()
        
        # Liquidity ratio (average volume / standard deviation)
        liquidity_ratio = volume_data.mean() / volume_data.std()
        
        return {
            'volume_volatility': volume_volatility,
            'volume_trend': volume_trend,
            'liquidity_ratio': liquidity_ratio
        }
    
    def generate_risk_report(self, risk_metrics: Dict[str, float]) -> str:
        """Generate a detailed risk report."""
        report = []
        report.append("=" * 60)
        report.append("RISK ANALYSIS REPORT")
        report.append("=" * 60)
        
        # VaR and CVaR
        report.append("\nVALUE AT RISK:")
        report.append(f"VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
        report.append(f"VaR (99%): {risk_metrics.get('var_99', 0):.2%}")
        report.append(f"CVaR (95%): {risk_metrics.get('cvar_95', 0):.2%}")
        report.append(f"CVaR (99%): {risk_metrics.get('cvar_99', 0):.2%}")
        
        # Tail risk
        report.append("\nTAIL RISK:")
        report.append(f"Expected Shortfall: {risk_metrics.get('expected_shortfall', 0):.2%}")
        report.append(f"Tail Ratio: {risk_metrics.get('tail_ratio', 0):.3f}")
        
        # Drawdown metrics
        report.append("\nDRAWDOWN METRICS:")
        report.append(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Avg Drawdown: {risk_metrics.get('avg_drawdown', 0):.2%}")
        report.append(f"Max Drawdown Duration: {risk_metrics.get('drawdown_duration', 0)} periods")
        report.append(f"Avg Recovery Time: {risk_metrics.get('recovery_time', 0):.1f} periods")
        
        # Volatility metrics
        report.append("\nVOLATILITY METRICS:")
        report.append(f"Realized Volatility: {risk_metrics.get('realized_volatility', 0):.2%}")
        report.append(f"Volatility of Volatility: {risk_metrics.get('volatility_of_volatility', 0):.2%}")
        
        # Correlation metrics
        if 'correlation_to_market' in risk_metrics:
            report.append("\nCORRELATION METRICS:")
            report.append(f"Correlation to Market: {risk_metrics.get('correlation_to_market', 0):.3f}")
            report.append(f"Correlation Stability: {risk_metrics.get('correlation_stability', 0):.3f}")
        
        # Stress testing
        stress_results = risk_metrics.get('stress_test_results', {})
        if stress_results:
            report.append("\nSTRESS TEST RESULTS:")
            for scenario, result in stress_results.items():
                report.append(f"{scenario.replace('_', ' ').title()}: {result:.2%}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)