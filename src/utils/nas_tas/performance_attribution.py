"""
Unified Performance Attribution for Backtesting

This module provides unified performance attribution functionality for
analyzing backtesting results across TAS, NAS, and hybrid systems.
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
class PerformanceAttributionConfig:
    """Configuration for performance attribution."""
    
    # Benchmark parameters
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02  # 2% annual
    
    # Analysis parameters
    enable_benchmark_comparison: bool = True
    enable_risk_adjusted_metrics: bool = True
    enable_factor_analysis: bool = True
    enable_regime_analysis: bool = True
    
    # Time period parameters
    analysis_frequency: str = "daily"  # daily, weekly, monthly
    rolling_window: int = 252  # 1 year for daily data
    
    # Factor parameters
    factors: List[str] = field(default_factory=lambda: ["market", "size", "value", "momentum"])
    
    # Output parameters
    enable_detailed_report: bool = True
    enable_visualization: bool = True


@dataclass
class PerformanceMetrics:
    """Performance metrics for attribution analysis."""
    
    # Basic returns
    total_return: float
    annualized_return: float
    volatility: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int
    
    # Benchmark comparison
    excess_return: float
    tracking_error: float
    beta: float
    alpha: float
    
    # Factor exposures
    factor_exposures: Optional[Dict[str, float]] = None
    factor_returns: Optional[Dict[str, float]] = None
    
    # Regime analysis
    regime_performance: Optional[Dict[str, Dict[str, float]]] = None
    
    # Time series data
    returns_series: Optional[pd.Series] = None
    benchmark_returns: Optional[pd.Series] = None
    excess_returns_series: Optional[pd.Series] = None


class PerformanceAttribution:
    """
    Unified performance attribution analyzer for backtesting results.
    
    Provides comprehensive performance analysis including benchmark comparison,
    risk-adjusted metrics, factor analysis, and regime-specific performance.
    """
    
    def __init__(self, config: PerformanceAttributionConfig):
        """Initialize the performance attribution analyzer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> PerformanceMetrics:
        """
        Analyze performance attribution for strategy returns.
        
        Args:
            strategy_returns: Strategy returns time series
            benchmark_returns: Benchmark returns time series (optional)
            regime_labels: Regime labels for regime analysis (optional)
            factor_returns: Factor returns for factor analysis (optional)
            
        Returns:
            PerformanceMetrics with comprehensive analysis
        """
        self.logger.info("Starting performance attribution analysis")
        
        # Validate inputs
        strategy_returns = self._validate_returns(strategy_returns)
        
        if benchmark_returns is not None:
            benchmark_returns = self._validate_returns(benchmark_returns)
            strategy_returns, benchmark_returns = self._align_series(strategy_returns, benchmark_returns)
        
        # Calculate basic metrics
        basic_metrics = self._calculate_basic_metrics(strategy_returns)
        
        # Calculate risk-adjusted metrics
        risk_metrics = self._calculate_risk_metrics(strategy_returns)
        
        # Calculate benchmark comparison if available
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = self._calculate_benchmark_metrics(strategy_returns, benchmark_returns)
        
        # Calculate factor analysis if available
        factor_metrics = {}
        if factor_returns is not None:
            factor_metrics = self._calculate_factor_analysis(strategy_returns, factor_returns)
        
        # Calculate regime analysis if available
        regime_metrics = {}
        if regime_labels is not None:
            regime_metrics = self._calculate_regime_analysis(strategy_returns, regime_labels)
        
        # Combine all metrics
        metrics = PerformanceMetrics(
            # Basic metrics
            total_return=basic_metrics['total_return'],
            annualized_return=basic_metrics['annualized_return'],
            volatility=basic_metrics['volatility'],
            
            # Risk metrics
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            sortino_ratio=risk_metrics['sortino_ratio'],
            calmar_ratio=risk_metrics['calmar_ratio'],
            information_ratio=risk_metrics.get('information_ratio', 0.0),
            
            # Drawdown metrics
            max_drawdown=risk_metrics['max_drawdown'],
            avg_drawdown=risk_metrics['avg_drawdown'],
            drawdown_duration=risk_metrics['drawdown_duration'],
            
            # Benchmark comparison
            excess_return=benchmark_metrics.get('excess_return', 0.0),
            tracking_error=benchmark_metrics.get('tracking_error', 0.0),
            beta=benchmark_metrics.get('beta', 0.0),
            alpha=benchmark_metrics.get('alpha', 0.0),
            
            # Factor analysis
            factor_exposures=factor_metrics.get('exposures'),
            factor_returns=factor_metrics.get('returns'),
            
            # Regime analysis
            regime_performance=regime_metrics.get('performance'),
            
            # Time series
            returns_series=strategy_returns,
            benchmark_returns=benchmark_returns,
            excess_returns_series=benchmark_metrics.get('excess_returns_series')
        )
        
        self.logger.info("Performance attribution analysis completed")
        return metrics
    
    def _validate_returns(self, returns: pd.Series) -> pd.Series:
        """Validate and clean returns series."""
        # Remove NaN values
        returns = returns.dropna()
        
        # Check for infinite values
        if np.isinf(returns).any():
            self.logger.warning("Found infinite values in returns, replacing with NaN")
            returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check for extreme values
        q99 = returns.quantile(0.99)
        q01 = returns.quantile(0.01)
        extreme_mask = (returns > q99) | (returns < q01)
        
        if extreme_mask.any():
            self.logger.warning(f"Found {extreme_mask.sum()} extreme values in returns")
        
        return returns
    
    def _align_series(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align two time series."""
        # Find common index
        common_index = series1.index.intersection(series2.index)
        
        if len(common_index) == 0:
            raise ValueError("No common dates between strategy and benchmark returns")
        
        # Align series
        aligned_series1 = series1.loc[common_index]
        aligned_series2 = series2.loc[common_index]
        
        self.logger.info(f"Aligned series with {len(common_index)} common dates")
        return aligned_series1, aligned_series2
    
    def _calculate_basic_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean()
        
        # Drawdown duration
        drawdown_periods = drawdown < -0.01  # 1% threshold
        drawdown_duration = self._calculate_drawdown_duration(drawdown_periods)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'drawdown_duration': drawdown_duration
        }
    
    def _calculate_drawdown_duration(self, drawdown_periods: pd.Series) -> int:
        """Calculate maximum drawdown duration."""
        if not drawdown_periods.any():
            return 0
        
        # Find consecutive drawdown periods
        groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
        drawdown_lengths = drawdown_periods.groupby(groups).sum()
        
        return int(drawdown_lengths.max())
    
    def _calculate_benchmark_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # Basic metrics
        excess_return = excess_returns.mean() * 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Beta calculation
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha calculation
        alpha = excess_return - beta * (benchmark_returns.mean() * 252)
        
        # Information ratio
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        return {
            'excess_return': excess_return,
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'excess_returns_series': excess_returns
        }
    
    def _calculate_factor_analysis(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate factor analysis."""
        try:
            # Align returns with factors
            common_index = strategy_returns.index.intersection(factor_returns.index)
            aligned_strategy = strategy_returns.loc[common_index]
            aligned_factors = factor_returns.loc[common_index]
            
            # Simple linear regression for factor exposures
            from sklearn.linear_model import LinearRegression
            
            X = aligned_factors.values
            y = aligned_strategy.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Factor exposures
            factor_exposures = dict(zip(factor_returns.columns, model.coef_))
            
            # Factor returns contribution
            factor_returns_contribution = {}
            for factor, exposure in factor_exposures.items():
                factor_returns_contribution[factor] = exposure * aligned_factors[factor].mean() * 252
            
            return {
                'exposures': factor_exposures,
                'returns': factor_returns_contribution,
                'r_squared': model.score(X, y)
            }
            
        except Exception as e:
            self.logger.warning(f"Factor analysis failed: {e}")
            return {}
    
    def _calculate_regime_analysis(
        self,
        strategy_returns: pd.Series,
        regime_labels: pd.Series
    ) -> Dict[str, Any]:
        """Calculate regime-specific performance."""
        try:
            # Align returns with regime labels
            common_index = strategy_returns.index.intersection(regime_labels.index)
            aligned_returns = strategy_returns.loc[common_index]
            aligned_regimes = regime_labels.loc[common_index]
            
            regime_performance = {}
            
            for regime in aligned_regimes.unique():
                regime_returns = aligned_returns[aligned_regimes == regime]
                
                if len(regime_returns) > 0:
                    regime_metrics = {
                        'return': regime_returns.mean() * 252,
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe_ratio': (regime_returns.mean() * 252 - self.config.risk_free_rate) / (regime_returns.std() * np.sqrt(252)),
                        'max_drawdown': self._calculate_regime_drawdown(regime_returns),
                        'observations': len(regime_returns)
                    }
                    
                    regime_performance[str(regime)] = regime_metrics
            
            return {'performance': regime_performance}
            
        except Exception as e:
            self.logger.warning(f"Regime analysis failed: {e}")
            return {}
    
    def _calculate_regime_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a regime."""
        if len(returns) == 0:
            return 0
        
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        return drawdown.min()
    
    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate a detailed performance report."""
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 60)
        
        # Basic metrics
        report.append("\nBASIC METRICS:")
        report.append(f"Total Return: {metrics.total_return:.2%}")
        report.append(f"Annualized Return: {metrics.annualized_return:.2%}")
        report.append(f"Volatility: {metrics.volatility:.2%}")
        
        # Risk metrics
        report.append("\nRISK-ADJUSTED METRICS:")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
        report.append(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        
        # Benchmark comparison
        if metrics.benchmark_returns is not None:
            report.append("\nBENCHMARK COMPARISON:")
            report.append(f"Excess Return: {metrics.excess_return:.2%}")
            report.append(f"Tracking Error: {metrics.tracking_error:.2%}")
            report.append(f"Beta: {metrics.beta:.3f}")
            report.append(f"Alpha: {metrics.alpha:.2%}")
            report.append(f"Information Ratio: {metrics.information_ratio:.3f}")
        
        # Factor analysis
        if metrics.factor_exposures:
            report.append("\nFACTOR EXPOSURES:")
            for factor, exposure in metrics.factor_exposures.items():
                report.append(f"{factor}: {exposure:.3f}")
        
        # Regime analysis
        if metrics.regime_performance:
            report.append("\nREGIME PERFORMANCE:")
            for regime, perf in metrics.regime_performance.items():
                report.append(f"\n{regime}:")
                report.append(f"  Return: {perf['return']:.2%}")
                report.append(f"  Volatility: {perf['volatility']:.2%}")
                report.append(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
                report.append(f"  Max Drawdown: {perf['max_drawdown']:.2%}")
                report.append(f"  Observations: {perf['observations']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)