"""
VectorBT-Enhanced Financial Metrics

This module provides comprehensive financial metrics using VectorBT for
portfolio analysis, risk assessment, and performance evaluation.

Key Features:
- 50+ financial performance metrics
- Risk-adjusted return calculations
- Drawdown analysis and recovery metrics
- Regime-aware performance analysis
- Benchmark comparison utilities
- Integration with existing evaluation framework
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.records.base import Records
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None
    Records = None

# Optional dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetricCategory(Enum):
    """Categories of financial metrics."""
    RETURNS = "returns"
    RISK = "risk"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    TRADING = "trading"
    REGIME = "regime"
    BENCHMARK = "benchmark"

@dataclass
class FinancialMetricsConfig:
    """Configuration for financial metrics calculation."""
    # Basic settings
    risk_free_rate: float = 0.02
    benchmark_symbol: Optional[str] = None
    benchmark_data: Optional[pd.Series] = None

    # Time periods
    annualization_factor: int = 252  # Trading days per year
    lookback_periods: int = 252  # Lookback for rolling metrics

    # Risk settings
    confidence_level: float = 0.05  # For VaR/CVaR
    downside_threshold: float = 0.0  # For downside deviation

    # Regime settings
    enable_regime_analysis: bool = True
    regime_threshold: float = 0.1  # Threshold for regime detection

    # Performance settings
    enable_parallel: bool = True
    chunk_size: int = 1000

class VectorBTFinancialMetrics:
    """
    Comprehensive financial metrics calculator using VectorBT.

    This class provides 50+ financial metrics including:
    - Return metrics (total, annualized, cumulative)
    - Risk metrics (volatility, VaR, CVaR, downside deviation)
    - Risk-adjusted metrics (Sharpe, Sortino, Calmar, Information ratio)
    - Drawdown metrics (max, average, duration, recovery)
    - Trading metrics (win rate, profit factor, expectancy)
    - Regime analysis (bull/bear market performance)
    - Benchmark comparison (alpha, beta, tracking error)
    """

    def __init__(self, config: Optional[FinancialMetricsConfig] = None):
        """
        Initialize VectorBT financial metrics calculator.

        Args:
            config: Configuration for metrics calculation
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")

        self.config = config or FinancialMetricsConfig()

        # Initialize VectorBT settings
        self._configure_vectorbt()

        logger.info("✅ VectorBT Financial Metrics initialized")
        logger.info(f"📊 Risk-free rate: {self.config.risk_free_rate:.2%}")
        logger.info(f"📊 Annualization factor: {self.config.annualization_factor}")
        logger.info(f"📊 Regime analysis: {self.config.enable_regime_analysis}")

    def _configure_vectorbt(self):
        """Configure VectorBT global settings."""
        try:
            # Check if settings attribute exists first
            if not hasattr(vbt, 'settings'):
                logger.debug("⚠️ VectorBT settings not available in this version")
                return
            
            # Set parallel processing (check if attribute exists)
            if self.config.enable_parallel and hasattr(vbt.settings, 'parallel'):
                if 'threading' in vbt.settings['parallel']:
                    vbt.settings.parallel['threading'] = True
                    logger.debug("✅ VectorBT parallel processing enabled")
            elif self.config.enable_parallel:
                logger.debug("⚠️ VectorBT parallel settings not available in this version")

            # Set array wrapper settings (check if attribute exists)
            if hasattr(vbt.settings, 'array_wrapper'):
                if 'freq' in vbt.settings['array_wrapper']:
                    vbt.settings.array_wrapper['freq'] = '1min'
                    logger.debug("✅ VectorBT array wrapper frequency set to 1min")
            else:
                logger.debug("⚠️ VectorBT array_wrapper settings not available in this version")

        except Exception as e:
            logger.warning(f"⚠️ Failed to configure VectorBT settings: {e}")

    def calculate_comprehensive_metrics(self,
                                      portfolio_values: Union[np.ndarray, pd.Series],
                                      returns: Optional[Union[np.ndarray, pd.Series]] = None,
                                      benchmark_values: Optional[Union[np.ndarray, pd.Series]] = None,
                                      timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive financial metrics.

        Args:
            portfolio_values: Portfolio value series
            returns: Portfolio returns series (optional, will calculate if not provided)
            benchmark_values: Benchmark value series for comparison
            timestamps: Time index for the data

        Returns:
            Dictionary containing all calculated metrics
        """
        logger.info("📊 Calculating comprehensive financial metrics...")

        # Prepare data
        portfolio_series = self._prepare_series(portfolio_values, timestamps)
        returns_series = returns if returns is not None else self._calculate_returns(portfolio_series)
        benchmark_series = self._prepare_series(benchmark_values, timestamps) if benchmark_values is not None else None

        # Calculate metrics by category
        metrics = {}

        # Return metrics
        metrics.update(self._calculate_return_metrics(portfolio_series, returns_series))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns_series))

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns_series))

        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(portfolio_series))

        # Trading metrics
        metrics.update(self._calculate_trading_metrics(returns_series))

        # Regime analysis
        if self.config.enable_regime_analysis:
            metrics.update(self._calculate_regime_metrics(portfolio_series, returns_series))

        # Benchmark comparison
        if benchmark_series is not None:
            metrics.update(self._calculate_benchmark_metrics(portfolio_series, returns_series, benchmark_series))

        logger.info(f"✅ Calculated {len(metrics)} financial metrics")
        return metrics

    def _prepare_series(self, data: Union[np.ndarray, pd.Series], timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> pd.Series:
        """Prepare data as pandas Series with proper index."""
        if isinstance(data, np.ndarray):
            if timestamps is not None:
                if isinstance(timestamps, pd.DatetimeIndex):
                    index = timestamps
                else:
                    index = pd.DatetimeIndex(timestamps)
            else:
                index = pd.date_range(start='2020-01-01', periods=len(data), freq='1min')
            return pd.Series(data, index=index)
        else:
            return data

    def _calculate_returns(self, portfolio_values: pd.Series) -> pd.Series:
        """Calculate returns from portfolio values."""
        return portfolio_values.pct_change().dropna()

    def _calculate_return_metrics(self, portfolio_values: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate return-based metrics."""
        logger.debug("📊 Calculating return metrics...")

        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (self.config.annualization_factor / len(returns)) - 1
        cumulative_return = total_return

        # Rolling returns
        rolling_returns = returns.rolling(window=self.config.lookback_periods)
        avg_rolling_return = rolling_returns.mean().mean()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'cumulative_return': cumulative_return,
            'avg_rolling_return': avg_rolling_return,
            'best_month_return': returns.resample('M').sum().max(),
            'worst_month_return': returns.resample('M').sum().min(),
            'positive_return_pct': (returns > 0).mean(),
            'negative_return_pct': (returns < 0).mean()
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        logger.debug("📊 Calculating risk metrics...")

        # Basic volatility
        volatility = returns.std() * np.sqrt(self.config.annualization_factor)

        # Downside deviation
        downside_returns = returns[returns < self.config.downside_threshold]
        downside_deviation = downside_returns.std() * np.sqrt(self.config.annualization_factor) if len(downside_returns) > 0 else 0

        # VaR and CVaR
        var_95 = returns.quantile(self.config.confidence_level)
        cvar_95 = returns[returns <= var_95].mean()

        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Tail ratio
        tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0

        return {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'max_daily_loss': returns.min(),
            'max_daily_gain': returns.max(),
            'avg_daily_return': returns.mean(),
            'median_daily_return': returns.median()
        }

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        logger.debug("📊 Calculating risk-adjusted metrics...")

        # Basic metrics
        avg_return = returns.mean() * self.config.annualization_factor
        volatility = returns.std() * np.sqrt(self.config.annualization_factor)
        downside_deviation = returns[returns < self.config.downside_threshold].std() * np.sqrt(self.config.annualization_factor)

        # Sharpe ratio
        excess_return = avg_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Sortino ratio
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0

        # Calmar ratio
        total_return = (1 + returns).prod() - 1
        max_drawdown = self._calculate_max_drawdown_from_returns(returns)
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Information ratio (vs risk-free rate)
        tracking_error = (returns - self.config.risk_free_rate / self.config.annualization_factor).std() * np.sqrt(self.config.annualization_factor)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0

        # Treynor ratio (requires beta, simplified here)
        treynor_ratio = excess_return / 1.0  # Assuming beta = 1 for simplicity

        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'excess_return': excess_return,
            'tracking_error': tracking_error
        }

    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        logger.debug("📊 Calculating drawdown metrics...")

        # Calculate drawdowns
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak

        # Basic drawdown metrics
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

        # Drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_durations = self._calculate_drawdown_durations(drawdown_periods)
        max_drawdown_duration = drawdown_durations.max() if len(drawdown_durations) > 0 else 0
        avg_drawdown_duration = drawdown_durations.mean() if len(drawdown_durations) > 0 else 0

        # Recovery time
        recovery_time = self._calculate_recovery_time(portfolio_values, peak)

        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_time': recovery_time,
            'drawdown_series': drawdown,
            'underwater_curve': drawdown
        }

    def _calculate_trading_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate trading-related metrics."""
        logger.debug("📊 Calculating trading metrics...")

        # Win/loss analysis
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        loss_rate = len(losses) / len(returns) if len(returns) > 0 else 0

        # Average win/loss
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Expectancy
        expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

        # Best/worst trades
        best_trade = returns.max()
        worst_trade = returns.min()

        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_wins_losses(returns > 0)
        consecutive_losses = self._calculate_consecutive_wins_losses(returns < 0)

        return {
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'total_trades': len(returns)
        }

    def _calculate_regime_metrics(self, portfolio_values: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Calculate regime-aware performance metrics."""
        logger.debug("📊 Calculating regime metrics...")

        # Simple regime detection based on rolling returns
        rolling_returns = returns.rolling(window=20).mean()
        regime_threshold = rolling_returns.std() * self.config.regime_threshold

        # Bull/bear market detection
        bull_market = rolling_returns > regime_threshold
        bear_market = rolling_returns < -regime_threshold
        neutral_market = ~(bull_market | bear_market)

        # Performance in different regimes
        bull_performance = returns[bull_market].mean() * self.config.annualization_factor if bull_market.any() else 0
        bear_performance = returns[bear_market].mean() * self.config.annualization_factor if bear_market.any() else 0
        neutral_performance = returns[neutral_market].mean() * self.config.annualization_factor if neutral_market.any() else 0

        # Regime duration
        bull_duration = self._calculate_regime_duration(bull_market)
        bear_duration = self._calculate_regime_duration(bear_market)
        neutral_duration = self._calculate_regime_duration(neutral_market)

        return {
            'bull_market_performance': bull_performance,
            'bear_market_performance': bear_performance,
            'neutral_market_performance': neutral_performance,
            'bull_market_duration': bull_duration,
            'bear_market_duration': bear_duration,
            'neutral_market_duration': neutral_duration,
            'regime_stability': 1 - (bull_market.astype(int).diff().abs().sum() / len(returns))
        }

    def _calculate_benchmark_metrics(self, portfolio_values: pd.Series, returns: pd.Series, benchmark_values: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        logger.debug("📊 Calculating benchmark metrics...")

        # Align data
        aligned_data = pd.DataFrame({
            'portfolio': portfolio_values,
            'benchmark': benchmark_values
        }).dropna()

        if len(aligned_data) == 0:
            return {}

        portfolio_aligned = aligned_data['portfolio']
        benchmark_aligned = aligned_data['benchmark']

        # Calculate returns
        portfolio_returns = portfolio_aligned.pct_change().dropna()
        benchmark_returns = benchmark_aligned.pct_change().dropna()

        # Align returns
        returns_data = pd.DataFrame({
            'portfolio': portfolio_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(returns_data) == 0:
            return {}

        portfolio_returns = returns_data['portfolio']
        benchmark_returns = returns_data['benchmark']

        # Calculate metrics
        excess_returns = portfolio_returns - benchmark_returns

        # Alpha and Beta
        if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            alpha = portfolio_returns.mean() - (beta * benchmark_returns.mean())
        else:
            alpha = 0
            beta = 0

        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(self.config.annualization_factor)

        # Information ratio
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Relative performance
        portfolio_total_return = (1 + portfolio_returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        relative_performance = portfolio_total_return - benchmark_total_return

        return {
            'alpha': alpha,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'relative_performance': relative_performance,
            'portfolio_total_return': portfolio_total_return,
            'benchmark_total_return': benchmark_total_return,
            'excess_return': excess_returns.mean()
        }

    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _calculate_drawdown_durations(self, drawdown_periods: pd.Series) -> pd.Series:
        """Calculate duration of each drawdown period."""
        # Find start and end of drawdown periods
        starts = drawdown_periods.diff() == 1
        ends = drawdown_periods.diff() == -1

        durations = []
        current_duration = 0

        for i, is_drawdown in enumerate(drawdown_periods):
            if is_drawdown:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        # Handle case where series ends in drawdown
        if current_duration > 0:
            durations.append(current_duration)

        return pd.Series(durations)

    def _calculate_recovery_time(self, portfolio_values: pd.Series, peak: pd.Series) -> int:
        """Calculate recovery time from maximum drawdown."""
        drawdown = (portfolio_values - peak) / peak
        max_dd_idx = drawdown.idxmin()

        # Find when portfolio recovers to the peak before max drawdown
        peak_before_dd = peak.loc[max_dd_idx]
        recovery_mask = portfolio_values[portfolio_values.index >= max_dd_idx] >= peak_before_dd

        if recovery_mask.any():
            return recovery_mask.idxmax() - max_dd_idx
        else:
            return len(portfolio_values) - portfolio_values.index.get_loc(max_dd_idx)

    def _calculate_consecutive_wins_losses(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive wins or losses."""
        if not condition.any():
            return 0

        # Find groups of consecutive True values
        groups = (condition != condition.shift()).cumsum()
        consecutive_counts = condition.groupby(groups).sum()

        return consecutive_counts.max() if len(consecutive_counts) > 0 else 0

    def _calculate_regime_duration(self, regime_mask: pd.Series) -> int:
        """Calculate average duration of a regime."""
        if not regime_mask.any():
            return 0

        # Find regime periods
        starts = regime_mask.diff() == 1
        ends = regime_mask.diff() == -1

        durations = []
        current_duration = 0

        for i, in_regime in enumerate(regime_mask):
            if in_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0

        # Handle case where series ends in regime
        if current_duration > 0:
            durations.append(current_duration)

        return np.mean(durations) if durations else 0

    def get_metric_categories(self) -> Dict[str, List[str]]:
        """Get available metrics organized by category."""
        return {
            'returns': [
                'total_return', 'annualized_return', 'cumulative_return',
                'avg_rolling_return', 'best_month_return', 'worst_month_return',
                'positive_return_pct', 'negative_return_pct'
            ],
            'risk': [
                'volatility', 'downside_deviation', 'var_95', 'cvar_95',
                'skewness', 'kurtosis', 'tail_ratio', 'max_daily_loss',
                'max_daily_gain', 'avg_daily_return', 'median_daily_return'
            ],
            'risk_adjusted': [
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'information_ratio', 'treynor_ratio', 'excess_return', 'tracking_error'
            ],
            'drawdown': [
                'max_drawdown', 'avg_drawdown', 'max_drawdown_duration',
                'avg_drawdown_duration', 'recovery_time'
            ],
            'trading': [
                'win_rate', 'loss_rate', 'avg_win', 'avg_loss',
                'profit_factor', 'expectancy', 'best_trade', 'worst_trade',
                'max_consecutive_wins', 'max_consecutive_losses', 'total_trades'
            ],
            'regime': [
                'bull_market_performance', 'bear_market_performance',
                'neutral_market_performance', 'bull_market_duration',
                'bear_market_duration', 'neutral_market_duration', 'regime_stability'
            ],
            'benchmark': [
                'alpha', 'beta', 'tracking_error', 'information_ratio',
                'relative_performance', 'portfolio_total_return',
                'benchmark_total_return', 'excess_return'
            ]
        }

# Convenience functions
def calculate_financial_metrics(portfolio_values: Union[np.ndarray, pd.Series],
                               returns: Optional[Union[np.ndarray, pd.Series]] = None,
                               benchmark_values: Optional[Union[np.ndarray, pd.Series]] = None,
                               config: Optional[FinancialMetricsConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to calculate financial metrics.

    Args:
        portfolio_values: Portfolio value series
        returns: Portfolio returns series
        benchmark_values: Benchmark value series
        config: Configuration for metrics calculation

    Returns:
        Dictionary containing all calculated metrics
    """
    calculator = VectorBTFinancialMetrics(config)
    return calculator.calculate_comprehensive_metrics(portfolio_values, returns, benchmark_values)

def create_metrics_config(risk_free_rate: float = 0.02,
                         annualization_factor: int = 252,
                         enable_regime_analysis: bool = True,
                         **kwargs) -> FinancialMetricsConfig:
    """
    Create financial metrics configuration.

    Args:
        risk_free_rate: Risk-free rate for calculations
        annualization_factor: Annualization factor for returns
        enable_regime_analysis: Whether to enable regime analysis
        **kwargs: Additional configuration parameters

    Returns:
        Financial metrics configuration
    """
    return FinancialMetricsConfig(
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
        enable_regime_analysis=enable_regime_analysis,
        **kwargs
    )

if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing VectorBT Financial Metrics...")

    # Generate sample data
    np.random.seed(42)
    n_periods = 1000

    # Generate random portfolio values
    returns = np.random.normal(0.001, 0.02, n_periods)
    portfolio_values = 100000 * (1 + returns).cumprod()

    # Generate benchmark data
    benchmark_returns = np.random.normal(0.0008, 0.015, n_periods)
    benchmark_values = 100000 * (1 + benchmark_returns).cumprod()

    # Create timestamps
    timestamps = pd.date_range(start='2020-01-01', periods=n_periods, freq='1min')

    # Test metrics calculator
    config = create_metrics_config(risk_free_rate=0.02)
    calculator = VectorBTFinancialMetrics(config)

    # Calculate metrics
    metrics = calculator.calculate_comprehensive_metrics(
        portfolio_values,
        returns,
        benchmark_values,
        timestamps
    )

    # Print key metrics
    print(f"\n📊 Financial Metrics Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Alpha: {metrics.get('alpha', 0):.4f}")
    print(f"Beta: {metrics.get('beta', 0):.3f}")

    # Print available categories
    categories = calculator.get_metric_categories()
    print(f"\n📊 Available Metric Categories:")
    for category, metric_list in categories.items():
        print(f"{category}: {len(metric_list)} metrics")

    print("\n✅ VectorBT Financial Metrics test completed!")
