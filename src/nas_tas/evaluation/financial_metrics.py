"""
Financial Metrics Calculator for NAS/TAS Systems

This module provides comprehensive financial metrics calculation and validation
for both NAS and TAS implementations, consolidating duplicate financial evaluation logic.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

@dataclass
class TradingPerformanceMetrics:
    """Trading performance metrics container."""

    # Basic performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR 95%

    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_duration: float = 0.0
    avg_bars_in_trade: float = 0.0

    # Advanced metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    beta: float = 0.0

    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (int, float)):
                result[field_name] = float(field_value)
            else:
                result[field_name] = field_value
        return result

@dataclass
class RiskMetrics:
    """Risk metrics container."""

    # Downside risk metrics
    downside_deviation: float = 0.0
    semi_deviation: float = 0.0
    downside_variance: float = 0.0

    # Tail risk metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0

    # Risk-adjusted returns
    omega_ratio: float = 0.0
    sterling_ratio: float = 0.0
    burke_ratio: float = 0.0

    # Extreme value metrics
    expected_shortfall: float = 0.0
    conditional_drawdown_at_risk: float = 0.0
    ulcer_index: float = 0.0

    # Stability metrics
    stability_of_returns: float = 0.0
    recovery_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {field_name: float(field_value) for field_name, field_value in self.__dict__.items()}

@dataclass
class FinancialValidationResult:
    """Result of financial validation."""

    # Validation status
    passed_validation: bool = False
    validation_score: float = 0.0

    # Threshold checks
    passed_sharpe_threshold: bool = False
    passed_drawdown_threshold: bool = False
    passed_win_rate_threshold: bool = False
    passed_profit_factor_threshold: bool = False

    # Performance metrics
    performance_metrics: TradingPerformanceMetrics = field(default_factory=TradingPerformanceMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)

    # Validation details
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'passed_validation': self.passed_validation,
            'validation_score': self.validation_score,
            'passed_sharpe_threshold': self.passed_sharpe_threshold,
            'passed_drawdown_threshold': self.passed_drawdown_threshold,
            'passed_win_rate_threshold': self.passed_win_rate_threshold,
            'passed_profit_factor_threshold': self.passed_profit_factor_threshold,
            'performance_metrics': self.performance_metrics.to_dict(),
            'risk_metrics': self.risk_metrics.to_dict(),
            'validation_errors': self.validation_errors,
            'validation_warnings': self.validation_warnings,
            'recommendations': self.recommendations,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'validation_duration': self.validation_duration
        }

class FinancialMetricsCalculator:
    """
    Comprehensive financial metrics calculator for NAS/TAS systems.

    This class consolidates financial evaluation logic that was previously
    duplicated between NAS and TAS implementations, providing a unified
    interface for financial performance analysis.
    """

    def __init__(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252):
        """
        Initialize financial metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            trading_days_per_year: Number of trading days per year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.logger = logging.getLogger(self.__class__.__name__)

        tprint_info(f"Financial metrics calculator initialized with risk-free rate: {risk_free_rate:.2%}")

    def calculate_performance_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> TradingPerformanceMetrics:
        """
        Calculate comprehensive trading performance metrics.

        Args:
            returns: Array of returns (daily or intraday)
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            TradingPerformanceMetrics object with calculated metrics
        """
        tprint_info("Calculating trading performance metrics")

        if len(returns) == 0:
            tprint_warning("Empty returns array provided")
            return TradingPerformanceMetrics()

        try:
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]  # Remove NaN values

            if len(returns) == 0:
                tprint_warning("No valid returns after removing NaN values")
                return TradingPerformanceMetrics()

            metrics = TradingPerformanceMetrics()

            # Basic performance metrics
            metrics.total_return = np.prod(1 + returns) - 1
            metrics.annualized_return = self._annualize_return(returns)
            metrics.volatility = np.std(returns) * np.sqrt(self.trading_days_per_year)

            # Risk-adjusted returns
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
            metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
            metrics.calmar_ratio = self._calculate_calmar_ratio(returns)

            # Risk metrics
            metrics.max_drawdown = self._calculate_max_drawdown(returns)
            metrics.max_drawdown_duration = self._calculate_max_drawdown_duration(returns)

            # Value at Risk metrics
            metrics.var_95 = np.percentile(returns, 5)
            cvar_returns = returns[returns <= metrics.var_95]
            metrics.cvar_95 = np.mean(cvar_returns) if len(cvar_returns) > 0 else metrics.var_95

            # Trading statistics
            metrics.win_rate = np.mean(returns > 0)
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]

            if len(winning_returns) > 0:
                metrics.avg_win = np.mean(winning_returns)
                metrics.largest_win = np.max(winning_returns)
                metrics.winning_trades = len(winning_returns)

            if len(losing_returns) > 0:
                metrics.avg_loss = np.mean(losing_returns)
                metrics.largest_loss = np.min(losing_returns)
                metrics.losing_trades = len(losing_returns)

            metrics.total_trades = len(returns)

            # Profit factor
            total_wins = np.sum(winning_returns) if len(winning_returns) > 0 else 0
            total_losses = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 0

            # Safe division with validation
            if total_losses > 1e-8:  # Avoid division by very small numbers
                metrics.profit_factor = total_wins / total_losses
            elif total_wins > 0:
                metrics.profit_factor = float('inf')  # All wins, no losses
            else:
                metrics.profit_factor = 0.0  # No wins, no losses

            # Benchmark comparison metrics
            if benchmark_returns is not None:
                benchmark_returns = np.array(benchmark_returns)
                if len(benchmark_returns) == len(returns):
                    excess_returns = returns - benchmark_returns
                    metrics.information_ratio = self._calculate_information_ratio(excess_returns)

                    # Beta calculation
                    if np.var(benchmark_returns) > 0:
                        metrics.beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)

                    # Jensen's Alpha
                    benchmark_annual_return = self._annualize_return(benchmark_returns)
                    metrics.jensen_alpha = metrics.annualized_return - (
                        self.risk_free_rate + metrics.beta * (benchmark_annual_return - self.risk_free_rate)
                    )

            tprint_success(f"Performance metrics calculated: Sharpe={metrics.sharpe_ratio:.3f}, "
                          f"Max DD={metrics.max_drawdown:.3f}, Win Rate={metrics.win_rate:.3f}")

            return metrics

        except Exception as e:
            tprint_error(f"Error calculating performance metrics: {e}")
            self.logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return TradingPerformanceMetrics()

    def calculate_risk_metrics(self, returns: np.ndarray) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Array of returns

        Returns:
            RiskMetrics object with calculated risk metrics
        """
        tprint_info("Calculating risk metrics")

        if len(returns) == 0:
            return RiskMetrics()

        try:
            returns = np.array(returns)
            returns = returns[~np.isnan(returns)]

            if len(returns) == 0:
                return RiskMetrics()

            metrics = RiskMetrics()

            # Downside risk metrics
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                metrics.downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
                metrics.semi_deviation = metrics.downside_deviation
                metrics.downside_variance = np.var(negative_returns)

            # Higher moment metrics
            if len(returns) > 3:
                metrics.skewness = stats.skew(returns)
                metrics.kurtosis = stats.kurtosis(returns)

                # Tail ratio (95th percentile / 5th percentile)
                p95 = np.percentile(returns, 95)
                p5 = np.percentile(returns, 5)
                metrics.tail_ratio = abs(p95 / p5) if p5 != 0 else float('inf')

            # Omega ratio
            if metrics.downside_deviation > 0:
                threshold = 0.0  # Risk-free rate threshold
                excess_returns = returns - threshold
                positive_excess = excess_returns[excess_returns > 0]
                negative_excess = excess_returns[excess_returns < 0]

                if len(negative_excess) > 0:
                    metrics.omega_ratio = (np.sum(positive_excess) / abs(np.sum(negative_excess))) if len(negative_excess) > 0 else float('inf')

            # Sterling ratio (annual return / average drawdown)
            avg_drawdown = self._calculate_average_drawdown(returns)
            if avg_drawdown > 0:
                metrics.sterling_ratio = self._annualize_return(returns) / avg_drawdown

            # Burke ratio (annual return / sqrt(sum of squared drawdowns))
            burke_denominator = self._calculate_burke_denominator(returns)
            if burke_denominator > 0:
                metrics.burke_ratio = self._annualize_return(returns) / burke_denominator

            # Expected shortfall (CVaR)
            metrics.expected_shortfall = np.mean(returns[returns <= np.percentile(returns, 5)])

            # Ulcer index
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (peak - cumulative_returns) / peak
            metrics.ulcer_index = np.sqrt(np.mean(drawdowns ** 2))

            # Stability of returns
            if len(returns) > 1:
                metrics.stability_of_returns = 1.0 - np.std(returns) / (np.mean(returns) + 1e-8)

            # Recovery factor
            max_dd = self._calculate_max_drawdown(returns)
            if max_dd > 0:
                metrics.recovery_factor = self._annualize_return(returns) / max_dd

            tprint_success(f"Risk metrics calculated: Skewness={metrics.skewness:.3f}, "
                          f"Kurtosis={metrics.kurtosis:.3f}, Omega={metrics.omega_ratio:.3f}")

            return metrics

        except Exception as e:
            tprint_error(f"Error calculating risk metrics: {e}")
            self.logger.error(f"Error calculating risk metrics: {e}", exc_info=True)
            return RiskMetrics()

    def validate_financial_performance(
        self,
        returns: np.ndarray,
        thresholds: Dict[str, float],
        benchmark_returns: Optional[np.ndarray] = None
    ) -> FinancialValidationResult:
        """
        Validate financial performance against thresholds.

        Args:
            returns: Array of returns
            thresholds: Dictionary of validation thresholds
            benchmark_returns: Optional benchmark returns

        Returns:
            FinancialValidationResult with validation results
        """
        tprint_info("Validating financial performance")

        start_time = datetime.now()

        try:
            # Calculate metrics
            performance_metrics = self.calculate_performance_metrics(returns, benchmark_returns)
            risk_metrics = self.calculate_risk_metrics(returns)

            # Create validation result
            result = FinancialValidationResult()
            result.performance_metrics = performance_metrics
            result.risk_metrics = risk_metrics

            # Check thresholds
            default_thresholds = {
                'min_sharpe_ratio': 1.0,
                'max_drawdown': 0.15,
                'min_win_rate': 0.4,
                'min_profit_factor': 1.2,
                'min_annual_return': 0.05,
                'max_volatility': 0.25
            }

            thresholds = {**default_thresholds, **thresholds}

            # Sharpe ratio check
            if performance_metrics.sharpe_ratio >= thresholds.get('min_sharpe_ratio', 1.0):
                result.passed_sharpe_threshold = True
            else:
                result.validation_errors.append(
                    f"Sharpe ratio {performance_metrics.sharpe_ratio:.3f} below threshold "
                    f"{thresholds.get('min_sharpe_ratio', 1.0)}"
                )

            # Drawdown check
            if abs(performance_metrics.max_drawdown) <= thresholds.get('max_drawdown', 0.15):
                result.passed_drawdown_threshold = True
            else:
                result.validation_errors.append(
                    f"Max drawdown {abs(performance_metrics.max_drawdown):.3f} exceeds threshold "
                    f"{thresholds.get('max_drawdown', 0.15)}"
                )

            # Win rate check
            if performance_metrics.win_rate >= thresholds.get('min_win_rate', 0.4):
                result.passed_win_rate_threshold = True
            else:
                result.validation_errors.append(
                    f"Win rate {performance_metrics.win_rate:.3f} below threshold "
                    f"{thresholds.get('min_win_rate', 0.4)}"
                )

            # Profit factor check
            if performance_metrics.profit_factor >= thresholds.get('min_profit_factor', 1.2):
                result.passed_profit_factor_threshold = True
            else:
                result.validation_errors.append(
                    f"Profit factor {performance_metrics.profit_factor:.3f} below threshold "
                    f"{thresholds.get('min_profit_factor', 1.2)}"
                )

            # Annual return check
            if performance_metrics.annualized_return >= thresholds.get('min_annual_return', 0.05):
                pass  # Good
            else:
                result.validation_warnings.append(
                    f"Annual return {performance_metrics.annualized_return:.3f} below recommended "
                    f"{thresholds.get('min_annual_return', 0.05)}"
                )

            # Volatility check
            if performance_metrics.volatility <= thresholds.get('max_volatility', 0.25):
                pass  # Good
            else:
                result.validation_warnings.append(
                    f"Volatility {performance_metrics.volatility:.3f} above recommended "
                    f"{thresholds.get('max_volatility', 0.25)}"
                )

            # Overall validation
            passed_checks = sum([
                result.passed_sharpe_threshold,
                result.passed_drawdown_threshold,
                result.passed_win_rate_threshold,
                result.passed_profit_factor_threshold
            ])

            result.validation_score = passed_checks / 4.0
            result.passed_validation = result.validation_score >= 0.75 and len(result.validation_errors) == 0

            # Generate recommendations
            if not result.passed_validation:
                if not result.passed_sharpe_threshold:
                    result.recommendations.append("Improve risk-adjusted returns by reducing volatility or increasing returns")
                if not result.passed_drawdown_threshold:
                    result.recommendations.append("Implement better risk management to reduce maximum drawdown")
                if not result.passed_win_rate_threshold:
                    result.recommendations.append("Improve signal quality to increase win rate")
                if not result.passed_profit_factor_threshold:
                    result.recommendations.append("Optimize trade management to improve profit factor")

            # Calculate validation duration
            result.validation_duration = (datetime.now() - start_time).total_seconds()

            tprint_success(f"Financial validation completed: {'PASSED' if result.passed_validation else 'FAILED'} "
                          f"(Score: {result.validation_score:.2f})")

            return result

        except Exception as e:
            tprint_error(f"Error during financial validation: {e}")
            self.logger.error(f"Error during financial validation: {e}", exc_info=True)

            result = FinancialValidationResult()
            result.validation_errors.append(f"Validation failed: {str(e)}")
            result.validation_duration = (datetime.now() - start_time).total_seconds()
            return result

    def _annualize_return(self, returns: np.ndarray) -> float:
        """Annualize return based on trading days."""
        if len(returns) == 0:
            return 0.0
        return (1 + np.mean(returns)) ** self.trading_days_per_year - 1

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(self.trading_days_per_year)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0

        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
        if downside_deviation == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0

        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        return np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year)

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0

        annual_return = self._annualize_return(returns)
        return annual_return / abs(max_dd)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / peak
        return np.max(drawdowns)

    def _calculate_max_drawdown_duration(self, returns: np.ndarray) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(returns) == 0:
            return 0

        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / peak

        max_dd = np.max(drawdowns)
        if max_dd == 0:
            return 0

        # Find the longest period at maximum drawdown
        max_dd_indices = np.where(drawdowns == max_dd)[0]
        if len(max_dd_indices) == 0:
            return 0

        return len(max_dd_indices)

    def _calculate_information_ratio(self, excess_returns: np.ndarray) -> float:
        """Calculate information ratio."""
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year)

    def _calculate_average_drawdown(self, returns: np.ndarray) -> float:
        """Calculate average drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / peak

        return np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0.0

    def _calculate_burke_denominator(self, returns: np.ndarray) -> float:
        """Calculate Burke ratio denominator (sqrt of sum of squared drawdowns)."""
        if len(returns) == 0:
            return 0.0

        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = (peak - cumulative_returns) / peak

        return np.sqrt(np.sum(drawdowns ** 2))

# Convenience functions
def calculate_trading_metrics(
    returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02
) -> TradingPerformanceMetrics:
    """Quick calculation of trading performance metrics."""
    calculator = FinancialMetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate_performance_metrics(returns, benchmark_returns)

def validate_trading_performance(
    returns: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02
) -> FinancialValidationResult:
    """Quick validation of trading performance."""
    calculator = FinancialMetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.validate_financial_performance(returns, thresholds or {}, benchmark_returns)
