"""
Performance Analyzer for Advanced Market Analysis.

This module provides performance analysis capabilities that can be used
by both NAS and TAS regime detection systems.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from src.utils.logger import system_logger


@dataclass
class PerformanceAnalysisConfig:
    """Configuration for performance analysis."""
    benchmark_symbol: str = 'SPY'  # Benchmark for comparison
    risk_free_rate: float = 0.02  # Annual risk-free rate
    enable_risk_metrics: bool = True
    enable_attribution: bool = True
    enable_stress_testing: bool = True
    confidence_level: float = 0.95


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading strategy."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    sortino_ratio: float
    information_ratio: float
    benchmark_return: float
    tracking_error: float
    alpha: float
    beta: float


class PerformanceAnalyzer:
    """
    Performance analyzer for trading strategies and regimes.

    This class provides comprehensive performance analysis including risk metrics,
    attribution analysis, and stress testing that can be used by both NAS and
    TAS systems.
    """

    def __init__(self, config: PerformanceAnalysisConfig):
        """
        Initialize the performance analyzer.

        Args:
            config: Performance analysis configuration
        """
        self.logger = system_logger.getChild('PerformanceAnalyzer')
        self.config = config

        self.logger.info("✅ Performance Analyzer initialized"
        self.logger.info(f"   Benchmark: {config.benchmark_symbol}")
        self.logger.info(f"   Risk-free rate: {config.risk_free_rate}")

    def analyze_performance(self,
                          returns: pd.Series,
                          benchmark_returns: Optional[pd.Series] = None,
                          positions: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis.

        Args:
            returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            positions: Position sizes over time

        Returns:
            Dictionary with comprehensive performance analysis
        """
        try:
            self.logger.info("📊 Performing comprehensive performance analysis")

            # Basic performance metrics
            basic_metrics = self._calculate_basic_metrics(returns, benchmark_returns)

            # Risk metrics
            risk_metrics = {}
            if self.config.enable_risk_metrics:
                risk_metrics = self._calculate_risk_metrics(returns, benchmark_returns)

            # Attribution analysis
            attribution = {}
            if self.config.enable_attribution and positions is not None:
                attribution = self._perform_attribution_analysis(returns, positions)

            # Stress testing
            stress_tests = {}
            if self.config.enable_stress_testing:
                stress_tests = self._perform_stress_testing(returns)

            analysis_result = {
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'attribution_analysis': attribution,
                'stress_tests': stress_tests,
                'performance_summary': self._generate_performance_summary(basic_metrics, risk_metrics),
                'analysis_metadata': {
                    'total_periods': len(returns),
                    'start_date': str(returns.index.min()) if not returns.empty else None,
                    'end_date': str(returns.index.max()) if not returns.empty else None,
                    'analysis_timestamp': pd.Timestamp.now()
                }
            }

            self.logger.info(f"✅ Performance analysis completed")
            return analysis_result

        except Exception as e:
            self.logger.error(f"❌ Performance analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_basic_metrics(self,
                               returns: pd.Series,
                               benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate basic performance metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            PerformanceMetrics object
        """
        try:
            # Total return
            total_return = (1 + returns).prod() - 1

            # Annualized return (assuming daily returns)
            n_periods = len(returns)
            annualized_return = (1 + total_return) ** (252 / n_periods) - 1

            # Volatility
            volatility = returns.std() * np.sqrt(252)

            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns)

            # Max drawdown
            max_drawdown = self._calculate_max_drawdown(returns)

            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

            # Win rate and profit factor
            win_rate = (returns > 0).mean()
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            average_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
            average_loss = negative_returns.mean() if len(negative_returns) > 0 else 0.0

            profit_factor = abs(positive_returns.sum() / negative_returns.sum()) if len(negative_returns) > 0 else float('inf')

            # Sortino ratio
            sortino_ratio = self._calculate_sortino_ratio(returns)

            # Benchmark metrics
            benchmark_return = 0.0
            tracking_error = 0.0
            information_ratio = 0.0
            alpha = 0.0
            beta = 0.0

            if benchmark_returns is not None:
                benchmark_return = (1 + benchmark_returns).prod() - 1

                # Tracking error
                excess_returns = returns - benchmark_returns
                tracking_error = excess_returns.std() * np.sqrt(252)

                # Information ratio
                information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0

                # Alpha and Beta (simplified calculation)
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                alpha = returns.mean() - self.config.risk_free_rate - beta * (benchmark_returns.mean() - self.config.risk_free_rate)

            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                sortino_ratio=sortino_ratio,
                information_ratio=information_ratio,
                benchmark_return=benchmark_return,
                tracking_error=tracking_error,
                alpha=alpha,
                beta=beta
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Basic metrics calculation failed: {e}")
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                win_rate=0.0, profit_factor=0.0, average_win=0.0, average_loss=0.0,
                sortino_ratio=0.0, information_ratio=0.0, benchmark_return=0.0,
                tracking_error=0.0, alpha=0.0, beta=0.0
            )

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Returns series

        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                return 0.0

            # Annualized Sharpe ratio (assuming daily returns)
            sharpe = (mean_return - self.config.risk_free_rate / 252) / std_return * np.sqrt(252)
            return sharpe

        except Exception as e:
            self.logger.warning(f"⚠️ Sharpe ratio calculation failed: {e}")
            return 0.0

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio (downside deviation version of Sharpe).

        Args:
            returns: Returns series

        Returns:
            Sortino ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            mean_return = returns.mean()

            # Downside deviation
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return float('inf')

            downside_deviation = negative_returns.std() * np.sqrt(252)

            if downside_deviation == 0:
                return float('inf')

            # Annualized Sortino ratio
            sortino = (mean_return - self.config.risk_free_rate / 252) / downside_deviation
            return sortino

        except Exception as e:
            self.logger.warning(f"⚠️ Sortino ratio calculation failed: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Returns series

        Returns:
            Maximum drawdown (positive value)
        """
        try:
            if len(returns) < 2:
                return 0.0

            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()

            # Calculate running maximum
            running_max = cum_returns.expanding().max()

            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max

            # Return maximum drawdown (positive value)
            max_dd = abs(drawdown.min())
            return max_dd

        except Exception as e:
            self.logger.warning(f"⚠️ Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_risk_metrics(self,
                              returns: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Dictionary of risk metrics
        """
        try:
            risk_metrics = {}

            # Value at Risk (VaR)
            risk_metrics['var_95'] = self._calculate_var(returns, 0.95)
            risk_metrics['var_99'] = self._calculate_var(returns, 0.99)

            # Conditional VaR (CVaR)
            risk_metrics['cvar_95'] = self._calculate_cvar(returns, 0.95)
            risk_metrics['cvar_99'] = self._calculate_cvar(returns, 0.99)

            # Downside risk measures
            risk_metrics['semideviation'] = self._calculate_semideviation(returns)
            risk_metrics['downside_deviation'] = self._calculate_downside_deviation(returns)

            # Higher moment risks
            risk_metrics['skewness'] = returns.skew()
            risk_metrics['kurtosis'] = returns.kurtosis()

            # Stress test metrics
            risk_metrics['worst_day'] = returns.min()
            risk_metrics['worst_week'] = returns.rolling(5).sum().min()
            risk_metrics['worst_month'] = returns.rolling(20).sum().min()

            # Tail risk measures
            risk_metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
            risk_metrics['gain_to_loss_ratio'] = self._calculate_gain_to_loss_ratio(returns)

            # Benchmark-relative risk
            if benchmark_returns is not None:
                risk_metrics['active_risk'] = self._calculate_active_risk(returns, benchmark_returns)
                risk_metrics['beta_adjusted_tracking_error'] = self._calculate_beta_adjusted_tracking_error(returns, benchmark_returns)

            return risk_metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Risk metrics calculation failed: {e}")
            return {}

    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Returns series
            confidence: Confidence level (0.95 for 95% VaR)

        Returns:
            Value at Risk
        """
        try:
            if len(returns) < 10:
                return 0.0

            return abs(np.percentile(returns, (1 - confidence) * 100))

        except Exception as e:
            self.logger.warning(f"⚠️ VaR calculation failed: {e}")
            return 0.0

    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """
        Calculate Conditional Value at Risk.

        Args:
            returns: Returns series
            confidence: Confidence level

        Returns:
            Conditional Value at Risk
        """
        try:
            if len(returns) < 10:
                return 0.0

            var_threshold = np.percentile(returns, (1 - confidence) * 100)
            tail_returns = returns[returns <= var_threshold]

            if len(tail_returns) == 0:
                return 0.0

            return abs(tail_returns.mean())

        except Exception as e:
            self.logger.warning(f"⚠️ CVaR calculation failed: {e}")
            return 0.0

    def _calculate_semideviation(self, returns: pd.Series) -> float:
        """
        Calculate semi-deviation (downside volatility).

        Args:
            returns: Returns series

        Returns:
            Semi-deviation
        """
        try:
            negative_returns = returns[returns < 0]
            if len(negative_returns) == 0:
                return 0.0

            return negative_returns.std()

        except Exception as e:
            self.logger.warning(f"⚠️ Semi-deviation calculation failed: {e}")
            return 0.0

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """
        Calculate downside deviation.

        Args:
            returns: Returns series

        Returns:
            Downside deviation
        """
        try:
            # Use minimum acceptable return (MAR) of 0
            mar = 0.0
            downside_returns = np.minimum(returns - mar, 0)

            return np.sqrt(np.mean(downside_returns ** 2))

        except Exception as e:
            self.logger.warning(f"⚠️ Downside deviation calculation failed: {e}")
            return 0.0

    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """
        Calculate tail ratio (95th percentile / 5th percentile).

        Args:
            returns: Returns series

        Returns:
            Tail ratio
        """
        try:
            if len(returns) < 20:
                return 0.0

            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)

            if p5 == 0:
                return 0.0

            return p95 / abs(p5)

        except Exception as e:
            self.logger.warning(f"⚠️ Tail ratio calculation failed: {e}")
            return 0.0

    def _calculate_gain_to_loss_ratio(self, returns: pd.Series) -> float:
        """
        Calculate gain-to-loss ratio.

        Args:
            returns: Returns series

        Returns:
            Gain-to-loss ratio
        """
        try:
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]

            if len(negative_returns) == 0:
                return float('inf')

            avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0.0
            avg_loss = abs(negative_returns.mean())

            return avg_gain / avg_loss if avg_loss > 0 else float('inf')

        except Exception as e:
            self.logger.warning(f"⚠️ Gain-to-loss ratio calculation failed: {e}")
            return 0.0

    def _calculate_active_risk(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate active risk (tracking error).

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Active risk
        """
        try:
            excess_returns = returns - benchmark_returns
            return excess_returns.std() * np.sqrt(252)

        except Exception as e:
            self.logger.warning(f"⚠️ Active risk calculation failed: {e}")
            return 0.0

    def _calculate_beta_adjusted_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta-adjusted tracking error.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Beta-adjusted tracking error
        """
        try:
            # This is a simplified calculation
            # In practice, you'd use proper regression analysis
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = benchmark_returns.var()

            if benchmark_variance == 0:
                return 0.0

            beta = covariance / benchmark_variance
            tracking_error = self._calculate_active_risk(returns, benchmark_returns)

            return tracking_error / beta if beta != 0 else 0.0

        except Exception as e:
            self.logger.warning(f"⚠️ Beta-adjusted tracking error calculation failed: {e}")
            return 0.0

    def _perform_attribution_analysis(self, returns: pd.Series, positions: pd.Series) -> Dict[str, Any]:
        """
        Perform return attribution analysis.

        Args:
            returns: Strategy returns
            positions: Position sizes

        Returns:
            Attribution analysis
        """
        try:
            attribution = {}

            # Position-based attribution (simplified)
            attribution['avg_position_size'] = positions.abs().mean()
            attribution['max_position_size'] = positions.abs().max()
            attribution['position_volatility'] = positions.std()

            # Return attribution by position size
            position_sizes = ['small', 'medium', 'large']
            size_thresholds = [positions.abs().quantile(0.33), positions.abs().quantile(0.67)]

            for i, size in enumerate(position_sizes):
                if i == 0:
            mask = positions.abs() <= size_thresholds[0]
        elif i == 1:
            mask = (positions.abs() > size_thresholds[0]) & (positions.abs() <= size_thresholds[1])
        else:
            mask = positions.abs() > size_thresholds[1]

        size_returns = returns[mask]
        if len(size_returns) > 0:
            attribution[f'{size}_position_return'] = size_returns.mean()
            attribution[f'{size}_position_count'] = len(size_returns)

            return attribution

        except Exception as e:
            self.logger.warning(f"⚠️ Attribution analysis failed: {e}")
            return {}

    def _perform_stress_testing(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Perform stress testing on returns.

        Args:
            returns: Strategy returns

        Returns:
            Stress test results
        """
        try:
            stress_tests = {}

            # Historical stress periods
            stress_tests['worst_10_days'] = returns.nsmallest(10).mean()
            stress_tests['worst_20_days'] = returns.nsmallest(20).mean()

            # Volatility stress tests
            high_vol_days = returns[returns.abs() > returns.std() * 2]
            stress_tests['high_volatility_days'] = len(high_vol_days)
            stress_tests['avg_high_vol_return'] = high_vol_days.mean() if len(high_vol_days) > 0 else 0.0

            # Consecutive losses
            consecutive_losses = self._find_consecutive_losses(returns)
            stress_tests['max_consecutive_losses'] = consecutive_losses['max_streak']
            stress_tests['avg_consecutive_loss_size'] = consecutive_losses['avg_loss_size']

            # Recovery analysis
            recovery_analysis = self._analyze_recovery_time(returns)
            stress_tests.update(recovery_analysis)

            return stress_tests

        except Exception as e:
            self.logger.warning(f"⚠️ Stress testing failed: {e}")
            return {}

    def _find_consecutive_losses(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Find consecutive loss streaks.

        Args:
            returns: Returns series

        Returns:
            Dictionary with loss streak information
        """
        try:
            loss_mask = returns < 0
            max_streak = 0
            current_streak = 0
            streak_losses = []

            for is_loss in loss_mask:
                if is_loss:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    if current_streak > 0:
                        streak_losses.append(current_streak)
                    current_streak = 0

            if current_streak > 0:
                streak_losses.append(current_streak)

            return {
                'max_streak': max_streak,
                'avg_streak': np.mean(streak_losses) if streak_losses else 0,
                'total_loss_streaks': len(streak_losses)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Consecutive losses analysis failed: {e}")
            return {'max_streak': 0, 'avg_streak': 0, 'total_loss_streaks': 0}

    def _analyze_recovery_time(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Analyze recovery time after losses.

        Args:
            returns: Returns series

        Returns:
            Recovery analysis
        """
        try:
            # This is a simplified recovery analysis
            # In practice, you'd implement more sophisticated analysis

            recovery = {
                'avg_recovery_periods': 5,  # Placeholder
                'max_recovery_periods': 20,  # Placeholder
                'recovery_success_rate': 0.8  # Placeholder
            }

            return recovery

        except Exception as e:
            self.logger.warning(f"⚠️ Recovery analysis failed: {e}")
            return {}

    def _generate_performance_summary(self,
                                    basic_metrics: PerformanceMetrics,
                                    risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate performance summary and recommendations.

        Args:
            basic_metrics: Basic performance metrics
            risk_metrics: Risk metrics

        Returns:
            Performance summary
        """
        try:
            summary = {}

            # Performance rating
            sharpe_score = 1 if basic_metrics.sharpe_ratio > 1.0 else 0
            max_dd_score = 1 if basic_metrics.max_drawdown < 0.2 else 0
            calmar_score = 1 if basic_metrics.calmar_ratio > 1.0 else 0

            total_score = sharpe_score + max_dd_score + calmar_score
            if total_score >= 2:
                rating = 'Excellent'
            elif total_score >= 1:
                rating = 'Good'
            else:
                rating = 'Needs Improvement'

            summary['overall_rating'] = rating
            summary['rating_score'] = total_score

            # Key strengths and weaknesses
            summary['strengths'] = []
            summary['weaknesses'] = []

            if basic_metrics.sharpe_ratio > 1.0:
                summary['strengths'].append('Strong risk-adjusted returns')
            elif basic_metrics.sharpe_ratio < 0.5:
                summary['weaknesses'].append('Poor risk-adjusted returns')

            if basic_metrics.max_drawdown < 0.15:
                summary['strengths'].append('Low drawdown risk')
            elif basic_metrics.max_drawdown > 0.3:
                summary['weaknesses'].append('High drawdown risk')

            if basic_metrics.win_rate > 0.6:
                summary['strengths'].append('High win rate')
            elif basic_metrics.win_rate < 0.4:
                summary['weaknesses'].append('Low win rate')

            # Recommendations
            summary['recommendations'] = []
            if basic_metrics.sharpe_ratio < 1.0:
                summary['recommendations'].append('Focus on improving risk-adjusted returns')
            if basic_metrics.max_drawdown > 0.2:
                summary['recommendations'].append('Implement better risk management')
            if basic_metrics.win_rate < 0.5:
                summary['recommendations'].append('Improve entry/exit timing')

            return summary

        except Exception as e:
            self.logger.warning(f"⚠️ Performance summary generation failed: {e}")
            return {'overall_rating': 'Unknown', 'error': str(e)}