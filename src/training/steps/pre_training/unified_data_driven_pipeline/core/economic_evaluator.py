"""
Enhanced Economic Evaluator for Period Selection

This module provides economic significance evaluation for period selection,
including backtesting against financial targets like Sharpe ratio, max drawdown, and win rate.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)

logger = logging.getLogger(__name__)

@dataclass
class EconomicEvaluationConfig:
    """Configuration for economic evaluation."""

    # Period configuration
    min_period: int = 1
    max_period: int = 50
    backtest_periods: int = 100
    min_backtest_periods: int = 50

    # Economic thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.15
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2

    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True

    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.02  # 2% annual

@dataclass
class PeriodBacktestResult:
    """Result from backtesting a specific period."""

    period: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float

    # Additional metrics
    avg_trade_return: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float

    # Economic score (0-1)
    economic_score: float

    # Validation
    is_valid: bool
    validation_errors: List[str]

@dataclass
class EconomicPeriodEvaluationResult:
    """Result from economic evaluation of multiple periods."""

    # Top performing periods
    top_periods: List[int]
    period_rankings: List[Tuple[int, float]]  # (period, score)

    # Backtest results
    backtest_results: Dict[int, PeriodBacktestResult]

    # Summary statistics
    successful_evaluations: int
    failed_evaluations: int
    average_sharpe: float
    average_drawdown: float
    average_win_rate: float

    # Performance metrics
    evaluation_time: float
    total_periods_evaluated: int

    # Success indicators
    success: bool
    error_message: Optional[str] = None

class EconomicPeriodEvaluator:
    """
    Evaluates periods based on economic significance through backtesting.

    This class provides comprehensive economic evaluation of periods by backtesting
    against financial targets like Sharpe ratio, max drawdown, and win rate.
    """

    def __init__(self, config: Optional[EconomicEvaluationConfig] = None):
        """
        Initialize the economic period evaluator.

        Args:
            config: Configuration for economic evaluation
        """
        self.config = config or EconomicEvaluationConfig()
        self.logger = logger

        # Performance tracking
        self.performance_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_evaluation_time': 0.0,
            'backtest_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0
        }

        tprint_info("💰 Economic Period Evaluator initialized")
        tprint_debug(f"📊 Configuration: min_sharpe={self.config.min_sharpe_ratio}, "
                    f"max_drawdown={self.config.max_drawdown_threshold}")

    def evaluate_periods(self,
                        data: pd.DataFrame,
                        candidate_periods: List[int],
                        target_timeframe: str = "15m") -> EconomicPeriodEvaluationResult:
        """
        Evaluate candidate periods for economic significance.

        Args:
            data: Input data for backtesting
            candidate_periods: List of periods to evaluate
            target_timeframe: Target timeframe for evaluation

        Returns:
            EconomicPeriodEvaluationResult with evaluation results
        """
        start_time = time.time()

        def _validate_inputs():
            if not isinstance(data, pd.DataFrame) or len(data) == 0:
                raise ValueError("Data must be a non-empty DataFrame")
            if 'close' not in data.columns:
                raise ValueError("Data must contain 'close' column")
            if not candidate_periods:
                raise ValueError("Candidate periods list cannot be empty")

        def _evaluate_periods():
            tprint_info("💰 Starting economic evaluation of periods...")
            tprint_debug(f"📊 Evaluating {len(candidate_periods)} periods: {candidate_periods}")

            backtest_results = {}
            successful_evaluations = 0
            failed_evaluations = 0

            for period in candidate_periods:
                try:
                    tprint_debug(f"🔍 Evaluating period {period}...")

                    # Backtest this period
                    backtest_result = self._backtest_period(data, period, target_timeframe)
                    backtest_results[period] = backtest_result

                    if backtest_result.is_valid:
                        successful_evaluations += 1
                        tprint_debug(f"✅ Period {period}: Sharpe={backtest_result.sharpe_ratio:.3f}, "
                                   f"DD={backtest_result.max_drawdown:.3f}, WR={backtest_result.win_rate:.3f}")
                    else:
                        failed_evaluations += 1
                        tprint_debug(f"❌ Period {period} failed validation: {backtest_result.validation_errors}")

                except Exception as e:
                    failed_evaluations += 1
                    tprint_warning(f"⚠️ Period {period} evaluation failed: {e}")
                    continue

            # Rank periods by economic score
            valid_results = {k: v for k, v in backtest_results.items() if v.is_valid}

            if not valid_results:
                tprint_warning("⚠️ No periods passed economic evaluation")
                return self._create_empty_result(start_time, "No periods passed economic evaluation")

            # Create rankings
            period_rankings = sorted(
                [(period, result.economic_score) for period, result in valid_results.items()],
                key=lambda x: x[1], reverse=True
            )

            top_periods = [period for period, _ in period_rankings]

            # Calculate summary statistics
            sharpe_ratios = [result.sharpe_ratio for result in valid_results.values()]
            drawdowns = [result.max_drawdown for result in valid_results.values()]
            win_rates = [result.win_rate for result in valid_results.values()]

            evaluation_time = time.time() - start_time

            # Update performance stats
            self.performance_stats.update({
                'total_evaluations': 1,
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': failed_evaluations,
                'total_evaluation_time': evaluation_time,
                'backtest_operations': len(candidate_periods)
            })

            tprint_success(f"✅ Economic evaluation completed in {evaluation_time:.3f}s")
            tprint_info(f"🏆 {successful_evaluations} periods passed, {failed_evaluations} failed")
            tprint_info(f"📊 Top periods: {top_periods[:5]}")

            return EconomicPeriodEvaluationResult(
                top_periods=top_periods,
                period_rankings=period_rankings,
                backtest_results=backtest_results,
                successful_evaluations=successful_evaluations,
                failed_evaluations=failed_evaluations,
                average_sharpe=np.mean(sharpe_ratios),
                average_drawdown=np.mean(drawdowns),
                average_win_rate=np.mean(win_rates),
                evaluation_time=evaluation_time,
                total_periods_evaluated=len(candidate_periods),
                success=True
            )

        # Execute with error handling
        try:
            _validate_inputs()
            return _evaluate_periods()
        except Exception as e:
            tprint_error(f"❌ Economic evaluation failed: {e}")
            return self._create_empty_result(start_time, str(e))

    def _backtest_period(self,
                        data: pd.DataFrame,
                        period: int,
                        target_timeframe: str) -> PeriodBacktestResult:
        """Backtest a specific period for economic significance."""
        try:
            # Create simple strategy based on period
            close_prices = data['close']

            # Generate signals based on period
            sma_short = close_prices.rolling(window=period).mean()
            sma_long = close_prices.rolling(window=period * 2).mean()

            # Simple crossover strategy
            signals = np.where(sma_short > sma_long, 1, -1)
            signals = pd.Series(signals, index=close_prices.index)

            # Calculate returns
            returns = close_prices.pct_change().fillna(0)
            strategy_returns = signals.shift(1) * returns

            # Remove NaN values
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) < self.config.min_backtest_periods:
                return self._create_invalid_result(period, ["Insufficient data for backtesting"])

            # Calculate financial metrics
            total_return = (1 + strategy_returns).prod() - 1
            volatility = strategy_returns.std() * np.sqrt(252)  # Annualized

            # Sharpe ratio
            excess_returns = strategy_returns - self.config.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0

            # Max drawdown
            cumulative_returns = (1 + strategy_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min())

            # Win rate
            win_rate = (strategy_returns > 0).mean()

            # Profit factor
            gross_profit = strategy_returns[strategy_returns > 0].sum()
            gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

            # Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_volatility * np.sqrt(252) if downside_volatility > 0 else 0

            # Additional metrics
            avg_trade_return = strategy_returns.mean()

            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0

            current_wins = 0
            current_losses = 0

            for ret in strategy_returns:
                if ret > 0:
                    current_wins += 1
                    current_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                elif ret < 0:
                    current_losses += 1
                    current_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)

            # Recovery factor
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0

            # Calculate economic score (0-1)
            economic_score = self._calculate_economic_score(
                sharpe_ratio, max_drawdown, win_rate, profit_factor
            )

            # Validate results
            validation_errors = []
            if sharpe_ratio < self.config.min_sharpe_ratio:
                validation_errors.append(f"Sharpe ratio {sharpe_ratio:.3f} below threshold {self.config.min_sharpe_ratio}")
            if max_drawdown > self.config.max_drawdown_threshold:
                validation_errors.append(f"Max drawdown {max_drawdown:.3f} above threshold {self.config.max_drawdown_threshold}")
            if win_rate < self.config.min_win_rate:
                validation_errors.append(f"Win rate {win_rate:.3f} below threshold {self.config.min_win_rate}")
            if profit_factor < self.config.min_profit_factor:
                validation_errors.append(f"Profit factor {profit_factor:.3f} below threshold {self.config.min_profit_factor}")

            is_valid = len(validation_errors) == 0

            return PeriodBacktestResult(
                period=period,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                avg_trade_return=avg_trade_return,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                recovery_factor=recovery_factor,
                economic_score=economic_score,
                is_valid=is_valid,
                validation_errors=validation_errors
            )

        except Exception as e:
            return self._create_invalid_result(period, [f"Backtesting error: {str(e)}"])

    def _calculate_economic_score(self,
                                 sharpe_ratio: float,
                                 max_drawdown: float,
                                 win_rate: float,
                                 profit_factor: float) -> float:
        """Calculate normalized economic score (0-1)."""
        try:
            # Normalize each metric to 0-1 scale
            sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1)  # 2.0 is excellent Sharpe
            drawdown_score = min(max(1 - max_drawdown / 0.2, 0), 1)  # 20% max drawdown is bad
            win_rate_score = min(max(win_rate, 0), 1)  # Win rate is already 0-1
            profit_factor_score = min(max(profit_factor / 3.0, 0), 1)  # 3.0 is excellent profit factor

            # Weighted combination
            economic_score = (
                0.4 * sharpe_score +
                0.3 * drawdown_score +
                0.2 * win_rate_score +
                0.1 * profit_factor_score
            )

            return min(max(economic_score, 0), 1)

        except Exception as e:
            self.logger.warning(f"Error calculating economic score: {e}")
            return 0.0

    def _create_invalid_result(self, period: int, errors: List[str]) -> PeriodBacktestResult:
        """Create an invalid backtest result."""
        return PeriodBacktestResult(
            period=period,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_return=0.0,
            volatility=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            avg_trade_return=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            recovery_factor=0.0,
            economic_score=0.0,
            is_valid=False,
            validation_errors=errors
        )

    def _create_empty_result(self, start_time: float, error_message: str) -> EconomicPeriodEvaluationResult:
        """Create empty result for failed evaluation."""
        return EconomicPeriodEvaluationResult(
            top_periods=[],
            period_rankings=[],
            backtest_results={},
            successful_evaluations=0,
            failed_evaluations=1,
            average_sharpe=0.0,
            average_drawdown=1.0,
            average_win_rate=0.0,
            evaluation_time=time.time() - start_time,
            total_periods_evaluated=0,
            success=False,
            error_message=error_message
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

# Convenience functions
def create_economic_evaluator(config: Optional[EconomicEvaluationConfig] = None) -> EconomicPeriodEvaluator:
    """Create an economic period evaluator with default configuration."""
    return EconomicPeriodEvaluator(config)

def evaluate_periods_economically(data: pd.DataFrame,
                                 candidate_periods: List[int],
                                 target_timeframe: str = "15m",
                                 config: Optional[EconomicEvaluationConfig] = None) -> EconomicPeriodEvaluationResult:
    """
    Convenience function to evaluate periods economically.

    Args:
        data: Input data for backtesting
        candidate_periods: List of periods to evaluate
        target_timeframe: Target timeframe
        config: Optional configuration

    Returns:
        EconomicPeriodEvaluationResult with evaluation results
    """
    evaluator = create_economic_evaluator(config)
    return evaluator.evaluate_periods(data, candidate_periods, target_timeframe)

# Export main classes and functions
__all__ = [
    'EconomicPeriodEvaluator',
    'EconomicEvaluationConfig',
    'EconomicPeriodEvaluationResult',
    'PeriodBacktestResult',
    'create_economic_evaluator',
    'evaluate_periods_economically'
]
