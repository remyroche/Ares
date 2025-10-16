"""
Economic Evaluation Component for UnifiedDataDrivenPipeline

This module provides sophisticated economic significance evaluation and backtesting
integrated from DataDrivenPeriodSelector for enhanced period selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import warnings

# Import math validation utilities
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, 
    validate_positive, validate_range, safe_correlation, safe_covariance,
    safe_mean, safe_std, safe_percentile, safe_percentage_change,
    safe_weighted_average, safe_kelly_calculation, MathValidation
)

# VectorBT imports for economic evaluation
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    warnings.warn("VectorBT not available for economic evaluation")

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
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

logger = logging.getLogger(__name__)


@dataclass
class EconomicEvaluationConfig:
    """Configuration for economic evaluation."""
    min_period: int = 1
    max_period: int = 50
    backtest_periods: int = 100
    min_backtest_periods: int = 50
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    min_economic_score: float = 0.4
    economic_weight: float = 0.6
    statistical_weight: float = 0.4


@dataclass
class PeriodBacktestResult:
    """Result from period backtesting."""
    period: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class EconomicPeriodEvaluationResult:
    """Result from economic period evaluation."""
    top_periods: List[int]
    period_rankings: List[Tuple[int, float]]
    period_scores: Dict[int, float]
    successful_evaluations: int
    failed_evaluations: int
    total_execution_time: float
    backtest_results: Dict[int, PeriodBacktestResult]
    success: bool
    error_message: Optional[str] = None


class EconomicPeriodEvaluator:
    """
    Economic Period Evaluator with sophisticated backtesting and economic significance evaluation.
    
    Integrates advanced economic evaluation logic from DataDrivenPeriodSelector
    with VectorBT optimization for high-performance backtesting.
    """
    
    def __init__(self, config: Optional[EconomicEvaluationConfig] = None):
        """Initialize the economic period evaluator."""
        self.config = config or EconomicEvaluationConfig()
        self.logger = logger
        
        # Performance tracking
        self.performance_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_execution_time': 0.0,
            'backtest_operations': 0,
            'vectorbt_operations': 0
        }
        
        tprint_info("💰 Economic Period Evaluator initialized")
        tprint_debug(f"📊 Configuration: {self.config}")
    
    def evaluate_periods(self, data: pd.DataFrame, candidate_periods: List[int], 
                        timeframe: str = "15m") -> EconomicPeriodEvaluationResult:
        """
        Evaluate periods for economic significance using advanced backtesting.
        
        Args:
            data: Input data with OHLCV columns
            candidate_periods: List of periods to evaluate
            timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            EconomicPeriodEvaluationResult with economic evaluation results
        """
        tprint_info(f"💰 Starting economic evaluation for {len(candidate_periods)} periods")
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(data, candidate_periods):
                return self._create_empty_result(start_time, "Invalid inputs")
            
            # Step 1: Generate trading signals for each period
            tprint_debug("Step 1: Generating trading signals")
            period_signals = self._generate_period_signals(data, candidate_periods)
            
            if not period_signals:
                return self._create_empty_result(start_time, "No valid signals generated")
            
            # Step 2: Perform backtesting for each period
            tprint_debug("Step 2: Performing backtesting")
            backtest_results = self._perform_backtesting(data, period_signals, candidate_periods)
            
            if not backtest_results:
                return self._create_empty_result(start_time, "No valid backtest results")
            
            # Step 3: Calculate economic scores
            tprint_debug("Step 3: Calculating economic scores")
            period_scores = self._calculate_economic_scores(backtest_results)
            
            # Step 4: Rank periods by economic significance
            tprint_debug("Step 4: Ranking periods by economic significance")
            period_rankings = self._rank_periods_by_economic_significance(period_scores)
            
            # Step 5: Select top periods
            top_periods = self._select_top_periods(period_rankings)
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_evaluations': 1,
                'successful_evaluations': 1,
                'total_execution_time': execution_time,
                'backtest_operations': len(backtest_results)
            })
            
            tprint_success(f"✅ Economic evaluation completed in {execution_time:.3f}s")
            tprint_info(f"🏆 Selected {len(top_periods)} economically significant periods")
            
            return EconomicPeriodEvaluationResult(
                top_periods=top_periods,
                period_rankings=period_rankings,
                period_scores=period_scores,
                successful_evaluations=len(backtest_results),
                failed_evaluations=len(candidate_periods) - len(backtest_results),
                total_execution_time=execution_time,
                backtest_results=backtest_results,
                success=True
            )
            
        except Exception as e:
            tprint_error(f"❌ Economic evaluation failed: {e}")
            return self._create_empty_result(start_time, str(e))
    
    def _validate_inputs(self, data: pd.DataFrame, candidate_periods: List[int]) -> bool:
        """Validate input data and parameters."""
        try:
            if data is None or data.empty:
                tprint_error("Data is None or empty")
                return False
            
            if 'close' not in data.columns:
                tprint_error("Data must contain 'close' column")
                return False
            
            if not candidate_periods:
                tprint_error("No candidate periods provided")
                return False
            
            if len(data) < self.config.min_backtest_periods:
                tprint_error(f"Data length {len(data)} is less than minimum required {self.config.min_backtest_periods}")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"Input validation failed: {e}")
            return False
    
    def _generate_period_signals(self, data: pd.DataFrame, candidate_periods: List[int]) -> Dict[int, pd.Series]:
        """Generate trading signals for each period using VectorBT optimization."""
        tprint_debug(f"Generating signals for {len(candidate_periods)} periods")
        
        period_signals = {}
        
        try:
            close_prices = data['close']
            
            for period in candidate_periods:
                try:
                    # Generate signals using VectorBT optimization
                    signals = self._generate_period_signal_vectorbt(close_prices, period)
                    
                    if signals is not None and not signals.isna().all():
                        period_signals[period] = signals
                        tprint_debug(f"Generated signals for period {period}")
                    else:
                        tprint_warning(f"No valid signals generated for period {period}")
                        
                except Exception as e:
                    tprint_warning(f"Signal generation failed for period {period}: {e}")
                    continue
            
            tprint_success(f"Generated signals for {len(period_signals)} periods")
            return period_signals
            
        except Exception as e:
            tprint_error(f"Signal generation failed: {e}")
            return {}
    
    def _generate_period_signal_vectorbt(self, prices: pd.Series, period: int) -> Optional[pd.Series]:
        """Generate trading signals for a specific period using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._generate_period_signal_fallback(prices, period)
            
            # VectorBT-optimized signal generation
            # Simple moving average crossover strategy
            sma_short = rolling_mean(prices, window=period)
            sma_long = rolling_mean(prices, window=period * 2)
            
            # Generate signals: 1 for long, -1 for short, 0 for neutral
            signals = pd.Series(0, index=prices.index)
            signals[sma_short > sma_long] = 1
            signals[sma_short < sma_long] = -1
            
            # Add momentum filter
            momentum = prices.pct_change(period)
            signals[momentum < 0] = 0  # No signals during negative momentum
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"VectorBT signal generation failed for period {period}: {e}")
            return self._generate_period_signal_fallback(prices, period)
    
    def _generate_period_signal_fallback(self, prices: pd.Series, period: int) -> Optional[pd.Series]:
        """Fallback signal generation when VectorBT is not available."""
        try:
            # Simple moving average crossover strategy
            sma_short = prices.rolling(window=period).mean()
            sma_long = prices.rolling(window=period * 2).mean()
            
            # Generate signals: 1 for long, -1 for short, 0 for neutral
            signals = pd.Series(0, index=prices.index)
            signals[sma_short > sma_long] = 1
            signals[sma_short < sma_long] = -1
            
            # Add momentum filter
            momentum = prices.pct_change(period)
            signals[momentum < 0] = 0  # No signals during negative momentum
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Fallback signal generation failed for period {period}: {e}")
            return None
    
    def _perform_backtesting(self, data: pd.DataFrame, period_signals: Dict[int, pd.Series], 
                           candidate_periods: List[int]) -> Dict[int, PeriodBacktestResult]:
        """Perform backtesting for each period using VectorBT optimization."""
        tprint_debug(f"Performing backtesting for {len(period_signals)} periods")
        
        backtest_results = {}
        
        try:
            close_prices = data['close']
            
            for period in candidate_periods:
                if period not in period_signals:
                    continue
                
                try:
                    # Perform backtesting using VectorBT optimization
                    result = self._backtest_period_vectorbt(
                        close_prices, period_signals[period], period
                    )
                    
                    if result.success:
                        backtest_results[period] = result
                        tprint_debug(f"Backtesting completed for period {period}")
                    else:
                        tprint_warning(f"Backtesting failed for period {period}: {result.error_message}")
                        
                except Exception as e:
                    tprint_warning(f"Backtesting failed for period {period}: {e}")
                    continue
            
            tprint_success(f"Backtesting completed for {len(backtest_results)} periods")
            return backtest_results
            
        except Exception as e:
            tprint_error(f"Backtesting failed: {e}")
            return {}
    
    def _backtest_period_vectorbt(self, prices: pd.Series, signals: pd.Series, 
                                period: int) -> PeriodBacktestResult:
        """Backtest a specific period using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._backtest_period_fallback(prices, signals, period)
            
            # VectorBT-optimized backtesting
            # Calculate returns
            returns = prices.pct_change()
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            
            # Remove NaN values
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < self.config.min_backtest_periods:
                return PeriodBacktestResult(
                    period=period,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    total_return=0.0,
                    volatility=0.0,
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    information_ratio=0.0,
                    success=False,
                    error_message="Insufficient data for backtesting"
                )
            
            # Calculate performance metrics using VectorBT
            sharpe_ratio = self._calculate_sharpe_ratio_vectorbt(strategy_returns)
            max_drawdown = self._calculate_max_drawdown_vectorbt(prices, signals)
            win_rate = self._calculate_win_rate_vectorbt(strategy_returns)
            total_return = self._calculate_total_return_vectorbt(strategy_returns)
            volatility = self._calculate_volatility_vectorbt(strategy_returns)
            calmar_ratio = self._calculate_calmar_ratio_vectorbt(total_return, max_drawdown)
            sortino_ratio = self._calculate_sortino_ratio_vectorbt(strategy_returns)
            information_ratio = self._calculate_information_ratio_vectorbt(strategy_returns, returns)
            
            return PeriodBacktestResult(
                period=period,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_return=total_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                information_ratio=information_ratio,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"VectorBT backtesting failed for period {period}: {e}")
            return PeriodBacktestResult(
                period=period,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_return=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                information_ratio=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _backtest_period_fallback(self, prices: pd.Series, signals: pd.Series, 
                                period: int) -> PeriodBacktestResult:
        """Fallback backtesting when VectorBT is not available."""
        try:
            # Calculate returns
            returns = prices.pct_change()
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            
            # Remove NaN values
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < self.config.min_backtest_periods:
                return PeriodBacktestResult(
                    period=period,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                    total_return=0.0,
                    volatility=0.0,
                    calmar_ratio=0.0,
                    sortino_ratio=0.0,
                    information_ratio=0.0,
                    success=False,
                    error_message="Insufficient data for backtesting"
                )
            
            # Calculate performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio_fallback(strategy_returns)
            max_drawdown = self._calculate_max_drawdown_fallback(prices, signals)
            win_rate = self._calculate_win_rate_fallback(strategy_returns)
            total_return = self._calculate_total_return_fallback(strategy_returns)
            volatility = self._calculate_volatility_fallback(strategy_returns)
            calmar_ratio = self._calculate_calmar_ratio_fallback(total_return, max_drawdown)
            sortino_ratio = self._calculate_sortino_ratio_fallback(strategy_returns)
            information_ratio = self._calculate_information_ratio_fallback(strategy_returns, returns)
            
            return PeriodBacktestResult(
                period=period,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_return=total_return,
                volatility=volatility,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                information_ratio=information_ratio,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Fallback backtesting failed for period {period}: {e}")
            return PeriodBacktestResult(
                period=period,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_return=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                information_ratio=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _calculate_sharpe_ratio_vectorbt(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio using VectorBT optimization with math validation."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_sharpe_ratio_fallback(returns)
            
            # Validate input data
            returns = validate_finite(returns, "returns")
            if len(returns) == 0:
                return 0.0
            
            # VectorBT-optimized Sharpe ratio calculation with safe math operations
            mean_return = safe_mean(returns.values, default=0.0)
            std_return = safe_std(returns.values, default=0.0)
            
            # Use safe division to prevent division by zero
            sharpe_ratio = safe_divide(mean_return, std_return, default=0.0)
            
            # Validate the result is finite and reasonable
            sharpe_ratio = validate_finite(sharpe_ratio, "sharpe_ratio")
            sharpe_ratio = validate_range(sharpe_ratio, -10.0, 10.0, "sharpe_ratio")
            
            return float(sharpe_ratio)
            
        except Exception as e:
            self.logger.warning(f"VectorBT Sharpe ratio calculation failed: {e}")
            return self._calculate_sharpe_ratio_fallback(returns)
    
    def _calculate_max_drawdown_vectorbt(self, prices: pd.Series, signals: pd.Series) -> float:
        """Calculate maximum drawdown using VectorBT optimization with math validation."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_max_drawdown_fallback(prices, signals)
            
            # Validate input data
            prices = validate_finite(prices, "prices")
            signals = validate_finite(signals, "signals")
            
            if len(prices) == 0 or len(signals) == 0:
                return 0.0
            
            # VectorBT-optimized drawdown calculation with safe math operations
            # Calculate cumulative returns with safe operations
            returns = prices.pct_change().fillna(0.0)
            strategy_returns = signals.shift(1).fillna(0.0) * returns
            
            # Use safe operations for cumulative product
            cumulative_returns = (1 + strategy_returns).cumprod()
            cumulative_returns = validate_finite(cumulative_returns, "cumulative_returns")
            
            # Calculate running maximum
            running_max = rolling_max(cumulative_returns, window=len(cumulative_returns))
            running_max = validate_finite(running_max, "running_max")
            
            # Calculate drawdown with safe division
            drawdown = safe_divide(
                cumulative_returns - running_max, 
                running_max, 
                default=0.0
            )
            drawdown = validate_finite(drawdown, "drawdown")
            
            # Get minimum drawdown safely
            max_drawdown = safe_percentile(drawdown.values, 0.0, default=0.0)
            max_drawdown = validate_range(max_drawdown, -1.0, 0.0, "max_drawdown")
            
            return float(max_drawdown)
            
        except Exception as e:
            self.logger.warning(f"VectorBT max drawdown calculation failed: {e}")
            return self._calculate_max_drawdown_fallback(prices, signals)
    
    def _calculate_win_rate_vectorbt(self, returns: pd.Series) -> float:
        """Calculate win rate using VectorBT optimization with math validation."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_win_rate_fallback(returns)
            
            # Validate input data
            returns = validate_finite(returns, "returns")
            if len(returns) == 0:
                return 0.0
            
            # VectorBT-optimized win rate calculation with safe operations
            positive_returns = (returns > 0).astype(int)
            win_rate = safe_mean(positive_returns.values, default=0.0)
            
            # Validate win rate is in valid range [0, 1]
            win_rate = validate_range(win_rate, 0.0, 1.0, "win_rate")
            
            return float(win_rate)
            
        except Exception as e:
            self.logger.warning(f"VectorBT win rate calculation failed: {e}")
            return self._calculate_win_rate_fallback(returns)
    
    def _calculate_total_return_vectorbt(self, returns: pd.Series) -> float:
        """Calculate total return using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_total_return_fallback(returns)
            
            # VectorBT-optimized total return calculation
            total_return = (1 + returns).prod() - 1
            
            return float(total_return)
            
        except Exception as e:
            self.logger.warning(f"VectorBT total return calculation failed: {e}")
            return self._calculate_total_return_fallback(returns)
    
    def _calculate_volatility_vectorbt(self, returns: pd.Series) -> float:
        """Calculate volatility using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_volatility_fallback(returns)
            
            # VectorBT-optimized volatility calculation
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            return float(volatility)
            
        except Exception as e:
            self.logger.warning(f"VectorBT volatility calculation failed: {e}")
            return self._calculate_volatility_fallback(returns)
    
    def _calculate_calmar_ratio_vectorbt(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio using VectorBT optimization."""
        try:
            if max_drawdown == 0:
                return 0.0
            
            calmar_ratio = total_return / abs(max_drawdown)
            return float(calmar_ratio)
            
        except Exception as e:
            self.logger.warning(f"Calmar ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_sortino_ratio_vectorbt(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_sortino_ratio_fallback(returns)
            
            # VectorBT-optimized Sortino ratio calculation
            mean_return = returns.mean()
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return 0.0
            
            downside_std = negative_returns.std()
            
            if downside_std == 0:
                return 0.0
            
            sortino_ratio = mean_return / downside_std
            return float(sortino_ratio)
            
        except Exception as e:
            self.logger.warning(f"VectorBT Sortino ratio calculation failed: {e}")
            return self._calculate_sortino_ratio_fallback(returns)
    
    def _calculate_information_ratio_vectorbt(self, strategy_returns: pd.Series, 
                                            benchmark_returns: pd.Series) -> float:
        """Calculate information ratio using VectorBT optimization."""
        try:
            if not VECTORBT_AVAILABLE:
                return self._calculate_information_ratio_fallback(strategy_returns, benchmark_returns)
            
            # VectorBT-optimized information ratio calculation
            excess_returns = strategy_returns - benchmark_returns
            mean_excess_return = excess_returns.mean()
            tracking_error = excess_returns.std()
            
            if tracking_error == 0:
                return 0.0
            
            information_ratio = mean_excess_return / tracking_error
            return float(information_ratio)
            
        except Exception as e:
            self.logger.warning(f"VectorBT information ratio calculation failed: {e}")
            return self._calculate_information_ratio_fallback(strategy_returns, benchmark_returns)
    
    def _calculate_economic_scores(self, backtest_results: Dict[int, PeriodBacktestResult]) -> Dict[int, float]:
        """Calculate economic scores for each period."""
        tprint_debug("Calculating economic scores")
        
        scores = {}
        
        try:
            for period, result in backtest_results.items():
                if not result.success:
                    scores[period] = 0.0
                    continue
                
                # Calculate composite economic score
                # Weighted combination of multiple metrics
                sharpe_weight = 0.3
                calmar_weight = 0.2
                win_rate_weight = 0.2
                information_weight = 0.2
                volatility_weight = 0.1
                
                # Normalize metrics to 0-1 range
                sharpe_score = min(max(result.sharpe_ratio / 2.0, 0), 1)  # Cap at 2.0
                calmar_score = min(max(result.calmar_ratio / 1.0, 0), 1)  # Cap at 1.0
                win_rate_score = result.win_rate
                information_score = min(max(result.information_ratio / 1.0, 0), 1)  # Cap at 1.0
                volatility_score = min(max(1 - result.volatility / 0.5, 0), 1)  # Lower volatility is better
                
                # Calculate composite score
                composite_score = (
                    sharpe_score * sharpe_weight +
                    calmar_score * calmar_weight +
                    win_rate_score * win_rate_weight +
                    information_score * information_weight +
                    volatility_score * volatility_weight
                )
                
                scores[period] = composite_score
                tprint_debug(f"Period {period}: score={composite_score:.3f}")
            
            tprint_success(f"Calculated economic scores for {len(scores)} periods")
            return scores
            
        except Exception as e:
            tprint_error(f"Economic score calculation failed: {e}")
            return {}
    
    def _rank_periods_by_economic_significance(self, period_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """Rank periods by economic significance."""
        tprint_debug("Ranking periods by economic significance")
        
        try:
            # Sort by score (descending)
            rankings = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)
            
            tprint_success(f"Ranked {len(rankings)} periods by economic significance")
            return rankings
            
        except Exception as e:
            tprint_error(f"Period ranking failed: {e}")
            return []
    
    def _select_top_periods(self, period_rankings: List[Tuple[int, float]]) -> List[int]:
        """Select top periods based on economic significance."""
        tprint_debug("Selecting top periods")
        
        try:
            # Filter by minimum economic score
            top_periods = [
                period for period, score in period_rankings
                if score >= self.config.min_economic_score
            ]
            
            # Limit to reasonable number of periods
            max_periods = 10
            if len(top_periods) > max_periods:
                top_periods = top_periods[:max_periods]
            
            tprint_success(f"Selected {len(top_periods)} top periods")
            return top_periods
            
        except Exception as e:
            tprint_error(f"Top period selection failed: {e}")
            return []
    
    def _create_empty_result(self, start_time: float, error_message: str) -> EconomicPeriodEvaluationResult:
        """Create empty result for failed evaluation."""
        return EconomicPeriodEvaluationResult(
            top_periods=[],
            period_rankings=[],
            period_scores={},
            successful_evaluations=0,
            failed_evaluations=0,
            total_execution_time=time.time() - start_time,
            backtest_results={},
            success=False,
            error_message=error_message
        )
    
    # Fallback methods for when VectorBT is not available
    def _calculate_sharpe_ratio_fallback(self, returns: pd.Series) -> float:
        """Fallback Sharpe ratio calculation with math validation."""
        try:
            # Validate input data
            returns = validate_finite(returns, "returns")
            if len(returns) == 0:
                return 0.0
            
            # Use safe math operations
            mean_return = safe_mean(returns.values, default=0.0)
            std_return = safe_std(returns.values, default=0.0)
            
            # Use safe division
            sharpe_ratio = safe_divide(mean_return, std_return, default=0.0)
            sharpe_ratio = validate_range(sharpe_ratio, -10.0, 10.0, "sharpe_ratio")
            
            return float(sharpe_ratio)
        except:
            return 0.0
    
    def _calculate_max_drawdown_fallback(self, prices: pd.Series, signals: pd.Series) -> float:
        """Fallback max drawdown calculation."""
        try:
            returns = prices.pct_change()
            strategy_returns = signals.shift(1) * returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            return float(drawdown.min())
        except:
            return 0.0
    
    def _calculate_win_rate_fallback(self, returns: pd.Series) -> float:
        """Fallback win rate calculation."""
        try:
            positive_returns = (returns > 0).astype(int)
            return float(positive_returns.mean())
        except:
            return 0.0
    
    def _calculate_total_return_fallback(self, returns: pd.Series) -> float:
        """Fallback total return calculation."""
        try:
            return float((1 + returns).prod() - 1)
        except:
            return 0.0
    
    def _calculate_volatility_fallback(self, returns: pd.Series) -> float:
        """Fallback volatility calculation."""
        try:
            return float(returns.std() * np.sqrt(252))
        except:
            return 0.0
    
    def _calculate_calmar_ratio_fallback(self, total_return: float, max_drawdown: float) -> float:
        """Fallback Calmar ratio calculation."""
        try:
            if max_drawdown == 0:
                return 0.0
            return float(total_return / abs(max_drawdown))
        except:
            return 0.0
    
    def _calculate_sortino_ratio_fallback(self, returns: pd.Series) -> float:
        """Fallback Sortino ratio calculation."""
        try:
            mean_return = returns.mean()
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return 0.0
            
            downside_std = negative_returns.std()
            
            if downside_std == 0:
                return 0.0
            
            return float(mean_return / downside_std)
        except:
            return 0.0
    
    def _calculate_information_ratio_fallback(self, strategy_returns: pd.Series, 
                                            benchmark_returns: pd.Series) -> float:
        """Fallback information ratio calculation."""
        try:
            excess_returns = strategy_returns - benchmark_returns
            mean_excess_return = excess_returns.mean()
            tracking_error = excess_returns.std()
            
            if tracking_error == 0:
                return 0.0
            
            return float(mean_excess_return / tracking_error)
        except:
            return 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'total_execution_time': 0.0,
            'backtest_operations': 0,
            'vectorbt_operations': 0
        }


def create_economic_evaluator(config: Optional[EconomicEvaluationConfig] = None) -> EconomicPeriodEvaluator:
    """Create an economic period evaluator with default configuration."""
    return EconomicPeriodEvaluator(config)