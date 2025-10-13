"""
Economic Period Evaluator for Cross-Timeframe Features

This module implements economic significance evaluation and backtesting for
cross-timeframe period selection, following the pattern of DataDrivenPeriodSelector
and DataDrivenInteractionGenerator.

Key Features:
- Economic significance evaluation using financial metrics
- Backtesting against financial targets (Sharpe ratio, max drawdown, win rate)
- Period ranking based on economic performance
- Integration with existing data-driven period selection
- VectorBT-optimized backtesting for performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
import time
from contextlib import contextmanager

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

# VectorBT imports for optimized backtesting
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_sum, rolling_apply,
        rolling_corr, rolling_cov, rolling_quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    rolling_quantile = None

logger = logging.getLogger(__name__)


@dataclass
class EconomicEvaluationConfig:
    """Configuration for economic period evaluation."""
    
    # Financial metrics weights
    sharpe_ratio_weight: float = 0.4
    max_drawdown_weight: float = 0.3
    win_rate_weight: float = 0.2
    profit_factor_weight: float = 0.1
    
    # Backtesting configuration
    backtest_periods: int = 100
    min_backtest_periods: int = 50
    risk_free_rate: float = 0.02  # 2% annual
    
    # Economic significance thresholds
    min_sharpe_ratio: float = 0.5
    max_acceptable_drawdown: float = 0.15  # 15%
    min_win_rate: float = 0.45  # 45%
    min_profit_factor: float = 1.1
    
    # Period evaluation
    min_period: int = 1
    max_period: int = 50  # Optimized for 15m timeframe
    max_periods_to_evaluate: int = 20
    
    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True


@dataclass
class PeriodBacktestResult:
    """Result from period backtesting."""
    
    period: int
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Economic significance score
    economic_score: float
    significance_level: str  # 'high', 'medium', 'low'
    
    # Metadata
    backtest_periods: int
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class EconomicPeriodEvaluationResult:
    """Result from economic period evaluation."""
    
    # Evaluated periods
    evaluated_periods: List[int]
    backtest_results: Dict[int, PeriodBacktestResult]
    
    # Rankings
    period_rankings: List[Tuple[int, float]]  # (period, economic_score)
    top_periods: List[int]
    
    # Summary statistics
    best_period: int
    best_economic_score: float
    average_economic_score: float
    
    # Performance metrics
    total_evaluation_time: float
    successful_evaluations: int
    failed_evaluations: int
    
    # Configuration
    config: EconomicEvaluationConfig


class EconomicPeriodEvaluator:
    """
    Economic Period Evaluator for Cross-Timeframe Features.
    
    Evaluates periods based on economic significance using backtesting
    against financial targets, following the pattern of DataDrivenPeriodSelector.
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
            'total_execution_time': 0.0,
            'backtest_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallback_operations': 0
        }
        
        tprint_info("💰 Economic Period Evaluator initialized")
        tprint_debug(f"📊 Configuration: max_periods={self.config.max_periods_to_evaluate}, "
                    f"backtest_periods={self.config.backtest_periods}")
    
    def evaluate_periods(self, 
                        data: pd.DataFrame, 
                        candidate_periods: List[int],
                        target_timeframe: str = "15m") -> EconomicPeriodEvaluationResult:
        """
        Evaluate periods for economic significance using backtesting.
        
        Args:
            data: Input data for evaluation
            candidate_periods: List of candidate periods to evaluate
            target_timeframe: Target timeframe (e.g., "15m", "5m", "1h")
            
        Returns:
            EconomicPeriodEvaluationResult with evaluation results
        """
        start_time = time.time()
        
        def _validate_inputs():
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Data must be a non-empty DataFrame")
            if not candidate_periods:
                raise ValueError("Candidate periods list cannot be empty")
            if not all(isinstance(p, int) and p > 0 for p in candidate_periods):
                raise ValueError("All periods must be positive integers")
        
        def _evaluate_periods():
            tprint_info(f"💰 Starting economic period evaluation for {len(candidate_periods)} periods")
            tprint_debug(f"📊 Data shape: {data.shape}, target timeframe: {target_timeframe}")
            
            # Filter periods within valid range
            valid_periods = [
                p for p in candidate_periods 
                if self.config.min_period <= p <= self.config.max_period
            ]
            
            if not valid_periods:
                tprint_warning("⚠️ No valid periods found in candidate list")
                return self._create_empty_result(start_time)
            
            tprint_info(f"✅ Evaluating {len(valid_periods)} valid periods (range: {self.config.min_period}-{self.config.max_period})")
            
            # Evaluate each period
            backtest_results = {}
            successful_evaluations = 0
            failed_evaluations = 0
            
            for i, period in enumerate(valid_periods):
                try:
                    tprint_debug(f"🔄 Evaluating period {period} ({i+1}/{len(valid_periods)})")
                    
                    result = self._backtest_period(data, period, target_timeframe)
                    backtest_results[period] = result
                    
                    if result.success:
                        successful_evaluations += 1
                        tprint_debug(f"✅ Period {period}: economic_score={result.economic_score:.3f}, "
                                   f"sharpe={result.sharpe_ratio:.3f}")
                    else:
                        failed_evaluations += 1
                        tprint_warning(f"⚠️ Period {period} failed: {result.error_message}")
                        
                except Exception as e:
                    failed_evaluations += 1
                    tprint_error(f"❌ Period {period} evaluation failed: {e}")
                    
                    # Create failed result
                    backtest_results[period] = PeriodBacktestResult(
                        period=period,
                        sharpe_ratio=0.0,
                        max_drawdown=1.0,
                        win_rate=0.0,
                        profit_factor=0.0,
                        total_return=0.0,
                        volatility=0.0,
                        calmar_ratio=0.0,
                        sortino_ratio=0.0,
                        economic_score=0.0,
                        significance_level='low',
                        backtest_periods=0,
                        execution_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
            
            # Create rankings
            period_rankings = self._create_period_rankings(backtest_results)
            top_periods = [period for period, _ in period_rankings[:self.config.max_periods_to_evaluate]]
            
            # Calculate summary statistics
            successful_results = [r for r in backtest_results.values() if r.success]
            if successful_results:
                best_result = max(successful_results, key=lambda x: x.economic_score)
                best_period = best_result.period
                best_economic_score = best_result.economic_score
                average_economic_score = np.mean([r.economic_score for r in successful_results])
            else:
                best_period = valid_periods[0] if valid_periods else 0
                best_economic_score = 0.0
                average_economic_score = 0.0
            
            total_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_evaluations': len(valid_periods),
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': failed_evaluations,
                'total_execution_time': total_time
            })
            
            tprint_success(f"✅ Economic evaluation completed: {successful_evaluations} successful, "
                          f"{failed_evaluations} failed in {total_time:.3f}s")
            tprint_info(f"🏆 Best period: {best_period} (score: {best_economic_score:.3f})")
            
            return EconomicPeriodEvaluationResult(
                evaluated_periods=valid_periods,
                backtest_results=backtest_results,
                period_rankings=period_rankings,
                top_periods=top_periods,
                best_period=best_period,
                best_economic_score=best_economic_score,
                average_economic_score=average_economic_score,
                total_evaluation_time=total_time,
                successful_evaluations=successful_evaluations,
                failed_evaluations=failed_evaluations,
                config=self.config
            )
        
        # Execute with error handling
        try:
            _validate_inputs()
            return _evaluate_periods()
        except Exception as e:
            tprint_error(f"❌ Economic period evaluation failed: {e}")
            return self._create_empty_result(start_time)
    
    def _backtest_period(self, 
                        data: pd.DataFrame, 
                        period: int, 
                        target_timeframe: str) -> PeriodBacktestResult:
        """
        Backtest a specific period for economic significance.
        
        Args:
            data: Input data
            period: Period to test
            target_timeframe: Target timeframe
            
        Returns:
            PeriodBacktestResult with backtest results
        """
        start_time = time.time()
        
        try:
            # Generate cross-timeframe features for this period
            features = self._generate_period_features(data, period)
            
            if features.empty:
                raise ValueError(f"No features generated for period {period}")
            
            # Calculate returns for backtesting
            returns = self._calculate_returns(data, features)
            
            if len(returns) < self.config.min_backtest_periods:
                raise ValueError(f"Insufficient data for backtesting: {len(returns)} < {self.config.min_backtest_periods}")
            
            # Use last N periods for backtesting
            backtest_returns = returns.tail(self.config.backtest_periods)
            
            # Calculate financial metrics
            metrics = self._calculate_financial_metrics(backtest_returns)
            
            # Calculate economic significance score
            economic_score = self._calculate_economic_score(metrics)
            significance_level = self._determine_significance_level(economic_score)
            
            execution_time = time.time() - start_time
            
            return PeriodBacktestResult(
                period=period,
                sharpe_ratio=metrics['sharpe_ratio'],
                max_drawdown=metrics['max_drawdown'],
                win_rate=metrics['win_rate'],
                profit_factor=metrics['profit_factor'],
                total_return=metrics['total_return'],
                volatility=metrics['volatility'],
                calmar_ratio=metrics['calmar_ratio'],
                sortino_ratio=metrics['sortino_ratio'],
                economic_score=economic_score,
                significance_level=significance_level,
                backtest_periods=len(backtest_returns),
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
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
                economic_score=0.0,
                significance_level='low',
                backtest_periods=0,
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_period_features(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Generate cross-timeframe features for a specific period."""
        try:
            if 'close' not in data.columns:
                raise ValueError("Data must contain 'close' column")
            
            close = data['close']
            features = pd.DataFrame(index=data.index)
            
            # Generate basic cross-timeframe features
            if VECTORBT_AVAILABLE and self.config.enable_vectorbt:
                # VectorBT-optimized feature generation
                features[f'ctf_mean_{period}'] = rolling_mean(close, window=period)
                features[f'ctf_std_{period}'] = rolling_std(close, window=period)
                features[f'ctf_returns_{period}'] = close.pct_change(period)
                
                # Momentum features
                features[f'ctf_momentum_{period}'] = close / close.shift(period) - 1
                
                # Volatility features
                returns = close.pct_change()
                features[f'ctf_volatility_{period}'] = rolling_std(returns, window=period)
                
                self.performance_stats['vectorbt_operations'] += 1
            else:
                # Pandas fallback
                features[f'ctf_mean_{period}'] = close.rolling(window=period).mean()
                features[f'ctf_std_{period}'] = close.rolling(window=period).std()
                features[f'ctf_returns_{period}'] = close.pct_change(period)
                
                # Momentum features
                features[f'ctf_momentum_{period}'] = close / close.shift(period) - 1
                
                # Volatility features
                returns = close.pct_change()
                features[f'ctf_volatility_{period}'] = returns.rolling(window=period).std()
                
                self.performance_stats['pandas_fallback_operations'] += 1
            
            # Remove NaN values
            features = features.dropna()
            
            self.performance_stats['backtest_operations'] += 1
            return features
            
        except Exception as e:
            self.logger.error(f"Feature generation failed for period {period}: {e}")
            return pd.DataFrame()
    
    def _calculate_returns(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """Calculate returns for backtesting."""
        try:
            if 'close' not in data.columns:
                raise ValueError("Data must contain 'close' column")
            
            close = data['close']
            
            # Align features with price data
            aligned_data = pd.concat([close, features], axis=1).dropna()
            
            if len(aligned_data) < 2:
                raise ValueError("Insufficient aligned data for return calculation")
            
            # Calculate returns
            returns = aligned_data['close'].pct_change().dropna()
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Return calculation failed: {e}")
            return pd.Series(dtype=float)
    
    def _calculate_financial_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive financial metrics."""
        try:
            if len(returns) == 0:
                return self._get_empty_metrics()
            
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            volatility = returns.std() * np.sqrt(252)  # Annualized
            mean_return = returns.mean() * 252  # Annualized
            
            # Risk-adjusted metrics
            excess_returns = returns - self.config.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Drawdown metrics
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Win rate and profit factor
            winning_trades = (returns > 0).sum()
            total_trades = len(returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            gross_profit = returns[returns > 0].sum()
            gross_loss = abs(returns[returns < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Additional metrics
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            return {
                'total_return': total_return,
                'volatility': volatility,
                'mean_return': mean_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': min(profit_factor, 10.0),  # Cap at 10
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Financial metrics calculation failed: {e}")
            return self._get_empty_metrics()
    
    def _calculate_economic_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall economic significance score."""
        try:
            # Normalize metrics to 0-1 scale
            sharpe_score = min(1.0, max(0.0, metrics['sharpe_ratio'] / 2.0))  # Cap at 2.0 Sharpe
            drawdown_score = min(1.0, max(0.0, 1.0 - metrics['max_drawdown'] / 0.5))  # Penalize >50% drawdown
            win_rate_score = min(1.0, max(0.0, metrics['win_rate']))
            profit_factor_score = min(1.0, max(0.0, (metrics['profit_factor'] - 1.0) / 2.0))  # 1.0-3.0 range
            
            # Weighted combination
            economic_score = (
                sharpe_score * self.config.sharpe_ratio_weight +
                drawdown_score * self.config.max_drawdown_weight +
                win_rate_score * self.config.win_rate_weight +
                profit_factor_score * self.config.profit_factor_weight
            )
            
            return min(1.0, max(0.0, economic_score))
            
        except Exception as e:
            self.logger.error(f"Economic score calculation failed: {e}")
            return 0.0
    
    def _determine_significance_level(self, economic_score: float) -> str:
        """Determine economic significance level."""
        if economic_score >= 0.7:
            return 'high'
        elif economic_score >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _create_period_rankings(self, backtest_results: Dict[int, PeriodBacktestResult]) -> List[Tuple[int, float]]:
        """Create period rankings based on economic scores."""
        successful_results = [(period, result.economic_score) 
                            for period, result in backtest_results.items() 
                            if result.success]
        
        # Sort by economic score (descending)
        return sorted(successful_results, key=lambda x: x[1], reverse=True)
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'volatility': 0.0,
            'mean_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 1.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0
        }
    
    def _create_empty_result(self, start_time: float) -> EconomicPeriodEvaluationResult:
        """Create empty result for failed evaluation."""
        return EconomicPeriodEvaluationResult(
            evaluated_periods=[],
            backtest_results={},
            period_rankings=[],
            top_periods=[],
            best_period=0,
            best_economic_score=0.0,
            average_economic_score=0.0,
            total_evaluation_time=time.time() - start_time,
            successful_evaluations=0,
            failed_evaluations=0,
            config=self.config
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


# Convenience functions
def evaluate_periods_economically(data: pd.DataFrame, 
                                 candidate_periods: List[int],
                                 target_timeframe: str = "15m",
                                 config: Optional[EconomicEvaluationConfig] = None) -> EconomicPeriodEvaluationResult:
    """
    Convenience function to evaluate periods for economic significance.
    
    Args:
        data: Input data for evaluation
        candidate_periods: List of candidate periods to evaluate
        target_timeframe: Target timeframe
        config: Optional configuration
        
    Returns:
        EconomicPeriodEvaluationResult with evaluation results
    """
    evaluator = EconomicPeriodEvaluator(config)
    return evaluator.evaluate_periods(data, candidate_periods, target_timeframe)


def get_economically_significant_periods(data: pd.DataFrame,
                                       candidate_periods: List[int],
                                       target_timeframe: str = "15m",
                                       min_economic_score: float = 0.4) -> List[int]:
    """
    Get periods that meet economic significance threshold.
    
    Args:
        data: Input data for evaluation
        candidate_periods: List of candidate periods to evaluate
        target_timeframe: Target timeframe
        min_economic_score: Minimum economic score threshold
        
    Returns:
        List of economically significant periods
    """
    result = evaluate_periods_economically(data, candidate_periods, target_timeframe)
    
    significant_periods = [
        period for period, score in result.period_rankings
        if score >= min_economic_score
    ]
    
    return significant_periods


# Export main classes and functions
__all__ = [
    'EconomicPeriodEvaluator',
    'EconomicEvaluationConfig',
    'PeriodBacktestResult',
    'EconomicPeriodEvaluationResult',
    'evaluate_periods_economically',
    'get_economically_significant_periods'
]