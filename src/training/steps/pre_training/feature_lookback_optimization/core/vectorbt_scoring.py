"""
VectorBT-Enhanced Scoring System for Feature Lookback Optimization.

This module provides comprehensive financial scoring using VectorBT's portfolio
analysis capabilities, replacing simple mutual information with more relevant
trading performance metrics.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from enum import Enum

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

from src.utils.tprint import tprint, tprint_error, tprint_success, tprint_warning, tprint_debug, tprint_info
from src.utils.logger import get_logger
from .utils.error_handling import safe_operation, get_error_handler

logger = get_logger('VectorBTScoring')


class ScoringMethod(Enum):
    """Available scoring methods."""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    COMPOSITE = "composite"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    MUTUAL_INFORMATION = "mutual_information"


@dataclass
class VectorBTScoringConfig:
    """Configuration for VectorBT scoring system."""
    initial_capital: float = 100000.0
    fees: float = 0.001
    slippage: float = 0.0005
    scoring_method: ScoringMethod = ScoringMethod.COMPOSITE
    risk_free_rate: float = 0.02
    lookback_period: int = 252  # Trading days in a year
    min_trades: int = 10
    max_drawdown_threshold: float = 0.2
    volatility_threshold: float = 0.3
    use_rolling_metrics: bool = True
    rolling_window: int = 50
    parallel_processing: bool = True


@dataclass
class ScoringResult:
    """Result from VectorBT scoring."""
    score: float
    method: str
    metrics: Dict[str, float]
    portfolio_stats: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    is_valid: bool = True
    error_message: Optional[str] = None


class VectorBTScoringSystem:
    """
    Comprehensive scoring system using VectorBT portfolio analysis.
    
    This class provides financial-relevant scoring for feature lookback optimization
    using VectorBT's portfolio management and performance analysis capabilities.
    """
    
    def __init__(self, config: Optional[VectorBTScoringConfig] = None):
        """Initialize VectorBT scoring system."""
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        self.config = config or VectorBTScoringConfig()
        self.logger = get_logger('VectorBTScoringSystem')
        self.error_handler = get_error_handler()
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        tprint_success("✅ VectorBT Scoring System initialized")
    
    def _configure_vectorbt(self):
        """Configure VectorBT for optimal scoring performance."""
        try:
            # Configure VectorBT settings
            vbt.settings.set_theme('dark')
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_shorten'] = True
            
            if self.config.parallel_processing:
                vbt.settings['array_wrapper']['parallel'] = True
            
            self.logger.debug("VectorBT scoring configuration applied")
            
        except Exception as e:
            self.logger.warning(f"Could not configure VectorBT settings: {e}")
    
    @safe_operation
    def score_feature_lookback(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_period: int,
        method: Optional[ScoringMethod] = None
    ) -> ScoringResult:
        """
        Score a feature lookback period using VectorBT portfolio analysis.
        
        Args:
            feature_values: Feature values array
            target_values: Target values array (returns)
            lookback_period: Lookback period being evaluated
            method: Scoring method to use
            
        Returns:
            ScoringResult with score and metrics
        """
        start_time = time.time()
        method = method or self.config.scoring_method
        
        try:
            # Validate inputs
            if not self._validate_inputs(feature_values, target_values):
                return ScoringResult(
                    score=0.0,
                    method=method.value,
                    metrics={},
                    execution_time=time.time() - start_time,
                    is_valid=False,
                    error_message="Invalid inputs"
                )
            
            # Create signals from feature values
            signals = self._create_signals_from_feature(feature_values, target_values)
            
            if signals is None or len(signals) == 0:
                return ScoringResult(
                    score=0.0,
                    method=method.value,
                    metrics={},
                    execution_time=time.time() - start_time,
                    is_valid=False,
                    error_message="Could not create signals"
                )
            
            # Create VectorBT portfolio
            portfolio = self._create_vectorbt_portfolio(signals, target_values)
            
            if portfolio is None:
                return ScoringResult(
                    score=0.0,
                    method=method.value,
                    metrics={},
                    execution_time=time.time() - start_time,
                    is_valid=False,
                    error_message="Could not create portfolio"
                )
            
            # Calculate score based on method
            score, metrics = self._calculate_score(portfolio, method)
            
            # Get portfolio statistics
            portfolio_stats = self._get_portfolio_statistics(portfolio)
            
            execution_time = time.time() - start_time
            
            return ScoringResult(
                score=score,
                method=method.value,
                metrics=metrics,
                portfolio_stats=portfolio_stats,
                execution_time=execution_time,
                is_valid=True
            )
            
        except Exception as e:
            self.logger.error(f"VectorBT scoring failed: {e}")
            return ScoringResult(
                score=0.0,
                method=method.value,
                metrics={},
                execution_time=time.time() - start_time,
                is_valid=False,
                error_message=str(e)
            )
    
    def _validate_inputs(self, feature_values: np.ndarray, target_values: np.ndarray) -> bool:
        """Validate input arrays for scoring."""
        if feature_values is None or target_values is None:
            return False
        
        if len(feature_values) == 0 or len(target_values) == 0:
            return False
        
        if len(feature_values) != len(target_values):
            return False
        
        # Check for sufficient data
        min_length = min(len(feature_values), len(target_values))
        if min_length < self.config.min_trades:
            return False
        
        # Check for valid values
        if np.all(np.isnan(feature_values)) or np.all(np.isnan(target_values)):
            return False
        
        return True
    
    def _create_signals_from_feature(
        self, 
        feature_values: np.ndarray, 
        target_values: np.ndarray
    ) -> Optional[np.ndarray]:
        """Create trading signals from feature values."""
        try:
            # Align arrays
            min_length = min(len(feature_values), len(target_values))
            feature_aligned = feature_values[:min_length]
            target_aligned = target_values[:min_length]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_aligned) | np.isnan(target_aligned))
            if not np.any(valid_mask):
                return None
            
            feature_clean = feature_aligned[valid_mask]
            target_clean = target_aligned[valid_mask]
            
            if len(feature_clean) < self.config.min_trades:
                return None
            
            # Create signals based on feature values
            # Simple strategy: buy when feature is above median, sell when below
            feature_median = np.median(feature_clean)
            signals = np.where(feature_clean > feature_median, 1, 0)
            
            # Add some sophistication: use rolling statistics
            if self.config.use_rolling_metrics and len(feature_clean) > self.config.rolling_window:
                rolling_mean = pd.Series(feature_clean).rolling(window=self.config.rolling_window).mean()
                rolling_std = pd.Series(feature_clean).rolling(window=self.config.rolling_window).std()
                
                # Z-score based signals
                z_scores = (feature_clean - rolling_mean) / (rolling_std + 1e-8)
                signals = np.where(z_scores > 1, 1, np.where(z_scores < -1, 0, signals))
            
            return signals
            
        except Exception as e:
            self.logger.warning(f"Signal creation failed: {e}")
            return None
    
    def _create_vectorbt_portfolio(
        self, 
        signals: np.ndarray, 
        returns: np.ndarray
    ) -> Optional[Portfolio]:
        """Create VectorBT portfolio from signals and returns."""
        try:
            # Align signals and returns
            min_length = min(len(signals), len(returns))
            signals_aligned = signals[:min_length]
            returns_aligned = returns[:min_length]
            
            # Create price series from returns (cumulative)
            prices = np.cumprod(1 + returns_aligned) * self.config.initial_capital
            
            # Create VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=prices,
                entries=signals_aligned == 1,
                exits=signals_aligned == 0,
                init_cash=self.config.initial_capital,
                fees=self.config.fees,
                slippage=self.config.slippage
            )
            
            return portfolio
            
        except Exception as e:
            self.logger.warning(f"Portfolio creation failed: {e}")
            return None
    
    def _calculate_score(self, portfolio: Portfolio, method: ScoringMethod) -> Tuple[float, Dict[str, float]]:
        """Calculate score based on specified method."""
        try:
            metrics = self._calculate_all_metrics(portfolio)
            
            if method == ScoringMethod.SHARPE_RATIO:
                score = metrics.get('sharpe_ratio', 0.0)
            elif method == ScoringMethod.SORTINO_RATIO:
                score = metrics.get('sortino_ratio', 0.0)
            elif method == ScoringMethod.CALMAR_RATIO:
                score = metrics.get('calmar_ratio', 0.0)
            elif method == ScoringMethod.RISK_ADJUSTED_RETURN:
                score = metrics.get('risk_adjusted_return', 0.0)
            elif method == ScoringMethod.MUTUAL_INFORMATION:
                # Fallback to correlation-based MI approximation
                score = metrics.get('correlation', 0.0) ** 2
            else:  # COMPOSITE
                score = self._calculate_composite_score(metrics)
            
            return score, metrics
            
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return 0.0, {}
    
    def _calculate_all_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """Calculate all available portfolio metrics."""
        try:
            metrics = {}
            
            # Basic returns
            returns = portfolio.returns()
            if len(returns) == 0:
                return {}
            
            # Performance metrics
            metrics['total_return'] = portfolio.total_return()
            metrics['annualized_return'] = portfolio.annualized_return()
            metrics['volatility'] = portfolio.volatility()
            metrics['sharpe_ratio'] = portfolio.sharpe_ratio()
            metrics['sortino_ratio'] = portfolio.sortino_ratio()
            metrics['calmar_ratio'] = portfolio.calmar_ratio()
            
            # Risk metrics
            metrics['max_drawdown'] = portfolio.max_drawdown()
            metrics['max_drawdown_duration'] = portfolio.max_drawdown_duration()
            metrics['var_95'] = portfolio.value_at_risk(0.05)
            metrics['cvar_95'] = portfolio.conditional_value_at_risk(0.05)
            
            # Trading metrics
            metrics['total_trades'] = portfolio.trades.count()
            metrics['win_rate'] = portfolio.trades.win_rate()
            metrics['profit_factor'] = portfolio.trades.profit_factor()
            metrics['expectancy'] = portfolio.trades.expectancy()
            
            # Additional metrics
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()
            
            # Risk-adjusted return
            if metrics['volatility'] > 0:
                metrics['risk_adjusted_return'] = metrics['annualized_return'] / metrics['volatility']
            else:
                metrics['risk_adjusted_return'] = 0.0
            
            # Correlation with market (if available)
            if hasattr(portfolio, 'benchmark_returns'):
                correlation = returns.corr(portfolio.benchmark_returns)
                metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
            else:
                metrics['correlation'] = 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Metrics calculation failed: {e}")
            return {}
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from multiple metrics."""
        try:
            # Weighted combination of key metrics
            weights = {
                'sharpe_ratio': 0.25,
                'sortino_ratio': 0.20,
                'calmar_ratio': 0.15,
                'win_rate': 0.15,
                'profit_factor': 0.10,
                'risk_adjusted_return': 0.15
            }
            
            score = 0.0
            total_weight = 0.0
            
            for metric, weight in weights.items():
                value = metrics.get(metric, 0.0)
                if not np.isnan(value) and not np.isinf(value):
                    score += value * weight
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                score = score / total_weight
            
            # Apply penalties for poor performance
            max_drawdown = metrics.get('max_drawdown', 0.0)
            if max_drawdown > self.config.max_drawdown_threshold:
                score *= (1 - max_drawdown)  # Penalty for high drawdown
            
            volatility = metrics.get('volatility', 0.0)
            if volatility > self.config.volatility_threshold:
                score *= (1 - volatility)  # Penalty for high volatility
            
            return max(0.0, score)  # Ensure non-negative
            
        except Exception as e:
            self.logger.warning(f"Composite score calculation failed: {e}")
            return 0.0
    
    def _get_portfolio_statistics(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get comprehensive portfolio statistics."""
        try:
            stats = {
                'total_trades': portfolio.trades.count(),
                'winning_trades': portfolio.trades.winning.count(),
                'losing_trades': portfolio.trades.losing.count(),
                'avg_win': portfolio.trades.winning.returns.mean(),
                'avg_loss': portfolio.trades.losing.returns.mean(),
                'largest_win': portfolio.trades.winning.returns.max(),
                'largest_loss': portfolio.trades.losing.returns.min(),
                'consecutive_wins': portfolio.trades.winning.consecutive.max(),
                'consecutive_losses': portfolio.trades.losing.consecutive.max(),
            }
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Portfolio statistics calculation failed: {e}")
            return {}
    
    def score_multiple_lookbacks(
        self,
        feature_values: np.ndarray,
        target_values: np.ndarray,
        lookback_periods: List[int],
        method: Optional[ScoringMethod] = None
    ) -> List[ScoringResult]:
        """
        Score multiple lookback periods efficiently.
        
        Args:
            feature_values: Feature values array
            target_values: Target values array
            lookback_periods: List of lookback periods to evaluate
            method: Scoring method to use
            
        Returns:
            List of ScoringResult objects
        """
        results = []
        
        for lookback in lookback_periods:
            try:
                result = self.score_feature_lookback(
                    feature_values, target_values, lookback, method
                )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Scoring failed for lookback {lookback}: {e}")
                results.append(ScoringResult(
                    score=0.0,
                    method=method.value if method else self.config.scoring_method.value,
                    metrics={},
                    is_valid=False,
                    error_message=str(e)
                ))
        
        return results


# Convenience functions
def create_vectorbt_scoring_system(
    initial_capital: float = 100000.0,
    fees: float = 0.001,
    scoring_method: ScoringMethod = ScoringMethod.COMPOSITE
) -> VectorBTScoringSystem:
    """Create a VectorBT scoring system with specified configuration."""
    config = VectorBTScoringConfig(
        initial_capital=initial_capital,
        fees=fees,
        scoring_method=scoring_method
    )
    return VectorBTScoringSystem(config)


def score_feature_with_vectorbt(
    feature_values: np.ndarray,
    target_values: np.ndarray,
    lookback_period: int,
    method: ScoringMethod = ScoringMethod.COMPOSITE
) -> ScoringResult:
    """Convenience function to score a single feature lookback."""
    scoring_system = create_vectorbt_scoring_system()
    return scoring_system.score_feature_lookback(
        feature_values, target_values, lookback_period, method
    )


# Test function
def test_vectorbt_scoring():
    """Test VectorBT scoring system."""
    if not VECTORBT_AVAILABLE:
        tprint_error("❌ VectorBT not available for testing")
        return False
    
    tprint("🧪 Testing VectorBT Scoring System...")
    
    try:
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic feature and return data
        feature_values = np.cumsum(np.random.randn(n_samples) * 0.01)
        target_values = np.random.randn(n_samples) * 0.02
        
        # Test different scoring methods
        scoring_system = create_vectorbt_scoring_system()
        
        methods_to_test = [
            ScoringMethod.SHARPE_RATIO,
            ScoringMethod.SORTINO_RATIO,
            ScoringMethod.CALMAR_RATIO,
            ScoringMethod.COMPOSITE
        ]
        
        for method in methods_to_test:
            result = scoring_system.score_feature_lookback(
                feature_values, target_values, 20, method
            )
            
            if result.is_valid:
                tprint_success(f"✅ {method.value}: {result.score:.4f}")
                tprint_info(f"📊 Execution time: {result.execution_time:.3f}s")
            else:
                tprint_warning(f"⚠️ {method.value}: {result.error_message}")
        
        # Test multiple lookbacks
        lookback_periods = [10, 20, 30, 50]
        results = scoring_system.score_multiple_lookbacks(
            feature_values, target_values, lookback_periods
        )
        
        tprint_success(f"✅ Scored {len(results)} lookback periods")
        
        # Find best lookback
        valid_results = [r for r in results if r.is_valid]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.score)
            tprint_info(f"🏆 Best lookback: {best_result.score:.4f}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ VectorBT scoring test failed: {e}")
        return False


if __name__ == "__main__":
    test_vectorbt_scoring()