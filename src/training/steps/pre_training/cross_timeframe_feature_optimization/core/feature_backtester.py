"""
Feature Backtester for Cross-Timeframe Features

This module implements backtesting for cross-timeframe features to evaluate
their economic significance, following the pattern of FeatureLookbackOptimizationComponent.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import logging

# Import optimization utilities
from src.utils.math_validation import safe_divide, validate_finite
from src.utils.common_operations import safe_dataframe_operation
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for feature backtesting."""
    
    # Backtesting parameters
    backtest_periods: int = 100
    min_backtest_periods: int = 50
    risk_free_rate: float = 0.02  # 2% annual
    
    # Performance optimization
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Financial metrics thresholds
    min_sharpe_ratio: float = 0.5
    max_acceptable_drawdown: float = 0.15  # 15%
    min_win_rate: float = 0.45  # 45%
    min_profit_factor: float = 1.1
    
    # Feature evaluation
    min_correlation: float = 0.01
    max_correlation: float = 0.95


@dataclass
class BacktestResult:
    """Result from feature backtesting."""
    
    # Feature information
    feature_name: str
    success: bool
    
    # Financial metrics
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Feature metrics
    correlation_with_target: float
    feature_volatility: float
    feature_stability: float
    
    # Economic significance
    economic_score: float
    significance_level: str  # 'high', 'medium', 'low'
    
    # Performance metrics
    execution_time: float
    backtest_periods: int
    
    # Error information
    error_message: Optional[str] = None


class FeatureBacktester:
    """
    Feature Backtester for Cross-Timeframe Features.
    
    Backtests features to evaluate their economic significance using
    financial metrics and performance indicators.
    """
    
    def __init__(self, config: BacktestConfig):
        """Initialize the feature backtester."""
        self.config = config
        self.logger = logger
        
        # Initialize VectorBT optimizer if available
        self.vectorbt_optimizer = None
        if VECTORBT_AVAILABLE and config.enable_vectorbt:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
                tprint("✅ VectorBT backtester initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT backtester initialization failed: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'pandas_operations': 0
        }
        
        tprint_info("🔧 FeatureBacktester initialized")
    
    def backtest_feature(self, 
                        price_data: pd.Series, 
                        feature_data: pd.Series,
                        feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Backtest a single feature for economic significance.
        
        Args:
            price_data: Price data (close prices)
            feature_data: Feature data
            feature_name: Name of the feature
            
        Returns:
            Dictionary with backtest results
        """
        start_time = time.time()
        
        try:
            tprint_debug(f"💰 Backtesting feature: {feature_name}")
            
            # Validate inputs
            if price_data.empty or feature_data.empty:
                tprint_warning(f"⚠️ Empty data for {feature_name}")
                return None
            
            # Align data
            common_index = price_data.index.intersection(feature_data.index)
            if len(common_index) < self.config.min_backtest_periods:
                tprint_warning(f"⚠️ Insufficient data for {feature_name}: {len(common_index)} < {self.config.min_backtest_periods}")
                return None
            
            aligned_price = price_data.loc[common_index]
            aligned_feature = feature_data.loc[common_index]
            
            # Use last N periods for backtesting
            backtest_data = pd.DataFrame({
                'price': aligned_price,
                'feature': aligned_feature
            }).tail(self.config.backtest_periods)
            
            if len(backtest_data) < self.config.min_backtest_periods:
                tprint_warning(f"⚠️ Insufficient backtest data for {feature_name}")
                return None
            
            # Calculate returns
            returns = backtest_data['price'].pct_change().dropna()
            
            if len(returns) < 10:  # Need minimum data points
                tprint_warning(f"⚠️ Insufficient returns data for {feature_name}")
                return None
            
            # Calculate financial metrics
            financial_metrics = self._calculate_financial_metrics(returns)
            
            # Calculate feature metrics
            feature_metrics = self._calculate_feature_metrics(
                backtest_data['price'], 
                backtest_data['feature']
            )
            
            # Calculate economic significance score
            economic_score = self._calculate_economic_score(financial_metrics, feature_metrics)
            significance_level = self._determine_significance_level(economic_score)
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self.performance_stats.update({
                'total_backtests': 1,
                'successful_backtests': 1,
                'total_execution_time': execution_time
            })
            
            result = {
                'success': True,
                'feature_name': feature_name,
                'sharpe_ratio': financial_metrics['sharpe_ratio'],
                'max_drawdown': financial_metrics['max_drawdown'],
                'win_rate': financial_metrics['win_rate'],
                'profit_factor': financial_metrics['profit_factor'],
                'total_return': financial_metrics['total_return'],
                'volatility': financial_metrics['volatility'],
                'calmar_ratio': financial_metrics['calmar_ratio'],
                'sortino_ratio': financial_metrics['sortino_ratio'],
                'correlation_with_target': feature_metrics['correlation_with_target'],
                'feature_volatility': feature_metrics['feature_volatility'],
                'feature_stability': feature_metrics['feature_stability'],
                'economic_score': economic_score,
                'significance_level': significance_level,
                'execution_time': execution_time,
                'backtest_periods': len(backtest_data)
            }
            
            tprint_debug(f"✅ Backtest completed for {feature_name}: score={economic_score:.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            tprint_error(f"❌ Backtest failed for {feature_name}: {e}")
            
            self.performance_stats.update({
                'total_backtests': 1,
                'failed_backtests': 1,
                'total_execution_time': execution_time
            })
            
            return {
                'success': False,
                'feature_name': feature_name,
                'error_message': str(e),
                'execution_time': execution_time
            }
    
    def _calculate_financial_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive financial metrics."""
        try:
            if len(returns) == 0:
                return self._get_empty_financial_metrics()
            
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
            return self._get_empty_financial_metrics()
    
    def _calculate_feature_metrics(self, price_data: pd.Series, feature_data: pd.Series) -> Dict[str, float]:
        """Calculate feature-specific metrics."""
        try:
            # Correlation with target
            correlation = price_data.corr(feature_data)
            correlation = correlation if not np.isnan(correlation) else 0.0
            
            # Feature volatility
            feature_volatility = feature_data.std()
            feature_volatility = feature_volatility if not np.isnan(feature_volatility) else 0.0
            
            # Feature stability (inverse of coefficient of variation)
            feature_mean = feature_data.mean()
            feature_stability = 1.0 / (feature_volatility / abs(feature_mean) + 1e-8) if feature_mean != 0 else 0.0
            feature_stability = min(feature_stability, 10.0)  # Cap at 10
            
            return {
                'correlation_with_target': correlation,
                'feature_volatility': feature_volatility,
                'feature_stability': feature_stability
            }
            
        except Exception as e:
            self.logger.error(f"Feature metrics calculation failed: {e}")
            return {
                'correlation_with_target': 0.0,
                'feature_volatility': 0.0,
                'feature_stability': 0.0
            }
    
    def _calculate_economic_score(self, 
                                 financial_metrics: Dict[str, float], 
                                 feature_metrics: Dict[str, float]) -> float:
        """Calculate overall economic significance score."""
        try:
            # Normalize financial metrics to 0-1 scale
            sharpe_score = min(1.0, max(0.0, financial_metrics['sharpe_ratio'] / 2.0))  # Cap at 2.0 Sharpe
            drawdown_score = min(1.0, max(0.0, 1.0 - financial_metrics['max_drawdown'] / 0.5))  # Penalize >50% drawdown
            win_rate_score = min(1.0, max(0.0, financial_metrics['win_rate']))
            profit_factor_score = min(1.0, max(0.0, (financial_metrics['profit_factor'] - 1.0) / 2.0))  # 1.0-3.0 range
            
            # Normalize feature metrics to 0-1 scale
            correlation_score = min(1.0, max(0.0, abs(feature_metrics['correlation_with_target'])))
            stability_score = min(1.0, max(0.0, feature_metrics['feature_stability'] / 5.0))  # Cap at 5.0 stability
            
            # Weighted combination
            economic_score = (
                sharpe_score * 0.25 +
                drawdown_score * 0.20 +
                win_rate_score * 0.15 +
                profit_factor_score * 0.10 +
                correlation_score * 0.20 +
                stability_score * 0.10
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
    
    def _get_empty_financial_metrics(self) -> Dict[str, float]:
        """Return empty financial metrics dictionary."""
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
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


# Export main classes
__all__ = [
    'FeatureBacktester',
    'BacktestConfig',
    'BacktestResult'
]