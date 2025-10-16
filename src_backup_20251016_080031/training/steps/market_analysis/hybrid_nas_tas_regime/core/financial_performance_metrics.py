"""
Financial-Specific Performance Metrics

This module provides comprehensive financial performance metrics for evaluating
trading strategies, including risk-adjusted returns, drawdown analysis, and
regime-aware performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import warnings

from .financial_architecture_primitives import RegimeType, FinancialActivationType

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

logger = logging.getLogger(__name__)

class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    RETURN_METRICS = "return_metrics"
    RISK_METRICS = "risk_metrics"
    RISK_ADJUSTED_METRICS = "risk_adjusted_metrics"
    DRAWDOWN_METRICS = "drawdown_metrics"
    REGIME_METRICS = "regime_metrics"
    TRADING_METRICS = "trading_metrics"
    VOLATILITY_METRICS = "volatility_metrics"
    MOMENTUM_METRICS = "momentum_metrics"

class RegimePerformanceMode(Enum):
    """Modes for regime performance evaluation."""
    REGIME_SPECIFIC = "regime_specific"
    REGIME_WEIGHTED = "regime_weighted"
    REGIME_ADAPTIVE = "regime_adaptive"
    REGIME_ENSEMBLE = "regime_ensemble"

@dataclass
class FinancialPerformanceConfig:
    """Configuration for financial performance metrics."""
    # Base metrics
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    lookback_period: int = 252  # Trading days in a year
    
    # Return metrics
    enable_return_metrics: bool = True
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    
    # Risk metrics
    enable_risk_metrics: bool = True
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    cvar_confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    
    # Risk-adjusted metrics
    enable_risk_adjusted_metrics: bool = True
    sharpe_ratios: List[int] = field(default_factory=lambda: [1, 3, 6, 12])  # Months
    sortino_ratios: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    calmar_ratios: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    
    # Drawdown metrics
    enable_drawdown_metrics: bool = True
    drawdown_periods: List[int] = field(default_factory=lambda: [1, 3, 6, 12])
    
    # Regime metrics
    enable_regime_metrics: bool = True
    regime_performance_mode: RegimePerformanceMode = RegimePerformanceMode.REGIME_WEIGHTED
    regime_weights: Dict[RegimeType, float] = field(default_factory=lambda: {
        RegimeType.BULL: 1.0,
        RegimeType.BEAR: 0.8,
        RegimeType.SIDEWAYS: 0.9,
        RegimeType.HIGH_VOLATILITY: 0.7,
        RegimeType.LOW_VOLATILITY: 1.1,
        RegimeType.TRENDING: 1.0,
        RegimeType.MEAN_REVERTING: 0.9
    })
    
    # Trading metrics
    enable_trading_metrics: bool = True
    transaction_costs: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    
    # Volatility metrics
    enable_volatility_metrics: bool = True
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    
    # Momentum metrics
    enable_momentum_metrics: bool = True
    momentum_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Performance thresholds
    min_performance_samples: int = 10
    max_performance_samples: int = 10000
    
    # Regime analysis
    enable_regime_analysis: bool = True
    regime_stability_threshold: float = 0.7
    regime_transition_threshold: float = 0.3

@dataclass
class FinancialPerformanceResult:
    """Result from financial performance evaluation."""
    return_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    risk_adjusted_metrics: Dict[str, float]
    drawdown_metrics: Dict[str, float]
    regime_metrics: Dict[str, float]
    trading_metrics: Dict[str, float]
    volatility_metrics: Dict[str, float]
    momentum_metrics: Dict[str, float]
    regime_analysis: Dict[str, Any]
    performance_summary: Dict[str, Any]
    execution_time: float
    n_samples: int

class FinancialPerformanceEvaluator:
    """Evaluates financial performance metrics."""
    
    def __init__(self, config: FinancialPerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.returns_history = []
        self.regime_history = []
        self.performance_history = []
        
        # Regime analysis
        self.regime_performance = {}
        self.regime_transitions = []
        
        self.logger.info("✅ Financial Performance Evaluator initialized")
        self.logger.info(f"   Risk-free Rate: {config.risk_free_rate}")
        self.logger.info(f"   Lookback Period: {config.lookback_period}")
        self.logger.info(f"   Regime Performance Mode: {config.regime_performance_mode.value}")
    
    def evaluate_performance(self, returns: np.ndarray, 
                           regime_labels: Optional[np.ndarray] = None,
                           prices: Optional[np.ndarray] = None,
                           volumes: Optional[np.ndarray] = None) -> FinancialPerformanceResult:
        """Evaluate comprehensive financial performance."""
        start_time = time.time()
        self.logger.info("📊 Evaluating financial performance...")
        
        try:
            # Validate inputs
            if len(returns) < self.config.min_performance_samples:
                raise ValueError(f"Insufficient data: {len(returns)} samples < {self.config.min_performance_samples}")
            
            # Calculate return metrics
            return_metrics = self._calculate_return_metrics(returns)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            
            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(returns)
            
            # Calculate drawdown metrics
            drawdown_metrics = self._calculate_drawdown_metrics(returns)
            
            # Calculate regime metrics
            regime_metrics = self._calculate_regime_metrics(returns, regime_labels)
            
            # Calculate trading metrics
            trading_metrics = self._calculate_trading_metrics(returns, prices, volumes)
            
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(returns)
            
            # Calculate momentum metrics
            momentum_metrics = self._calculate_momentum_metrics(returns)
            
            # Analyze regimes
            regime_analysis = self._analyze_regimes(returns, regime_labels)
            
            # Create performance summary
            performance_summary = self._create_performance_summary(
                return_metrics, risk_metrics, risk_adjusted_metrics, 
                drawdown_metrics, regime_metrics
            )
            
            execution_time = time.time() - start_time
            
            return FinancialPerformanceResult(
                return_metrics=return_metrics,
                risk_metrics=risk_metrics,
                risk_adjusted_metrics=risk_adjusted_metrics,
                drawdown_metrics=drawdown_metrics,
                regime_metrics=regime_metrics,
                trading_metrics=trading_metrics,
                volatility_metrics=volatility_metrics,
                momentum_metrics=momentum_metrics,
                regime_analysis=regime_analysis,
                performance_summary=performance_summary,
                execution_time=execution_time,
                n_samples=len(returns)
            )
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _calculate_return_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate return-based metrics."""
        if not self.config.enable_return_metrics:
            return {}
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = np.sum(returns)
        metrics['mean_return'] = np.mean(returns)
        metrics['median_return'] = np.median(returns)
        metrics['std_return'] = np.std(returns)
        metrics['min_return'] = np.min(returns)
        metrics['max_return'] = np.max(returns)
        
        # Annualized metrics
        if len(returns) > 0:
            metrics['annualized_return'] = np.mean(returns) * self.config.lookback_period
            metrics['annualized_volatility'] = np.std(returns) * np.sqrt(self.config.lookback_period)
        
        # Return percentiles
        metrics['return_5th_percentile'] = np.percentile(returns, 5)
        metrics['return_25th_percentile'] = np.percentile(returns, 25)
        metrics['return_75th_percentile'] = np.percentile(returns, 75)
        metrics['return_95th_percentile'] = np.percentile(returns, 95)
        
        # Period-specific returns
        for period in self.config.return_periods:
            if len(returns) >= period:
                period_returns = returns[-period:]
                metrics[f'return_{period}d'] = np.sum(period_returns)
                metrics[f'mean_return_{period}d'] = np.mean(period_returns)
                metrics[f'std_return_{period}d'] = np.std(period_returns)
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-based metrics."""
        if not self.config.enable_risk_metrics:
            return {}
        
        metrics = {}
        
        # Basic risk metrics
        metrics['volatility'] = np.std(returns)
        metrics['variance'] = np.var(returns)
        metrics['skewness'] = self._calculate_skewness(returns)
        metrics['kurtosis'] = self._calculate_kurtosis(returns)
        
        # Value at Risk (VaR)
        for confidence in self.config.var_confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100)
            metrics[f'var_{int(confidence*100)}'] = var
        
        # Conditional Value at Risk (CVaR)
        for confidence in self.config.cvar_confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100)
            cvar = np.mean(returns[returns <= var])
            metrics[f'cvar_{int(confidence*100)}'] = cvar
        
        # Downside risk
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['downside_volatility'] = np.std(negative_returns)
            metrics['downside_variance'] = np.var(negative_returns)
        else:
            metrics['downside_volatility'] = 0.0
            metrics['downside_variance'] = 0.0
        
        # Tail risk
        metrics['tail_risk'] = self._calculate_tail_risk(returns)
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate risk-adjusted metrics."""
        if not self.config.enable_risk_adjusted_metrics:
            return {}
        
        metrics = {}
        
        # Sharpe ratio
        for months in self.config.sharpe_ratios:
            if len(returns) >= months * 21:  # Approximate trading days per month
                period_returns = returns[-months * 21:]
                sharpe = (np.mean(period_returns) - self.config.risk_free_rate / self.config.lookback_period) / (np.std(period_returns) + 1e-8)
                metrics[f'sharpe_ratio_{months}m'] = sharpe
        
        # Sortino ratio
        for months in self.config.sortino_ratios:
            if len(returns) >= months * 21:
                period_returns = returns[-months * 21:]
                downside_returns = period_returns[period_returns < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    sortino = (np.mean(period_returns) - self.config.risk_free_rate / self.config.lookback_period) / (downside_std + 1e-8)
                else:
                    sortino = np.inf
                metrics[f'sortino_ratio_{months}m'] = sortino
        
        # Calmar ratio
        for months in self.config.calmar_ratios:
            if len(returns) >= months * 21:
                period_returns = returns[-months * 21:]
                cumulative_returns = np.cumprod(1 + period_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (running_max - cumulative_returns) / running_max
                max_drawdown = np.max(drawdown)
                
                if max_drawdown > 0:
                    calmar = (np.mean(period_returns) - self.config.risk_free_rate / self.config.lookback_period) / max_drawdown
                else:
                    calmar = np.inf
                metrics[f'calmar_ratio_{months}m'] = calmar
        
        # Information ratio
        if len(returns) > 0:
            excess_returns = returns - self.config.risk_free_rate / self.config.lookback_period
            tracking_error = np.std(excess_returns)
            if tracking_error > 0:
                information_ratio = np.mean(excess_returns) / tracking_error
            else:
                information_ratio = 0.0
            metrics['information_ratio'] = information_ratio
        
        return metrics
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate drawdown-based metrics."""
        if not self.config.enable_drawdown_metrics:
            return {}
        
        metrics = {}
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (running_max - cumulative_returns) / running_max
        
        # Basic drawdown metrics
        metrics['max_drawdown'] = np.max(drawdown)
        metrics['mean_drawdown'] = np.mean(drawdown)
        metrics['std_drawdown'] = np.std(drawdown)
        
        # Drawdown duration
        drawdown_duration = self._calculate_drawdown_duration(drawdown)
        metrics['max_drawdown_duration'] = np.max(drawdown_duration)
        metrics['mean_drawdown_duration'] = np.mean(drawdown_duration)
        
        # Recovery factor
        if metrics['max_drawdown'] > 0:
            metrics['recovery_factor'] = np.sum(returns) / metrics['max_drawdown']
        else:
            metrics['recovery_factor'] = np.inf
        
        # Period-specific drawdowns
        for period in self.config.drawdown_periods:
            if len(returns) >= period:
                period_returns = returns[-period:]
                period_cumulative = np.cumprod(1 + period_returns)
                period_running_max = np.maximum.accumulate(period_cumulative)
                period_drawdown = (period_running_max - period_cumulative) / period_running_max
                metrics[f'max_drawdown_{period}d'] = np.max(period_drawdown)
        
        return metrics
    
    def _calculate_regime_metrics(self, returns: np.ndarray, 
                                regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate regime-based metrics."""
        if not self.config.enable_regime_metrics or regime_labels is None:
            return {}
        
        metrics = {}
        
        # Regime-specific performance
        unique_regimes = np.unique(regime_labels)
        
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                # Basic regime metrics
                metrics[f'regime_{regime}_mean_return'] = np.mean(regime_returns)
                metrics[f'regime_{regime}_std_return'] = np.std(regime_returns)
                metrics[f'regime_{regime}_sharpe_ratio'] = np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)
                metrics[f'regime_{regime}_count'] = len(regime_returns)
        
        # Regime-weighted performance
        if self.config.regime_performance_mode == RegimePerformanceMode.REGIME_WEIGHTED:
            weighted_returns = self._calculate_weighted_returns(returns, regime_labels)
            metrics['weighted_mean_return'] = np.mean(weighted_returns)
            metrics['weighted_std_return'] = np.std(weighted_returns)
            metrics['weighted_sharpe_ratio'] = np.mean(weighted_returns) / (np.std(weighted_returns) + 1e-8)
        
        # Regime stability
        regime_stability = self._calculate_regime_stability(regime_labels)
        metrics['regime_stability'] = regime_stability
        
        # Regime transition frequency
        regime_transitions = np.sum(np.diff(regime_labels) != 0)
        metrics['regime_transition_frequency'] = regime_transitions / len(regime_labels)
        
        return metrics
    
    def _calculate_trading_metrics(self, returns: np.ndarray, 
                                 prices: Optional[np.ndarray] = None,
                                 volumes: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate trading-based metrics."""
        if not self.config.enable_trading_metrics:
            return {}
        
        metrics = {}
        
        # Win rate
        positive_returns = returns[returns > 0]
        metrics['win_rate'] = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
        
        # Profit factor
        total_profit = np.sum(positive_returns)
        total_loss = np.sum(returns[returns < 0])
        if total_loss != 0:
            metrics['profit_factor'] = total_profit / abs(total_loss)
        else:
            metrics['profit_factor'] = np.inf if total_profit > 0 else 0.0
        
        # Average win/loss
        if len(positive_returns) > 0:
            metrics['average_win'] = np.mean(positive_returns)
        else:
            metrics['average_win'] = 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['average_loss'] = np.mean(negative_returns)
        else:
            metrics['average_loss'] = 0.0
        
        # Win/loss ratio
        if metrics['average_loss'] != 0:
            metrics['win_loss_ratio'] = metrics['average_win'] / abs(metrics['average_loss'])
        else:
            metrics['win_loss_ratio'] = np.inf if metrics['average_win'] > 0 else 0.0
        
        # Transaction costs (simplified)
        n_trades = len(returns)  # Simplified: assume one trade per period
        total_costs = n_trades * self.config.transaction_costs
        metrics['total_transaction_costs'] = total_costs
        metrics['net_return'] = np.sum(returns) - total_costs
        
        return metrics
    
    def _calculate_volatility_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate volatility-based metrics."""
        if not self.config.enable_volatility_metrics:
            return {}
        
        metrics = {}
        
        # Rolling volatility
        for window in self.config.volatility_windows:
            if len(returns) >= window:
                rolling_vol = pd.Series(returns).rolling(window=window).std().values
                metrics[f'volatility_{window}d_mean'] = np.mean(rolling_vol[~np.isnan(rolling_vol)])
                metrics[f'volatility_{window}d_std'] = np.std(rolling_vol[~np.isnan(rolling_vol)])
                metrics[f'volatility_{window}d_max'] = np.max(rolling_vol[~np.isnan(rolling_vol)])
        
        # Volatility of volatility
        if len(returns) >= 20:
            rolling_vol = pd.Series(returns).rolling(window=20).std().values
            vol_of_vol = np.std(rolling_vol[~np.isnan(rolling_vol)])
            metrics['volatility_of_volatility'] = vol_of_vol
        
        # Volatility clustering
        volatility_clustering = self._calculate_volatility_clustering(returns)
        metrics['volatility_clustering'] = volatility_clustering
        
        return metrics
    
    def _calculate_momentum_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate momentum-based metrics."""
        if not self.config.enable_momentum_metrics:
            return {}
        
        metrics = {}
        
        # Momentum indicators
        for period in self.config.momentum_periods:
            if len(returns) >= period:
                period_returns = returns[-period:]
                metrics[f'momentum_{period}d'] = np.sum(period_returns)
                metrics[f'momentum_{period}d_mean'] = np.mean(period_returns)
        
        # Momentum persistence
        momentum_persistence = self._calculate_momentum_persistence(returns)
        metrics['momentum_persistence'] = momentum_persistence
        
        # Momentum reversal
        momentum_reversal = self._calculate_momentum_reversal(returns)
        metrics['momentum_reversal'] = momentum_reversal
        
        return metrics
    
    def _analyze_regimes(self, returns: np.ndarray, 
                        regime_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze regime characteristics."""
        if not self.config.enable_regime_analysis or regime_labels is None:
            return {}
        
        analysis = {}
        
        # Regime distribution
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        regime_distribution = dict(zip(unique_regimes, counts))
        analysis['regime_distribution'] = regime_distribution
        
        # Regime performance
        regime_performance = {}
        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            regime_returns = returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_performance[regime] = {
                    'mean_return': np.mean(regime_returns),
                    'std_return': np.std(regime_returns),
                    'sharpe_ratio': np.mean(regime_returns) / (np.std(regime_returns) + 1e-8),
                    'count': len(regime_returns)
                }
        
        analysis['regime_performance'] = regime_performance
        
        # Regime stability
        regime_stability = self._calculate_regime_stability(regime_labels)
        analysis['regime_stability'] = regime_stability
        
        # Regime transitions
        transitions = np.sum(np.diff(regime_labels) != 0)
        analysis['n_transitions'] = transitions
        analysis['transition_frequency'] = transitions / len(regime_labels)
        
        return analysis
    
    def _create_performance_summary(self, return_metrics: Dict[str, float],
                                  risk_metrics: Dict[str, float],
                                  risk_adjusted_metrics: Dict[str, float],
                                  drawdown_metrics: Dict[str, float],
                                  regime_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Create performance summary."""
        summary = {}
        
        # Overall performance score
        sharpe_ratio = risk_adjusted_metrics.get('sharpe_ratio_1m', 0.0)
        max_drawdown = drawdown_metrics.get('max_drawdown', 1.0)
        win_rate = regime_metrics.get('win_rate', 0.5)
        
        # Simple performance score
        performance_score = sharpe_ratio * (1 - max_drawdown) * win_rate
        summary['performance_score'] = performance_score
        
        # Performance grade
        if performance_score > 0.5:
            grade = 'A'
        elif performance_score > 0.3:
            grade = 'B'
        elif performance_score > 0.1:
            grade = 'C'
        else:
            grade = 'D'
        summary['performance_grade'] = grade
        
        # Key metrics
        summary['key_metrics'] = {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_return': return_metrics.get('total_return', 0.0),
            'volatility': risk_metrics.get('volatility', 0.0)
        }
        
        return summary
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis."""
        if len(returns) < 4:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis
    
    def _calculate_tail_risk(self, returns: np.ndarray) -> float:
        """Calculate tail risk."""
        if len(returns) < 10:
            return 0.0
        
        # Calculate tail risk as the ratio of extreme returns to normal returns
        extreme_threshold = np.percentile(np.abs(returns), 95)
        extreme_returns = returns[np.abs(returns) > extreme_threshold]
        
        if len(extreme_returns) > 0:
            tail_risk = np.std(extreme_returns) / (np.std(returns) + 1e-8)
        else:
            tail_risk = 0.0
        
        return tail_risk
    
    def _calculate_drawdown_duration(self, drawdown: np.ndarray) -> np.ndarray:
        """Calculate drawdown duration."""
        duration = np.zeros(len(drawdown))
        current_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown[i] > 0:
                current_duration += 1
            else:
                current_duration = 0
            duration[i] = current_duration
        
        return duration
    
    def _calculate_weighted_returns(self, returns: np.ndarray, 
                                  regime_labels: np.ndarray) -> np.ndarray:
        """Calculate regime-weighted returns."""
        weighted_returns = np.zeros_like(returns)
        
        for i, (return_val, regime) in enumerate(zip(returns, regime_labels)):
            regime_type = RegimeType(regime) if regime < len(RegimeType) else RegimeType.BULL
            weight = self.config.regime_weights.get(regime_type, 1.0)
            weighted_returns[i] = return_val * weight
        
        return weighted_returns
    
    def _calculate_regime_stability(self, regime_labels: np.ndarray) -> float:
        """Calculate regime stability."""
        if len(regime_labels) < 2:
            return 0.0
        
        # Calculate regime consistency
        unique_regimes = np.unique(regime_labels)
        regime_counts = {}
        
        for regime in unique_regimes:
            regime_counts[regime] = np.sum(regime_labels == regime)
        
        # Stability is the ratio of the most frequent regime to total length
        max_count = max(regime_counts.values())
        stability = max_count / len(regime_labels)
        
        return stability
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering."""
        if len(returns) < 20:
            return 0.0
        
        # Calculate rolling volatility
        rolling_vol = pd.Series(returns).rolling(window=20).std().values
        
        # Calculate autocorrelation of volatility
        vol_autocorr = np.corrcoef(rolling_vol[:-1], rolling_vol[1:])[0, 1]
        
        return vol_autocorr if not np.isnan(vol_autocorr) else 0.0
    
    def _calculate_momentum_persistence(self, returns: np.ndarray) -> float:
        """Calculate momentum persistence."""
        if len(returns) < 10:
            return 0.0
        
        # Calculate momentum persistence as autocorrelation of returns
        momentum_persistence = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        return momentum_persistence if not np.isnan(momentum_persistence) else 0.0
    
    def _calculate_momentum_reversal(self, returns: np.ndarray) -> float:
        """Calculate momentum reversal."""
        if len(returns) < 10:
            return 0.0
        
        # Calculate momentum reversal as negative autocorrelation
        momentum_reversal = -np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        return momentum_reversal if not np.isnan(momentum_reversal) else 0.0
    
    def _create_error_result(self, error_message: str, execution_time: float) -> FinancialPerformanceResult:
        """Create error result."""
        return FinancialPerformanceResult(
            return_metrics={},
            risk_metrics={},
            risk_adjusted_metrics={},
            drawdown_metrics={},
            regime_metrics={},
            trading_metrics={},
            volatility_metrics={},
            momentum_metrics={},
            regime_analysis={'error': error_message},
            performance_summary={},
            execution_time=execution_time,
            n_samples=0
        )

def create_financial_performance_evaluator(config: FinancialPerformanceConfig) -> FinancialPerformanceEvaluator:
    """Create financial performance evaluator instance."""
    return FinancialPerformanceEvaluator(config)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
