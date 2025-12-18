"""Shared Evaluation Metrics Utilities.

This module provides common evaluation metrics that can be used by both
NAS and TAS systems. It includes financial metrics, statistical metrics,
and regime-aware evaluation capabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import time
import warnings
warnings.filterwarnings('ignore')

from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Try to import financial libraries
try:
    import scipy.stats as stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        roc_auc_score, log_loss
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FinancialMetrics:
    """Financial performance metrics."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    volatility: float = 0.0
    downside_volatility: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    hit_rate: float = 0.0
    payoff_ratio: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0

@dataclass
class StatisticalMetrics:
    """Statistical performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    log_loss: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0

@dataclass
class RegimeMetrics:
    """Regime-specific performance metrics."""
    regime_accuracy: float = 0.0
    regime_stability: float = 0.0
    regime_transition_accuracy: float = 0.0
    regime_duration_accuracy: float = 0.0
    regime_volatility_accuracy: float = 0.0
    regime_trend_accuracy: float = 0.0
    regime_correlation: float = 0.0
    regime_consistency: float = 0.0

@dataclass
class EconomicMetrics:
    """Economic significance metrics."""
    economic_significance: float = 0.0
    trading_viability: float = 0.0
    transaction_cost_impact: float = 0.0
    slippage_impact: float = 0.0
    market_impact: float = 0.0
    liquidity_impact: float = 0.0
    capacity_utilization: float = 0.0
    risk_adjusted_capacity: float = 0.0

@dataclass
class UnifiedEvaluationResult:
    """Unified evaluation result."""

    # Core metrics
    financial_metrics: FinancialMetrics
    statistical_metrics: StatisticalMetrics
    regime_metrics: RegimeMetrics
    economic_metrics: EconomicMetrics

    # Composite scores
    overall_score: float = 0.0
    risk_adjusted_score: float = 0.0
    regime_aware_score: float = 0.0
    economic_score: float = 0.0
    trading_score: float = 0.0
    custom_balanced_score: float = 0.0

    # Performance breakdown
    performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_by_timeframe: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

class BaseMetricCalculator(ABC):
    """Abstract base class for metric calculators."""

    def __init__(self):
        """Initialize metric calculator."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate metrics."""
        pass

class FinancialMetricCalculator(BaseMetricCalculator):
    """Calculator for financial metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> FinancialMetrics:
        """Calculate financial metrics."""
        try:
            if returns is None:
                # Use predictions as returns if not provided
                returns = predictions

            returns = np.array(returns)

            # Basic statistics
            mean_return = np.mean(returns)
            volatility = np.std(returns)
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) if len(downside_returns) > 0 else 0.0

            # Sharpe ratio
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0

            # Sortino ratio
            sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else 0.0

            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)

            # Maximum drawdown duration
            drawdown_periods = np.where(drawdown < 0)[0]
            max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_periods)

            # Calmar ratio
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)

            # Conditional Value at Risk (CVaR)
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])

            # Higher moments
            skewness = stats.skew(returns) if SCIPY_AVAILABLE else 0.0
            kurtosis = stats.kurtosis(returns) if SCIPY_AVAILABLE else 0.0

            # Hit rate and payoff ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            hit_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

            avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0.0
            avg_loss = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf') if avg_win > 0 else 0.0

            # Total and annualized returns
            total_return = np.sum(returns)
            annualized_return = mean_return * 252  # Assuming daily returns

            return FinancialMetrics(
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                volatility=volatility,
                downside_volatility=downside_volatility,
                skewness=skewness,
                kurtosis=kurtosis,
                hit_rate=hit_rate,
                payoff_ratio=payoff_ratio,
                total_return=total_return,
                annualized_return=annualized_return
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Financial metrics calculation failed: {e}")
            return FinancialMetrics()

    def _calculate_max_drawdown_duration(self, drawdown_periods: np.ndarray) -> int:
        """Calculate maximum drawdown duration."""
        try:
            if len(drawdown_periods) == 0:
                return 0

            # Find consecutive periods
            consecutive_periods = []
            current_period = [drawdown_periods[0]]

            for i in range(1, len(drawdown_periods)):
                if drawdown_periods[i] == drawdown_periods[i-1] + 1:
                    current_period.append(drawdown_periods[i])
                else:
                    consecutive_periods.append(len(current_period))
                    current_period = [drawdown_periods[i]]

            consecutive_periods.append(len(current_period))
            return max(consecutive_periods)

        except Exception:
            return 0

class StatisticalMetricCalculator(BaseMetricCalculator):
    """Calculator for statistical metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> StatisticalMetrics:
        """Calculate statistical metrics."""
        try:
            if not SKLEARN_AVAILABLE:
                return StatisticalMetrics()

            # Basic metrics
            accuracy = accuracy_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

            # Regression metrics
            mse = mean_squared_error(targets, predictions)
            mae = mean_absolute_error(targets, predictions)
            r2 = r2_score(targets, predictions)
            rmse = np.sqrt(mse)

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

            # AUC (for binary classification)
            try:
                if len(np.unique(targets)) == 2:
                    auc = roc_auc_score(targets, predictions)
                else:
                    auc = 0.0
            except:
                auc = 0.0

            # Log loss (for classification)
            try:
                if len(np.unique(targets)) == 2:
                    log_loss_val = log_loss(targets, predictions)
                else:
                    log_loss_val = 0.0
            except:
                log_loss_val = 0.0

            return StatisticalMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_score=auc,
                log_loss=log_loss_val,
                mse=mse,
                mae=mae,
                r2_score=r2,
                rmse=rmse,
                mape=mape
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Statistical metrics calculation failed: {e}")
            return StatisticalMetrics()

class RegimeMetricCalculator(BaseMetricCalculator):
    """Calculator for regime-specific metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> RegimeMetrics:
        """Calculate regime metrics."""
        try:
            if regime_labels is None:
                # Create dummy regime labels
                regime_labels = np.zeros(len(predictions))

            regime_labels = np.array(regime_labels)

            # Regime accuracy
            regime_accuracy = self._calculate_regime_accuracy(predictions, targets, regime_labels)

            # Regime stability
            regime_stability = self._calculate_regime_stability(regime_labels)

            # Regime transition accuracy
            regime_transition_accuracy = self._calculate_regime_transition_accuracy(
                predictions, targets, regime_labels
            )

            # Regime duration accuracy
            regime_duration_accuracy = self._calculate_regime_duration_accuracy(
                predictions, targets, regime_labels
            )

            # Regime volatility accuracy
            regime_volatility_accuracy = self._calculate_regime_volatility_accuracy(
                predictions, targets, regime_labels
            )

            # Regime trend accuracy
            regime_trend_accuracy = self._calculate_regime_trend_accuracy(
                predictions, targets, regime_labels
            )

            # Regime correlation
            regime_correlation = self._calculate_regime_correlation(
                predictions, targets, regime_labels
            )

            # Regime consistency
            regime_consistency = self._calculate_regime_consistency(
                predictions, targets, regime_labels
            )

            return RegimeMetrics(
                regime_accuracy=regime_accuracy,
                regime_stability=regime_stability,
                regime_transition_accuracy=regime_transition_accuracy,
                regime_duration_accuracy=regime_duration_accuracy,
                regime_volatility_accuracy=regime_volatility_accuracy,
                regime_trend_accuracy=regime_trend_accuracy,
                regime_correlation=regime_correlation,
                regime_consistency=regime_consistency
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Regime metrics calculation failed: {e}")
            return RegimeMetrics()

    def _calculate_regime_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                                 regime_labels: np.ndarray) -> float:
        """Calculate regime prediction accuracy."""
        try:
            correct_predictions = np.sum(predictions == targets)
            total_predictions = len(predictions)
            return correct_predictions / total_predictions if total_predictions > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_regime_stability(self, regime_labels: np.ndarray) -> float:
        """Calculate regime stability."""
        try:
            transitions = np.sum(np.diff(regime_labels) != 0)
            total_periods = len(regime_labels) - 1
            stability = 1.0 - (transitions / total_periods) if total_periods > 0 else 1.0
            return max(0.0, min(1.0, stability))
        except Exception:
            return 0.0

    def _calculate_regime_transition_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                                           regime_labels: np.ndarray) -> float:
        """Calculate regime transition prediction accuracy."""
        try:
            transition_points = np.where(np.diff(regime_labels) != 0)[0]

            if len(transition_points) == 0:
                return 1.0  # No transitions, perfect accuracy

            transition_accuracy = []
            for point in transition_points:
                if point < len(predictions) and point < len(targets):
                    transition_accuracy.append(predictions[point] == targets[point])

            return np.mean(transition_accuracy) if transition_accuracy else 0.0
        except Exception:
            return 0.0

    def _calculate_regime_duration_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                                          regime_labels: np.ndarray) -> float:
        """Calculate regime duration prediction accuracy."""
        try:
            regime_durations = []
            current_regime = regime_labels[0]
            current_duration = 1

            for i in range(1, len(regime_labels)):
                if regime_labels[i] == current_regime:
                    current_duration += 1
                else:
                    regime_durations.append(current_duration)
                    current_regime = regime_labels[i]
                    current_duration = 1

            regime_durations.append(current_duration)

            if len(regime_durations) > 1:
                duration_std = np.std(regime_durations)
                duration_mean = np.mean(regime_durations)
                consistency = 1.0 - (duration_std / duration_mean) if duration_mean > 0 else 0.0
                return max(0.0, min(1.0, consistency))
            else:
                return 1.0
        except Exception:
            return 0.0

    def _calculate_regime_volatility_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                                            regime_labels: np.ndarray) -> float:
        """Calculate regime volatility prediction accuracy."""
        try:
            unique_regimes = np.unique(regime_labels)
            regime_volatilities = []

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_predictions = predictions[regime_mask]
                regime_targets = targets[regime_mask]

                if len(regime_predictions) > 1:
                    pred_volatility = np.std(regime_predictions)
                    target_volatility = np.std(regime_targets)

                    if target_volatility > 0:
                        accuracy = 1.0 - abs(pred_volatility - target_volatility) / target_volatility
                        regime_volatilities.append(max(0.0, min(1.0, accuracy)))

            return np.mean(regime_volatilities) if regime_volatilities else 0.0
        except Exception:
            return 0.0

    def _calculate_regime_trend_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                                       regime_labels: np.ndarray) -> float:
        """Calculate regime trend prediction accuracy."""
        try:
            unique_regimes = np.unique(regime_labels)
            regime_trends = []

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_predictions = predictions[regime_mask]
                regime_targets = targets[regime_mask]

                if len(regime_predictions) > 1:
                    pred_trend = np.polyfit(range(len(regime_predictions)), regime_predictions, 1)[0]
                    target_trend = np.polyfit(range(len(regime_targets)), regime_targets, 1)[0]

                    if target_trend != 0:
                        accuracy = 1.0 - abs(pred_trend - target_trend) / abs(target_trend)
                        regime_trends.append(max(0.0, min(1.0, accuracy)))

            return np.mean(regime_trends) if regime_trends else 0.0
        except Exception:
            return 0.0

    def _calculate_regime_correlation(self, predictions: np.ndarray, targets: np.ndarray,
                                   regime_labels: np.ndarray) -> float:
        """Calculate regime correlation."""
        try:
            if len(predictions) != len(targets):
                return 0.0

            correlation = np.corrcoef(predictions, targets)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _calculate_regime_consistency(self, predictions: np.ndarray, targets: np.ndarray,
                                    regime_labels: np.ndarray) -> float:
        """Calculate regime consistency."""
        try:
            unique_regimes = np.unique(regime_labels)
            regime_consistencies = []

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_predictions = predictions[regime_mask]
                regime_targets = targets[regime_mask]

                if len(regime_predictions) > 1:
                    pred_variance = np.var(regime_predictions)
                    target_variance = np.var(regime_targets)

                    consistency = 1.0 / (1.0 + pred_variance + target_variance)
                    regime_consistencies.append(consistency)

            return np.mean(regime_consistencies) if regime_consistencies else 0.0
        except Exception:
            return 0.0

class EconomicMetricCalculator(BaseMetricCalculator):
    """Calculator for economic significance metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> EconomicMetrics:
        """Calculate economic metrics."""
        try:
            if returns is None:
                returns = predictions

            returns = np.array(returns)

            # Economic significance
            economic_significance = self._calculate_economic_significance(returns)

            # Trading viability
            trading_viability = self._calculate_trading_viability(returns)

            # Transaction cost impact
            transaction_cost_impact = self._calculate_transaction_cost_impact(returns)

            # Slippage impact
            slippage_impact = self._calculate_slippage_impact(returns)

            # Market impact
            market_impact = self._calculate_market_impact(returns)

            # Liquidity impact
            liquidity_impact = self._calculate_liquidity_impact(returns)

            # Capacity utilization
            capacity_utilization = self._calculate_capacity_utilization(returns)

            # Risk-adjusted capacity
            risk_adjusted_capacity = self._calculate_risk_adjusted_capacity(returns)

            return EconomicMetrics(
                economic_significance=economic_significance,
                trading_viability=trading_viability,
                transaction_cost_impact=transaction_cost_impact,
                slippage_impact=slippage_impact,
                market_impact=market_impact,
                liquidity_impact=liquidity_impact,
                capacity_utilization=capacity_utilization,
                risk_adjusted_capacity=risk_adjusted_capacity
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Economic metrics calculation failed: {e}")
            return EconomicMetrics()

    def _calculate_economic_significance(self, returns: np.ndarray) -> float:
        """Calculate economic significance."""
        try:
            mean_return = np.mean(returns)
            volatility = np.std(returns)

            if volatility > 0:
                significance = mean_return / volatility
                return max(0.0, min(1.0, significance))
            else:
                return 0.0
        except Exception:
            return 0.0

    def _calculate_trading_viability(self, returns: np.ndarray) -> float:
        """Calculate trading viability."""
        try:
            positive_returns = returns[returns > 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0

            consistency = 1.0 - np.std(returns) / (np.mean(returns) + 1e-8)
            consistency = max(0.0, min(1.0, consistency))

            viability = (win_rate + consistency) / 2.0
            return max(0.0, min(1.0, viability))
        except Exception:
            return 0.0

    def _calculate_transaction_cost_impact(self, returns: np.ndarray) -> float:
        """Calculate transaction cost impact."""
        try:
            transaction_cost = DEFAULT_TRANSACTION_COST
            net_returns = returns - transaction_cost

            original_return = np.sum(returns)
            net_return = np.sum(net_returns)

            if original_return != 0:
                impact = 1.0 - (net_return / original_return)
                return max(0.0, min(1.0, impact))
            else:
                return 0.0
        except Exception:
            return 0.0

    def _calculate_slippage_impact(self, returns: np.ndarray) -> float:
        """Calculate slippage impact."""
        try:
            slippage = 0.0005
            net_returns = returns - slippage

            original_return = np.sum(returns)
            net_return = np.sum(net_returns)

            if original_return != 0:
                impact = 1.0 - (net_return / original_return)
                return max(0.0, min(1.0, impact))
            else:
                return 0.0
        except Exception:
            return 0.0

    def _calculate_market_impact(self, returns: np.ndarray) -> float:
        """Calculate market impact."""
        try:
            large_returns = returns[np.abs(returns) > np.std(returns)]
            impact = len(large_returns) / len(returns) if len(returns) > 0 else 0.0
            return max(0.0, min(1.0, impact))
        except Exception:
            return 0.0

    def _calculate_liquidity_impact(self, returns: np.ndarray) -> float:
        """Calculate liquidity impact."""
        try:
            volatility = np.std(returns)
            impact = volatility / (volatility + 1.0)
            return max(0.0, min(1.0, impact))
        except Exception:
            return 0.0

    def _calculate_capacity_utilization(self, returns: np.ndarray) -> float:
        """Calculate capacity utilization."""
        try:
            mean_abs_return = np.mean(np.abs(returns))
            utilization = mean_abs_return / (mean_abs_return + 1.0)
            return max(0.0, min(1.0, utilization))
        except Exception:
            return 0.0

    def _calculate_risk_adjusted_capacity(self, returns: np.ndarray) -> float:
        """Calculate risk-adjusted capacity."""
        try:
            capacity = self._calculate_capacity_utilization(returns)
            volatility = np.std(returns)
            risk_adjustment = 1.0 / (1.0 + volatility)

            return capacity * risk_adjustment
        except Exception:
            return 0.0

class UnifiedEvaluator:
    """Unified evaluator for comprehensive model assessment."""

    def __init__(self):
        """Initialize unified evaluator."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize metric calculators
        self.financial_calculator = FinancialMetricCalculator()
        self.statistical_calculator = StatisticalMetricCalculator()
        self.regime_calculator = RegimeMetricCalculator()
        self.economic_calculator = EconomicMetricCalculator()

        self.logger.info("✅ Unified Evaluator initialized")

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray,
                 returns: Optional[np.ndarray] = None,
                 regime_labels: Optional[np.ndarray] = None) -> UnifiedEvaluationResult:
        """Perform comprehensive evaluation."""
        try:
            self.logger.info("🔍 Starting unified evaluation...")

            # Calculate all metric categories
            financial_metrics = self.financial_calculator.calculate(predictions, targets, returns, regime_labels)
            statistical_metrics = self.statistical_calculator.calculate(predictions, targets, returns, regime_labels)
            regime_metrics = self.regime_calculator.calculate(predictions, targets, returns, regime_labels)
            economic_metrics = self.economic_calculator.calculate(predictions, targets, returns, regime_labels)

            # Calculate composite scores
            overall_score = self._calculate_overall_score(
                financial_metrics, statistical_metrics, regime_metrics, economic_metrics
            )

            risk_adjusted_score = self._calculate_risk_adjusted_score(financial_metrics, statistical_metrics)
            regime_aware_score = self._calculate_regime_aware_score(regime_metrics, statistical_metrics)
            economic_score = self._calculate_economic_score(economic_metrics, financial_metrics)
            trading_score = self._calculate_trading_score(financial_metrics, economic_metrics)
            
            # Enhanced custom_balanced_score with all available metrics
            custom_balanced_score = self._calculate_custom_balanced_score(
                financial_metrics, 
                statistical_metrics,
                regime_metrics=regime_metrics,
                economic_metrics=economic_metrics,
                sample_count=len(predictions) if predictions is not None else None,
                use_pareto_scalarization=False  # Can be enabled for Pareto-optimal scoring
            )
           
            
            self.logger.info("✅ Unified evaluation completed")
            self.logger.info(f"   Overall score: {overall_score:.4f}")
            self.logger.info(f"   Risk-adjusted score: {risk_adjusted_score:.4f}")
            self.logger.info(f"   Regime-aware score: {regime_aware_score:.4f}")
            self.logger.info(f"   Economic score: {economic_score:.4f}")
            self.logger.info(f"   Trading score: {trading_score:.4f}")

            return UnifiedEvaluationResult(
                financial_metrics=financial_metrics,
                statistical_metrics=statistical_metrics,
                regime_metrics=regime_metrics,
                economic_metrics=economic_metrics,
                overall_score=overall_score,
                risk_adjusted_score=risk_adjusted_score,
                regime_aware_score=regime_aware_score,
                economic_score=economic_score,
                trading_score=trading_score,
                custom_balanced_score=custom_balanced_score,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Unified evaluation failed: {e}")
            return UnifiedEvaluationResult(
                financial_metrics=FinancialMetrics(),
                statistical_metrics=StatisticalMetrics(),
                regime_metrics=RegimeMetrics(),
                economic_metrics=EconomicMetrics(),
                success=False,
                error_message=str(e)
            )

    def _calculate_overall_score(self, financial_metrics: FinancialMetrics,
                               statistical_metrics: StatisticalMetrics,
                               regime_metrics: RegimeMetrics,
                               economic_metrics: EconomicMetrics) -> float:
        """Calculate overall composite score."""
        try:
            # Weighted combination of all metrics
            weights = {
                'financial': 0.3,
                'statistical': 0.3,
                'regime': 0.2,
                'economic': 0.2
            }

            # Financial component
            financial_score = (financial_metrics.sharpe_ratio + financial_metrics.sortino_ratio +
                             (1.0 - abs(financial_metrics.max_drawdown))) / 3.0

            # Statistical component
            statistical_score = (statistical_metrics.accuracy + statistical_metrics.f1_score) / 2.0

            # Regime component
            regime_score = (regime_metrics.regime_accuracy + regime_metrics.regime_stability) / 2.0

            # Economic component
            economic_score = (economic_metrics.economic_significance +
                            economic_metrics.trading_viability) / 2.0

            # Combine scores
            overall_score = (
                weights['financial'] * financial_score +
                weights['statistical'] * statistical_score +
                weights['regime'] * regime_score +
                weights['economic'] * economic_score
            )

            return max(0.0, min(1.0, overall_score))

        except Exception:
            return 0.0

    def _calculate_risk_adjusted_score(self, financial_metrics: FinancialMetrics,
                                     statistical_metrics: StatisticalMetrics) -> float:
        """Calculate risk-adjusted score."""
        try:
            risk_score = (financial_metrics.sharpe_ratio + financial_metrics.sortino_ratio) / 2.0
            accuracy_score = statistical_metrics.accuracy

            return (risk_score + accuracy_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_regime_aware_score(self, regime_metrics: RegimeMetrics,
                                    statistical_metrics: StatisticalMetrics) -> float:
        """Calculate regime-aware score."""
        try:
            regime_score = (regime_metrics.regime_accuracy + regime_metrics.regime_stability) / 2.0
            accuracy_score = statistical_metrics.accuracy

            return (regime_score + accuracy_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_economic_score(self, economic_metrics: EconomicMetrics,
                                financial_metrics: FinancialMetrics) -> float:
        """Calculate economic score."""
        try:
            economic_score = (economic_metrics.economic_significance +
                            economic_metrics.trading_viability) / 2.0
            financial_score = financial_metrics.sharpe_ratio / 2.0  # Normalize

            return (economic_score + financial_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_trading_score(self, financial_metrics: FinancialMetrics,
                               economic_metrics: EconomicMetrics) -> float:
        """Calculate trading score."""
        try:
            trading_score = (financial_metrics.hit_rate + financial_metrics.payoff_ratio / 10.0) / 2.0
            economic_score = economic_metrics.trading_viability

            return (trading_score + economic_score) / 2.0
        except Exception:
            return 0.0
    
    def _calculate_custom_balanced_score(
        self,
        financial_metrics,
        statistical_metrics,
        *,
        weights: Optional[Dict[str, Any]] = None,
        norm_config: Optional[Dict[str, Any]] = None,
        sample_count: Optional[int] = None,
        sample_count_min: int = 30,
        apply_sample_penalty: bool = True,
        return_components: bool = False,
        regime_metrics=None,
        economic_metrics=None,
        use_pareto_scalarization: bool = False
    ) -> float:
        """
        Simplified custom balanced score for ML trading models in HPO.
        
        This is the DEFAULT scoring metric for all ML-related trading models in HPO.
        Uses pareto.py's scalarize_financial_goals for financial scoring (70%) 
        combined with statistical metrics (30%).
        
        Financial scoring leverages:
        - Non-linear scaling (log for PnL, sigmoid for Sharpe, power for win rate)
        - Proven Pareto optimization utilities
        - Better optimization landscapes for HPO
        
        Returns:
          - If return_components=False -> single scalar score in [0,1].
          - If return_components=True -> tuple (single_score, financial_obj, statistical_obj, normed_dict, regime_obj, economic_obj)
            where each component is normalized to [0,1]
              
        Args:
          financial_metrics: FinancialMetrics object with:
            - sharpe_ratio: Risk-adjusted returns (used by Pareto scalarization)
            - total_return: Strategy PnL proxy (mapped to Pareto's 'pnl'; profit_factor used if available)
            - hit_rate: Win percentage (mapped to Pareto's 'win_rate')
            - max_drawdown: Maximum decline (25% weight, separate from Pareto scoring)
            
          statistical_metrics: StatisticalMetrics object with:
            - f1_score: Harmonic mean of precision and recall - 20% weight
            - accuracy: Correct predictions / total predictions - 10% weight
            - r2_score: Coefficient of determination - 10% weight
            
          weights: Optional custom weights (NOT USED - Pareto uses its own defaults)
          norm_config: Optional normalization configuration (min/max ranges)
          sample_count: Number of samples (for penalty if too few)
          sample_count_min: Minimum samples before penalty applies (default: 30)
          apply_sample_penalty: Whether to penalize small sample counts
          return_components: Return detailed components breakdown for analysis
          
          regime_metrics: Optional RegimeMetrics for regime-aware scoring (NOT used by default)
          economic_metrics: Optional EconomicMetrics (NOT used by default)
          use_pareto_scalarization: Legacy parameter (always uses Pareto now)
        
        Implementation:
          **Financial Component (70%) via pareto.py**:
          - Uses scalarize_financial_goals() with non-linear scaling
          - Maps: total_return (or profit_factor if available) → 'pnl', hit_rate → 'win_rate', sharpe_ratio → 'sharpe'
          - Pareto's default weights: approx pnl=66%, sharpe=33%, win_rate=0% (win_rate kept for legacy compatibility)
          - Max drawdown handled separately (25% weight as penalty)
          
          **Statistical Component (30%)**:
          - F1 score: 50%, Accuracy: 25%, R²: 25%
          - Standard linear combination of normalized metrics
          
          **Benefits of Pareto Integration**:
          - Non-linear scaling improves optimization landscapes
          - Log scaling for PnL handles extreme values
          - Sigmoid for Sharpe bounds the metric
          - Power scaling for win_rate enhances discrimination
          - Consistent with other Pareto-based code
        """
        import math, os
    
        # ---------- Simplified Financial & Statistical Metrics ----------
        # Economic viability removed per user request
        # Total return merged into profit_factor (they measure similar things)
        default_weights = {
            # Core financial metrics (60%)
            'sharpe': 0.30,           # Risk-adjusted returns (return / volatility)
            'max_drawdown': 0.15,     # Maximum peak-to-trough decline (lower is better)
            'profit_factor': 0.15,    # Profitability: gross profit / gross loss (includes return impact)
            # Statistical metrics (40%)
            'f1_score': 0.20,         # Harmonic mean of precision and recall
            'r2_score': 0.10,         # Coefficient of determination
            'accuracy': 0.10          # Correct predictions / total predictions
        }
        if weights is None:
            weights = default_weights
        # normalize passed weights
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}
    
        # ---------- Enhanced Normalization Config ----------
        default_norm = {
            # Financial metrics
            'sharpe': {'method': 'clamp', 'min': -1.0, 'max': 3.0, 'higher_is_better': True},
            'max_drawdown': {'method': 'clamp', 'min': 0.0, 'max': 0.6, 'higher_is_better': False},
            'profit_factor': {'method': 'clamp', 'min': 0.0, 'max': 5.0, 'higher_is_better': True},
            'total_return': {'method': 'clamp', 'min': -1.0, 'max': 2.0, 'higher_is_better': True},
            'sortino_ratio': {'method': 'clamp', 'min': -1.0, 'max': 4.0, 'higher_is_better': True},
            'calmar_ratio': {'method': 'clamp', 'min': -1.0, 'max': 3.0, 'higher_is_better': True},
            # Statistical metrics
            'f1_score': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'accuracy': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'r2_score': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'precision': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'recall': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            # Regime-aware metrics
            'regime_accuracy': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'regime_stability': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'regime_consistency': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            # Economic metrics
            'economic_significance': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
            'trading_viability': {'method': 'clamp', 'min': 0.0, 'max': 1.0, 'higher_is_better': True},
        }
        if norm_config is None:
            norm_config = {}
        # merge defaults
        for k, v in default_norm.items():
            norm_config.setdefault(k, v)
    
        def _norm(value, conf):
            try:
                if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    return 0.0
                method = conf.get('method', 'clamp')
                hib = conf.get('higher_is_better', True)
                if method == 'clamp':
                    lo = conf.get('min', 0.0)
                    hi = conf.get('max', 1.0)
                    # Protect if hi == lo
                    if hi == lo:
                        v = 0.0
                    else:
                        v = max(lo, min(hi, value))
                        v = (v - lo) / (hi - lo)
                else:
                    # fallback: treat as clamp
                    lo = conf.get('min', 0.0)
                    hi = conf.get('max', 1.0)
                    v = max(lo, min(hi, value))
                    v = (v - lo) / (hi - lo) if hi != lo else 0.0
                if not hib:
                    v = 1.0 - v
                return max(0.0, min(1.0, v))
            except Exception:
                return 0.0
    
        # ---------- Extract raw metrics (Enhanced with Regime & Economic) ----------
        raw = {}
        
        # Financial metrics
        raw['sharpe'] = getattr(financial_metrics, 'sharpe_ratio', None)
        raw_mdd = getattr(financial_metrics, 'max_drawdown', None)
        if raw_mdd is not None:
            raw['max_drawdown'] = abs(raw_mdd)
        else:
            raw['max_drawdown'] = None
        raw['profit_factor'] = getattr(financial_metrics, 'profit_factor', None)
        raw['total_return'] = getattr(financial_metrics, 'total_return', None)
        raw['sortino_ratio'] = getattr(financial_metrics, 'sortino_ratio', None)
        raw['calmar_ratio'] = getattr(financial_metrics, 'calmar_ratio', None)
        
        # Statistical metrics
        raw['f1_score'] = getattr(statistical_metrics, 'f1_score', 0.0)
        raw['accuracy'] = getattr(statistical_metrics, 'accuracy', 0.0)
        raw['r2_score'] = getattr(statistical_metrics, 'r2_score', 0.0)
        raw['precision'] = getattr(statistical_metrics, 'precision', None)
        raw['recall'] = getattr(statistical_metrics, 'recall', None)
        
        # Regime-aware metrics (if provided)
        if regime_metrics is not None:
            raw['regime_accuracy'] = getattr(regime_metrics, 'regime_accuracy', None)
            raw['regime_stability'] = getattr(regime_metrics, 'regime_stability', None)
            raw['regime_consistency'] = getattr(regime_metrics, 'regime_consistency', None)
        else:
            raw['regime_accuracy'] = None
            raw['regime_stability'] = None
            raw['regime_consistency'] = None
        
        # Economic metrics (if provided)
        if economic_metrics is not None:
            raw['economic_significance'] = getattr(economic_metrics, 'economic_significance', None)
            raw['trading_viability'] = getattr(economic_metrics, 'trading_viability', None)
        else:
            raw['economic_significance'] = None
            raw['trading_viability'] = None
    
        # ---------- Normalize ----------
        normed = {k: _norm(v, norm_config.get(k, {})) for k, v in raw.items()}
    
        # ---------- Compose multi-objective components using Pareto utilities ----------
        
        # Use Pareto.py's scalarize_financial_goals for financial scoring
        # This leverages existing non-linear scaling and optimization
        try:
            from ..pareto import scalarize_financial_goals
            
            # Map financial metrics to Pareto's expected format
            # Note: We map our metrics to PnL/win_rate/sharpe that Pareto understands
            pareto_financial_metrics = {}
            
            if raw.get('sharpe') is not None:
                pareto_financial_metrics['sharpe'] = raw['sharpe']
            
            # Map PnL proxy to 'pnl': prefer profit_factor if available, otherwise total_return
            if raw.get('profit_factor') is not None:
                pareto_financial_metrics['pnl'] = raw['profit_factor']
            elif raw.get('total_return') is not None:
                pareto_financial_metrics['pnl'] = raw['total_return']
            
            # Use hit_rate if available, otherwise derive from statistical metrics
            if hasattr(financial_metrics, 'hit_rate') and financial_metrics.hit_rate is not None:
                pareto_financial_metrics['win_rate'] = financial_metrics.hit_rate
            elif raw.get('accuracy') is not None:
                # Fallback: use accuracy as proxy for win_rate
                pareto_financial_metrics['win_rate'] = raw['accuracy']
            
            # Add max_drawdown as additional constraint (lower is better)
            # We'll handle this separately since Pareto's scalarize expects different metrics
            mdd_penalty = 1.0
            if raw.get('max_drawdown') is not None:
                # Convert drawdown to a bonus (1.0 = no drawdown, 0.0 = max drawdown)
                mdd_normalized = normed.get('max_drawdown', 0.5)  # Already inverted in normalization
                mdd_penalty = mdd_normalized
            
            # Use Pareto's scalarization with non-linear scaling
            financial_obj = scalarize_financial_goals(
                pareto_financial_metrics,
                weights=None,  # Use Pareto's default weights (pnl:50%, win_rate:25%, sharpe:25%)
                use_nonlinear_scaling=True  # Enable log/sigmoid/power scaling
            )
            
            # Adjust for max drawdown penalty (25% weight in standard mode)
            # Optional risk-focus mode increases drawdown contribution while keeping PnL dominant.
            risk_focus_env = os.getenv("ARES_HPO_RISK_FOCUS", "").lower()
            risk_focus = risk_focus_env in ("1", "true", "yes", "risk", "high")
            if risk_focus:
                base_weight, mdd_weight = 0.60, 0.40
            else:
                base_weight, mdd_weight = 0.75, 0.25
            financial_obj = base_weight * financial_obj + mdd_weight * mdd_penalty
            
            # Clamp to [0, 1]
            financial_obj = max(0.0, min(1.0, financial_obj))
            
        except Exception as e:
            self.logger.warning(f"Pareto scalarization failed: {e}, using fallback")
            # Fallback: simple weighted average if Pareto unavailable
            fin_weights = {
                'sharpe': 0.50,
                'profit_factor': 0.25,
                'max_drawdown': 0.25
            }
            available_fin = {k: v for k, v in fin_weights.items() if normed.get(k) is not None}
            s = sum(available_fin.values()) or 1.0
            fin_weights = {k: v / s for k, v in available_fin.items()}
            financial_obj = sum(fin_weights.get(k, 0.0) * normed.get(k, 0.0) for k in fin_weights.keys())
        
        # Statistical objective: prediction quality (always calculated)
        stat_weights = {'f1_score': 0.50, 'accuracy': 0.25, 'r2_score': 0.25}
        # precision and recall available if provided, but not in default weights
        available_stat = {k: v for k, v in stat_weights.items() if normed.get(k) is not None}
        s2 = sum(available_stat.values()) or 1.0
        stat_weights = {k: v / s2 for k, v in available_stat.items()}
        statistical_obj = sum(stat_weights.get(k, 0.0) * normed.get(k, 0.0) for k in stat_weights.keys())
        
        # Regime-aware objective: market regime adaptation (if available)
        regime_obj = 0.0
        if regime_metrics is not None:
            regime_weights = {'regime_accuracy': 0.5, 'regime_stability': 0.3, 'regime_consistency': 0.2}
            available_regime = {k: v for k, v in regime_weights.items() if normed.get(k) is not None}
            if available_regime:
                s3 = sum(available_regime.values()) or 1.0
                regime_weights = {k: v / s3 for k, v in available_regime.items()}
                regime_obj = sum(regime_weights.get(k, 0.0) * normed.get(k, 0.0) for k in regime_weights.keys())
        
        # Economic objective: NOT USED BY DEFAULT (removed per user request)
        # Can still be calculated if economic_metrics provided, but not included in score
        economic_obj = 0.0
        if economic_metrics is not None:
            econ_weights = {'economic_significance': 0.6, 'trading_viability': 0.4}
            available_econ = {k: v for k, v in econ_weights.items() if normed.get(k) is not None}
            if available_econ:
                s4 = sum(available_econ.values()) or 1.0
                econ_weights = {k: v / s4 for k, v in available_econ.items()}
                economic_obj = sum(econ_weights.get(k, 0.0) * normed.get(k, 0.0) for k in econ_weights.keys())
        
        # ---------- Simplified Composite Score ----------
        # Clean 70/30 split: Financial vs Statistical
        # Financial component uses Pareto.py's scalarize_financial_goals
        # Economic viability removed (redundant with profit_factor)
        # Regime awareness optional (only if explicitly provided)
        composite_weights = {
            'financial': 0.70,      # Risk-adjusted returns via Pareto, drawdown, profitability
            'statistical': 0.30,    # Prediction accuracy
            'regime': 0.0,          # Optional (only if regime_metrics provided)
            'economic': 0.0         # Removed from default
        }
        
        # If regime metrics are explicitly provided, include them
        if regime_obj > 0.0 and regime_metrics is not None:
            # Reduce financial and statistical proportionally (keeping ~70/30 ratio) to make room for regime (20%)
            composite_weights['financial'] = 0.56
            composite_weights['statistical'] = 0.24
            composite_weights['regime'] = 0.20  # Significant weight if explicitly provided
            composite_weights['economic'] = 0.0
        
        # Normalize composite weights to ensure they sum to 1.0
        total_composite = sum(composite_weights.values()) or 1.0
        composite_weights = {k: v / total_composite for k, v in composite_weights.items()}
        
        scalar_score = (
            composite_weights['financial'] * financial_obj +
            composite_weights['statistical'] * statistical_obj +
            composite_weights['regime'] * regime_obj
            # economic_obj not included (removed per user request)
        )
    
        # ---------- Sample count penalty ----------
        if apply_sample_penalty and sample_count is not None:
            if sample_count < sample_count_min:
                penalty = float(sample_count) / float(sample_count_min)
                penalty = max(0.0, min(1.0, penalty))
                scalar_score *= penalty
                financial_obj *= penalty
                statistical_obj *= penalty
    
        # clamp safety
        scalar_score = max(0.0, min(1.0, scalar_score))
        financial_obj = max(0.0, min(1.0, financial_obj))
        statistical_obj = max(0.0, min(1.0, statistical_obj))
        regime_obj = max(0.0, min(1.0, regime_obj)) if regime_obj > 0 else 0.0
        economic_obj = max(0.0, min(1.0, economic_obj)) if economic_obj > 0 else 0.0
    
        if return_components:
            # Enhanced return with all components
            return scalar_score, financial_obj, statistical_obj, normed, regime_obj, economic_obj
    
        return scalar_score

            

# Convenience functions
def create_unified_evaluator() -> UnifiedEvaluator:
    """Create unified evaluator instance."""
    return UnifiedEvaluator()

def quick_unified_evaluation(predictions: np.ndarray, targets: np.ndarray,
                            returns: Optional[np.ndarray] = None,
                            regime_labels: Optional[np.ndarray] = None) -> UnifiedEvaluationResult:
    """Quick unified evaluation with default settings."""
    evaluator = create_unified_evaluator()
    return evaluator.evaluate(predictions, targets, returns, regime_labels)

def calculate_custom_balanced_score_for_hpo(
    predictions: np.ndarray,
    targets: np.ndarray,
    returns: Optional[np.ndarray] = None,
    regime_labels: Optional[np.ndarray] = None,
    **kwargs
) -> float:
    """
    Convenience function to calculate custom_balanced_score for HPO.
    
    This is the recommended scoring function for all ML trading models in HPO.
    Uses pareto.py's proven scalarization for financial metrics with non-linear scaling.
    
    Clean evaluation balancing:
    - **Financial performance (60%)**: Via pareto.py's scalarize_financial_goals with non-linear scaling
    - **Statistical accuracy (40%)**: Prediction quality metrics
    
    Args:
        predictions: Model predictions (e.g., direction, signal strength, price forecast)
        targets: Target values (actual outcomes)
        returns: Optional return series for financial metrics
                 Example: For direction prediction: predictions * actual_returns
                          For price prediction: (predicted_price - actual_price) / actual_price
        regime_labels: Optional regime labels (NOT used by default, but available if provided)
        **kwargs: Additional arguments passed to custom_balanced_score
        
    Returns:
        float: Balanced score in [0, 1] (higher is better, maximize this in HPO)
        
    Score Breakdown:
        **Financial (60%)** - via pareto.py with non-linear scaling:
        - Uses scalarize_financial_goals() function
        - PnL/Profit Factor: Log scaling (handles extreme values)
        - Sharpe Ratio: Sigmoid scaling (bounded transformation)
        - Win Rate: Power scaling (better discrimination)
        - Max Drawdown: 25% weight as risk penalty
        
        **Statistical (40%)**:
        - F1 score (50%): Balance of precision and recall
        - Accuracy (25%): Percentage of correct predictions
        - R² score (25%): How well predictions explain variance
        
    Why Pareto Integration:
        - Better optimization landscapes (non-linear transformations)
        - Proven and tested in production
        - Consistent with other Pareto-based code
        - Handles extreme values gracefully (log scaling)
        
    Example:
        ```python
        # In HPO objective function
        def objective_func(params, X_train, y_train, X_val, y_val, **kwargs):
            model = create_model(params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            
            # Calculate returns (example for direction prediction)
            # If predicting market direction (+1 = up, -1 = down):
            actual_returns = calculate_market_returns(y_val)
            strategy_returns = predictions * actual_returns  # +1 if correct, -1 if wrong
            
            score = calculate_custom_balanced_score_for_hpo(
                predictions=predictions,
                targets=y_val,
                returns=strategy_returns  # This generates Sharpe, drawdown, profit factor
            )
            return score  # Maximize this! (benefits from Pareto's non-linear scaling)
        ```
    """
    try:
        evaluator = create_unified_evaluator()
        result = evaluator.evaluate(predictions, targets, returns, regime_labels)
        
        if result.success and result.custom_balanced_score is not None:
            return result.custom_balanced_score
        else:
            # Fallback to overall_score if custom_balanced_score is not available
            logger.warning("custom_balanced_score not available, using overall_score")
            return result.overall_score if result.overall_score is not None else 0.0
            
    except Exception as e:
        logger.error(f"Error calculating custom_balanced_score for HPO: {e}")
        return 0.0
