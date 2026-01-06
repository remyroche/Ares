"""
Advanced Evaluation Metrics for NAS and TAS

This module provides comprehensive evaluation metrics that can be used by both
NAS and TAS systems. It includes risk-adjusted returns, regime stability,
economic significance, and trading viability metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
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
class RiskMetrics:
    """Risk-adjusted performance metrics."""
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

@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    total_trades: int = 0
    profitable_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

@dataclass
class RegimeMetrics:
    """Regime-specific performance metrics."""
    regime_accuracy: float = 0.0
    regime_stability: float = 0.0
    regime_transition_accuracy: float = 0.0
    regime_duration_accuracy: float = 0.0
    regime_volatility_accuracy: float = 0.0
    regime_trend_accuracy: float = 0.0
    regime_volume_accuracy: float = 0.0
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
class ModelMetrics:
    """Model performance metrics."""
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
class AdvancedEvaluationResult:
    """Comprehensive evaluation result."""

    # Core metrics
    risk_metrics: RiskMetrics
    trading_metrics: TradingMetrics
    regime_metrics: RegimeMetrics
    economic_metrics: EconomicMetrics
    model_metrics: ModelMetrics

    # Composite scores
    overall_score: float = 0.0
    risk_adjusted_score: float = 0.0
    regime_aware_score: float = 0.0
    economic_score: float = 0.0
    trading_score: float = 0.0

    # Performance breakdown
    performance_by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_by_timeframe: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Model characteristics
    model_complexity: float = 0.0
    model_stability: float = 0.0
    model_robustness: float = 0.0
    model_interpretability: float = 0.0

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

class RiskMetricCalculator(BaseMetricCalculator):
    """Calculator for risk-adjusted metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> RiskMetrics:
        """Calculate risk metrics."""
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
            if len(drawdown_periods) > 0:
                max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_periods)
            else:
                max_drawdown_duration = 0

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

            return RiskMetrics(
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
                kurtosis=kurtosis
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Risk metrics calculation failed: {e}")
            return RiskMetrics()

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

class TradingMetricCalculator(BaseMetricCalculator):
    """Calculator for trading performance metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> TradingMetrics:
        """Calculate trading metrics."""
        try:
            if returns is None:
                returns = predictions

            returns = np.array(returns)

            # Basic trading statistics
            total_return = np.sum(returns)
            annualized_return = np.mean(returns) * 252  # Assuming daily returns

            # Win/Loss analysis
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            zero_returns = returns[returns == 0]

            total_trades = len(returns)
            profitable_trades = len(positive_returns)
            losing_trades = len(negative_returns)
            breakeven_trades = len(zero_returns)

            win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0

            # Profit/Loss metrics
            average_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0.0
            average_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0.0

            largest_win = np.max(positive_returns) if len(positive_returns) > 0 else 0.0
            largest_loss = np.min(negative_returns) if len(negative_returns) > 0 else 0.0

            # Profit factor
            total_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
            total_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_trades(returns)

            return TradingMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                losing_trades=losing_trades,
                breakeven_trades=breakeven_trades
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Trading metrics calculation failed: {e}")
            return TradingMetrics()

    def _calculate_consecutive_trades(self, returns: np.ndarray) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        try:
            consecutive_wins = 0
            consecutive_losses = 0
            current_wins = 0
            current_losses = 0

            for ret in returns:
                if ret > 0:
                    current_wins += 1
                    current_losses = 0
                    consecutive_wins = max(consecutive_wins, current_wins)
                elif ret < 0:
                    current_losses += 1
                    current_wins = 0
                    consecutive_losses = max(consecutive_losses, current_losses)
                else:
                    current_wins = 0
                    current_losses = 0

            return consecutive_wins, consecutive_losses

        except Exception:
            return 0, 0

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

            # Regime volume accuracy
            regime_volume_accuracy = self._calculate_regime_volume_accuracy(
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
                regime_volume_accuracy=regime_volume_accuracy,
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

    def _calculate_regime_volume_accuracy(self, predictions: np.ndarray, targets: np.ndarray,
                                        regime_labels: np.ndarray) -> float:
        """Calculate regime volume prediction accuracy."""
        try:
            # Similar to volatility accuracy but for volume
            return self._calculate_regime_volatility_accuracy(predictions, targets, regime_labels)
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

class ModelMetricCalculator(BaseMetricCalculator):
    """Calculator for model performance metrics."""

    def calculate(self, predictions: np.ndarray, targets: np.ndarray,
                  returns: Optional[np.ndarray] = None,
                  regime_labels: Optional[np.ndarray] = None) -> ModelMetrics:
        """Calculate model metrics."""
        try:
            if not SKLEARN_AVAILABLE:
                return ModelMetrics()

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

            return ModelMetrics(
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
            self.logger.warning(f"⚠️ Model metrics calculation failed: {e}")
            return ModelMetrics()

class AdvancedEvaluator:
    """Advanced evaluator for comprehensive model assessment."""

    def __init__(self):
        """Initialize advanced evaluator."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize metric calculators
        self.risk_calculator = RiskMetricCalculator()
        self.trading_calculator = TradingMetricCalculator()
        self.regime_calculator = RegimeMetricCalculator()
        self.economic_calculator = EconomicMetricCalculator()
        self.model_calculator = ModelMetricCalculator()

        self.logger.info("✅ Advanced Evaluator initialized")

    def evaluate(self, predictions: np.ndarray, targets: np.ndarray,
                 returns: Optional[np.ndarray] = None,
                 regime_labels: Optional[np.ndarray] = None,
                 model_complexity: float = 0.0) -> AdvancedEvaluationResult:
        """Perform comprehensive evaluation.

        Args:
            predictions: Model predictions
            targets: True targets
            returns: Returns data (optional)
            regime_labels: Regime labels (optional)
            model_complexity: Model complexity score

        Returns:
            AdvancedEvaluationResult with comprehensive metrics
        """
        try:
            self.logger.info("🔍 Starting advanced evaluation...")

            # Calculate all metric categories
            risk_metrics = self.risk_calculator.calculate(predictions, targets, returns, regime_labels)
            trading_metrics = self.trading_calculator.calculate(predictions, targets, returns, regime_labels)
            regime_metrics = self.regime_calculator.calculate(predictions, targets, returns, regime_labels)
            economic_metrics = self.economic_calculator.calculate(predictions, targets, returns, regime_labels)
            model_metrics = self.model_calculator.calculate(predictions, targets, returns, regime_labels)

            # Calculate composite scores
            overall_score = self._calculate_overall_score(
                risk_metrics, trading_metrics, regime_metrics, economic_metrics, model_metrics
            )

            risk_adjusted_score = self._calculate_risk_adjusted_score(risk_metrics, trading_metrics)
            regime_aware_score = self._calculate_regime_aware_score(regime_metrics, model_metrics)
            economic_score = self._calculate_economic_score(economic_metrics, trading_metrics)
            trading_score = self._calculate_trading_score(trading_metrics, risk_metrics)

            # Calculate model characteristics
            model_stability = self._calculate_model_stability(predictions, targets)
            model_robustness = self._calculate_model_robustness(predictions, targets)
            model_interpretability = self._calculate_model_interpretability(model_complexity)

            self.logger.info("✅ Advanced evaluation completed")
            self.logger.info(f"   Overall score: {overall_score:.4f}")
            self.logger.info(f"   Risk-adjusted score: {risk_adjusted_score:.4f}")
            self.logger.info(f"   Regime-aware score: {regime_aware_score:.4f}")
            self.logger.info(f"   Economic score: {economic_score:.4f}")
            self.logger.info(f"   Trading score: {trading_score:.4f}")

            return AdvancedEvaluationResult(
                risk_metrics=risk_metrics,
                trading_metrics=trading_metrics,
                regime_metrics=regime_metrics,
                economic_metrics=economic_metrics,
                model_metrics=model_metrics,
                overall_score=overall_score,
                risk_adjusted_score=risk_adjusted_score,
                regime_aware_score=regime_aware_score,
                economic_score=economic_score,
                trading_score=trading_score,
                model_complexity=model_complexity,
                model_stability=model_stability,
                model_robustness=model_robustness,
                model_interpretability=model_interpretability,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ Advanced evaluation failed: {e}")
            return AdvancedEvaluationResult(
                risk_metrics=RiskMetrics(),
                trading_metrics=TradingMetrics(),
                regime_metrics=RegimeMetrics(),
                economic_metrics=EconomicMetrics(),
                model_metrics=ModelMetrics(),
                success=False,
                error_message=str(e)
            )

    def _calculate_overall_score(self, risk_metrics: RiskMetrics, trading_metrics: TradingMetrics,
                               regime_metrics: RegimeMetrics, economic_metrics: EconomicMetrics,
                               model_metrics: ModelMetrics) -> float:
        """Calculate overall composite score."""
        try:
            # Weighted combination of all metrics
            weights = {
                'risk': 0.25,
                'trading': 0.25,
                'regime': 0.20,
                'economic': 0.15,
                'model': 0.15
            }

            # Risk component
            risk_score = (risk_metrics.sharpe_ratio + risk_metrics.sortino_ratio +
                         (1.0 - abs(risk_metrics.max_drawdown))) / 3.0

            # Trading component
            trading_score = (trading_metrics.win_rate + trading_metrics.profit_factor / 10.0) / 2.0

            # Regime component
            regime_score = (regime_metrics.regime_accuracy + regime_metrics.regime_stability) / 2.0

            # Economic component
            economic_score = (economic_metrics.economic_significance +
                            economic_metrics.trading_viability) / 2.0

            # Model component
            model_score = (model_metrics.accuracy + model_metrics.f1_score) / 2.0

            # Combine scores
            overall_score = (
                weights['risk'] * risk_score +
                weights['trading'] * trading_score +
                weights['regime'] * regime_score +
                weights['economic'] * economic_score +
                weights['model'] * model_score
            )

            return max(0.0, min(1.0, overall_score))

        except Exception:
            return 0.0

    def _calculate_risk_adjusted_score(self, risk_metrics: RiskMetrics,
                                     trading_metrics: TradingMetrics) -> float:
        """Calculate risk-adjusted score."""
        try:
            risk_score = (risk_metrics.sharpe_ratio + risk_metrics.sortino_ratio) / 2.0
            trading_score = trading_metrics.win_rate

            return (risk_score + trading_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_regime_aware_score(self, regime_metrics: RegimeMetrics,
                                    model_metrics: ModelMetrics) -> float:
        """Calculate regime-aware score."""
        try:
            regime_score = (regime_metrics.regime_accuracy + regime_metrics.regime_stability) / 2.0
            model_score = model_metrics.accuracy

            return (regime_score + model_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_economic_score(self, economic_metrics: EconomicMetrics,
                                trading_metrics: TradingMetrics) -> float:
        """Calculate economic score."""
        try:
            economic_score = (economic_metrics.economic_significance +
                            economic_metrics.trading_viability) / 2.0
            trading_score = trading_metrics.profit_factor / 10.0  # Normalize

            return (economic_score + trading_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_trading_score(self, trading_metrics: TradingMetrics,
                               risk_metrics: RiskMetrics) -> float:
        """Calculate trading score."""
        try:
            trading_score = (trading_metrics.win_rate + trading_metrics.profit_factor / 10.0) / 2.0
            risk_score = 1.0 - abs(risk_metrics.max_drawdown)

            return (trading_score + risk_score) / 2.0
        except Exception:
            return 0.0

    def _calculate_model_stability(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate model stability."""
        try:
            prediction_std = np.std(predictions)
            target_std = np.std(targets)

            if target_std > 0:
                stability = 1.0 - (prediction_std / target_std)
                return max(0.0, min(1.0, stability))
            else:
                return 1.0
        except Exception:
            return 0.0

    def _calculate_model_robustness(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate model robustness."""
        try:
            errors = np.abs(predictions - targets)
            error_std = np.std(errors)
            error_mean = np.mean(errors)

            if error_mean > 0:
                robustness = 1.0 - (error_std / error_mean)
                return max(0.0, min(1.0, robustness))
            else:
                return 1.0
        except Exception:
            return 0.0

    def _calculate_model_interpretability(self, model_complexity: float) -> float:
        """Calculate model interpretability."""
        try:
            interpretability = 1.0 / (1.0 + model_complexity)
            return max(0.0, min(1.0, interpretability))
        except Exception:
            return 0.0

# Convenience functions
def create_advanced_evaluator() -> AdvancedEvaluator:
    """Create advanced evaluator instance."""
    return AdvancedEvaluator()

def quick_advanced_evaluation(predictions: np.ndarray, targets: np.ndarray,
                             returns: Optional[np.ndarray] = None,
                             regime_labels: Optional[np.ndarray] = None) -> AdvancedEvaluationResult:
    """Quick advanced evaluation with default settings.

    Args:
        predictions: Model predictions
        targets: True targets
        returns: Returns data (optional)
        regime_labels: Regime labels (optional)

    Returns:
        AdvancedEvaluationResult with comprehensive metrics
    """
    evaluator = create_advanced_evaluator()
    return evaluator.evaluate(predictions, targets, returns, regime_labels)
