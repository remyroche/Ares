"""
Loss Calculator Module.

This module handles various loss calculations for model training and evaluation.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.core.decorators import handles_errors

from .base import PnLLossFunctionsBase
from src.core.decorators.errors import handles_errors


class LossCalculator(PnLLossFunctionsBase):
    """
    Loss Calculator for computing various loss metrics.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize loss calculator."""
        super().__init__(config)
        self.enable_loss_calculation: bool = self.pnl_config.get(
            "enable_loss_calculation", True
        )

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_trading_loss(self, predictions: np.ndarray, 
                              actuals: np.ndarray,
                              costs: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate trading-specific loss metrics.

        Args:
            predictions: Model predictions
            actuals: Actual values
            costs: Transaction costs (optional)

        Returns:
            Dictionary containing loss metrics
        """
        try:
            if len(predictions) != len(actuals):
                raise ValueError("Predictions and actuals must have same length")

            # Basic MSE loss
            mse_loss = np.mean((predictions - actuals) ** 2)
            
            # Mean Absolute Error
            mae_loss = np.mean(np.abs(predictions - actuals))
            
            # Directional accuracy
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            directional_accuracy = np.mean(pred_direction == actual_direction)
            
            # Trading loss with costs
            if costs is not None:
                # Penalize wrong directions more when costs are high
                wrong_direction_mask = pred_direction != actual_direction
                cost_adjusted_loss = mse_loss + np.mean(costs * wrong_direction_mask)
            else:
                cost_adjusted_loss = mse_loss

            return {
                "mse_loss": float(mse_loss),
                "mae_loss": float(mae_loss),
                "directional_accuracy": float(directional_accuracy),
                "cost_adjusted_loss": float(cost_adjusted_loss),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating trading loss: {e}")
            return {}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_risk_adjusted_loss(self, returns: np.ndarray,
                                    predictions: np.ndarray,
                                    risk_penalties: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate risk-adjusted loss metrics.

        Args:
            returns: Actual returns
            predictions: Predicted returns
            risk_penalties: Risk penalty parameters

        Returns:
            Dictionary containing risk-adjusted loss metrics
        """
        try:
            if risk_penalties is None:
                risk_penalties = {
                    "downside_penalty": 2.0,
                    "volatility_penalty": 0.5,
                    "drawdown_penalty": 1.5,
                }

            # Basic prediction error
            prediction_error = predictions - returns
            
            # Asymmetric loss (penalize losses more than gains)
            downside_mask = returns < 0
            asymmetric_loss = np.mean(
                prediction_error ** 2 * (1 + downside_mask * risk_penalties["downside_penalty"])
            )
            
            # Volatility penalty
            volatility = np.std(returns)
            volatility_loss = volatility * risk_penalties["volatility_penalty"]
            
            # Drawdown penalty
            cumulative_returns = np.cumprod(1 + returns) - 1
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / (1 + running_max)
            max_drawdown = np.min(drawdown)
            drawdown_loss = abs(max_drawdown) * risk_penalties["drawdown_penalty"]
            
            # Combined risk-adjusted loss
            total_loss = asymmetric_loss + volatility_loss + drawdown_loss

            return {
                "asymmetric_loss": float(asymmetric_loss),
                "volatility_loss": float(volatility_loss),
                "drawdown_loss": float(drawdown_loss),
                "total_risk_adjusted_loss": float(total_loss),
                "max_drawdown": float(max_drawdown),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating risk-adjusted loss: {e}")
            return {}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_regime_aware_loss(self, predictions: np.ndarray,
                                   actuals: np.ndarray,
                                   regimes: np.ndarray) -> Dict[str, Any]:
        """
        Calculate regime-aware loss metrics.

        Args:
            predictions: Model predictions
            actuals: Actual values
            regimes: Market regime labels

        Returns:
            Dictionary containing regime-aware loss metrics
        """
        try:
            unique_regimes = np.unique(regimes)
            regime_losses = {}
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_preds = predictions[regime_mask]
                regime_actuals = actuals[regime_mask]
                
                if len(regime_preds) > 0:
                    regime_mse = np.mean((regime_preds - regime_actuals) ** 2)
                    regime_mae = np.mean(np.abs(regime_preds - regime_actuals))
                    
                    regime_losses[f"regime_{regime}"] = {
                        "mse": float(regime_mse),
                        "mae": float(regime_mae),
                        "count": int(np.sum(regime_mask)),
                    }
            
            # Overall weighted loss
            total_mse = np.mean((predictions - actuals) ** 2)
            total_mae = np.mean(np.abs(predictions - actuals))
            
            return {
                "total_mse": float(total_mse),
                "total_mae": float(total_mae),
                "regime_losses": regime_losses,
                "num_regimes": len(unique_regimes),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating regime-aware loss: {e}")
            return {}