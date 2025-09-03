"""
Risk Metrics Calculator Module.

This module handles various risk metric calculations including VaR, CVaR,
and other risk measures.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.core.decorators import handles_errors
from .base import PnLLossFunctionsBase


class RiskMetricsCalculator(PnLLossFunctionsBase):
    """
    Risk Metrics Calculator for computing various risk measures.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize risk metrics calculator."""
        super().__init__(config)
        self.enable_risk_metrics: bool = self.pnl_config.get(
            "enable_risk_metrics", True
        )

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

        Returns:
            Dictionary containing VaR metrics
        """
        try:
            if len(returns) < 10:  # Need sufficient data
                return {"var": 0.0, "confidence_level": confidence_level}

            # Calculate percentile
            var_percentile = (1 - confidence_level) * 100
            var = np.percentile(returns, var_percentile)

            return {
                "var": float(abs(var)),
                "confidence_level": confidence_level,
                "percentile": var_percentile,
            }

        except Exception as e:
            self.logger.exception(f"Error calculating VaR: {e}")
            return {"var": 0.0, "confidence_level": confidence_level}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            Dictionary containing CVaR metrics
        """
        try:
            if len(returns) < 10:
                return {"cvar": 0.0, "confidence_level": confidence_level}

            # Calculate VaR threshold
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(returns, var_percentile)

            # Calculate CVaR as mean of returns below VaR threshold
            tail_returns = returns[returns <= var_threshold]
            cvar = np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold

            return {
                "cvar": float(abs(cvar)),
                "var_threshold": float(var_threshold),
                "confidence_level": confidence_level,
                "tail_observations": len(tail_returns),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating CVaR: {e}")
            return {"cvar": 0.0, "confidence_level": confidence_level}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_tail_risk(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate tail risk metrics.

        Args:
            returns: Array of returns

        Returns:
            Dictionary containing tail risk metrics
        """
        try:
            if len(returns) < 30:
                return {
                    "tail_ratio": 0.0,
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                }

            # Calculate percentiles
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)

            # Tail ratio (upside potential vs downside risk)
            tail_ratio = abs(p95) / abs(p5) if p5 != 0 else 0.0

            # Calculate higher moments
            mean_return = np.mean(returns)
            std_return = np.std(returns)

            if std_return > 0:
                standardized_returns = (returns - mean_return) / std_return
                skewness = np.mean(standardized_returns ** 3)
                kurtosis = np.mean(standardized_returns ** 4) - 3  # Excess kurtosis
            else:
                skewness = 0.0
                kurtosis = 0.0

            return {
                "tail_ratio": float(tail_ratio),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "p95": float(p95),
                "p5": float(p5),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating tail risk: {e}")
            return {
                "tail_ratio": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_risk_budget(self, portfolio_weights: np.ndarray, 
                            asset_covariances: np.ndarray) -> Dict[str, Any]:
        """
        Calculate risk budget allocation.

        Args:
            portfolio_weights: Array of portfolio weights
            asset_covariances: Covariance matrix of assets

        Returns:
            Dictionary containing risk budget metrics
        """
        try:
            if len(portfolio_weights) == 0 or asset_covariances.shape[0] == 0:
                return {"risk_contributions": [], "total_risk": 0.0}

            # Calculate portfolio variance
            portfolio_variance = np.dot(portfolio_weights.T, 
                                      np.dot(asset_covariances, portfolio_weights))
            portfolio_risk = np.sqrt(portfolio_variance)

            # Calculate marginal risk contributions
            marginal_contributions = np.dot(asset_covariances, portfolio_weights) / portfolio_risk

            # Calculate risk contributions
            risk_contributions = portfolio_weights * marginal_contributions

            # Normalize to get risk budget percentages
            risk_budget_pct = risk_contributions / portfolio_risk

            return {
                "risk_contributions": risk_contributions.tolist(),
                "risk_budget_pct": risk_budget_pct.tolist(),
                "total_risk": float(portfolio_risk),
                "marginal_contributions": marginal_contributions.tolist(),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating risk budget: {e}")
            return {"risk_contributions": [], "total_risk": 0.0}