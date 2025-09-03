"""
Optimization Metrics Calculator Module.

This module handles metrics used for strategy optimization and parameter tuning.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.core.decorators import handles_errors

from .base import PnLLossFunctionsBase


class OptimizationMetricsCalculator(PnLLossFunctionsBase):
    """
    Optimization Metrics Calculator for strategy optimization.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize optimization metrics calculator."""
        super().__init__(config)
        self.enable_optimization_metrics: bool = self.pnl_config.get(
            "enable_optimization_metrics", True
        )

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_kelly_criterion(self, win_rate: float, 
                                 avg_win: float, 
                                 avg_loss: float) -> Dict[str, float]:
        """
        Calculate Kelly Criterion for optimal position sizing.

        Args:
            win_rate: Probability of winning
            avg_win: Average winning amount
            avg_loss: Average losing amount

        Returns:
            Dictionary containing Kelly criterion metrics
        """
        try:
            if avg_loss == 0 or win_rate == 0 or win_rate == 1:
                return {"kelly_fraction": 0.0, "conservative_kelly": 0.0}

            # Basic Kelly formula: f = (p*b - q) / b
            # where p = win_rate, q = 1-p, b = avg_win/avg_loss
            b = avg_win / avg_loss
            q = 1 - win_rate
            
            kelly_fraction = (win_rate * b - q) / b
            
            # Conservative Kelly (25% of full Kelly)
            conservative_kelly = kelly_fraction * 0.25
            
            # Cap at reasonable levels
            kelly_fraction = max(0, min(kelly_fraction, 1.0))
            conservative_kelly = max(0, min(conservative_kelly, 0.25))

            return {
                "kelly_fraction": float(kelly_fraction),
                "conservative_kelly": float(conservative_kelly),
                "win_loss_ratio": float(b),
            }

        except Exception as e:
            self.logger.exception(f"Error calculating Kelly criterion: {e}")
            return {"kelly_fraction": 0.0, "conservative_kelly": 0.0}

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_information_ratio(self, excess_returns: np.ndarray, 
                                   benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.

        Args:
            excess_returns: Strategy returns minus risk-free rate
            benchmark_returns: Benchmark returns

        Returns:
            Information ratio
        """
        try:
            if len(excess_returns) < 2 or len(benchmark_returns) < 2:
                return 0.0

            # Calculate active returns
            active_returns = excess_returns - benchmark_returns
            
            # Calculate tracking error
            tracking_error = np.std(active_returns)
            
            if tracking_error == 0:
                return 0.0

            # Calculate information ratio
            mean_active_return = np.mean(active_returns)
            information_ratio = mean_active_return / tracking_error
            
            # Annualize
            information_ratio *= np.sqrt(252)
            
            return float(information_ratio)

        except Exception as e:
            self.logger.exception(f"Error calculating information ratio: {e}")
            return 0.0

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_omega_ratio(self, returns: np.ndarray, 
                             threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.

        Args:
            returns: Array of returns
            threshold: Threshold return level

        Returns:
            Omega ratio
        """
        try:
            if len(returns) < 2:
                return 0.0

            # Separate returns above and below threshold
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns <= threshold]
            
            # Calculate sums
            sum_gains = np.sum(gains) if len(gains) > 0 else 0.0
            sum_losses = np.sum(losses) if len(losses) > 0 else 0.0
            
            if sum_losses == 0:
                return float('inf') if sum_gains > 0 else 0.0
            
            omega_ratio = sum_gains / sum_losses
            return float(omega_ratio)

        except Exception as e:
            self.logger.exception(f"Error calculating Omega ratio: {e}")
            return 0.0

    @handles_errors(
        exceptions=(ValueError, KeyError, TypeError),
        default_return={},
    )
    def calculate_stability_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate strategy stability metrics.

        Args:
            returns: Array of returns

        Returns:
            Dictionary containing stability metrics
        """
        try:
            if len(returns) < 10:
                return {
                    "return_stability": 0.0,
                    "volatility_stability": 0.0,
                    "consistency_score": 0.0,
                }

            # Split returns into windows
            window_size = max(20, len(returns) // 10)
            n_windows = len(returns) // window_size
            
            window_means = []
            window_stds = []
            
            for i in range(n_windows):
                start = i * window_size
                end = (i + 1) * window_size
                window_returns = returns[start:end]
                
                window_means.append(np.mean(window_returns))
                window_stds.append(np.std(window_returns))
            
            # Calculate stability metrics
            return_stability = 1 - (np.std(window_means) / (np.mean(np.abs(window_means)) + 1e-6))
            volatility_stability = 1 - (np.std(window_stds) / (np.mean(window_stds) + 1e-6))
            
            # Consistency score (how often returns are positive)
            positive_returns = np.sum(returns > 0)
            consistency_score = positive_returns / len(returns)
            
            return {
                "return_stability": float(max(0, return_stability)),
                "volatility_stability": float(max(0, volatility_stability)),
                "consistency_score": float(consistency_score),
                "windows_analyzed": n_windows,
            }

        except Exception as e:
            self.logger.exception(f"Error calculating stability metrics: {e}")
            return {
                "return_stability": 0.0,
                "volatility_stability": 0.0,
                "consistency_score": 0.0,
            }