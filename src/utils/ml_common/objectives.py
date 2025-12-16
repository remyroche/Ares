"""
Custom LightGBM Objectives for Diversity Defense

This module implements custom objective functions (Sharpe, Tanh, Huber)
for LightGBM training, designed to encourage model diversity.
"""

import numpy as np
from typing import Tuple, Callable, Any, Optional

def get_sharpe_objective(lambda_reg: float = 1.0) -> Callable:
    """
    Returns a custom Sharpe Ratio objective function for LightGBM.

    The objective maximizes the Sharpe Ratio of the strategy PnL.
    Loss = - (Mean(PnL) / Std(PnL)) + Regularization

    Args:
        lambda_reg: Regularization strength to penalize extreme weights/signals.
    """
    def sharpe_loss(preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label()
        if hasattr(train_data, 'get_weight'):
            weights = train_data.get_weight()
            if weights is None:
                weights = np.ones_like(y_true)
        else:
            weights = np.ones_like(y_true)

        s = np.tanh(preds)
        gs = 1 - s**2  # derivative of tanh

        pnl = s * y_true * weights

        # Stats
        n = np.sum(weights)
        sum_pnl = np.sum(pnl)
        sum_pnl2 = np.sum(pnl**2)

        mean_pnl = sum_pnl / n
        mean_pnl2 = sum_pnl2 / n
        var_pnl = mean_pnl2 - mean_pnl**2
        std_pnl = np.sqrt(var_pnl + 1e-6)

        dMean_ds = (y_true * weights) / n
        dMean2_ds = (2 * s * (y_true**2) * weights) / n
        dVar_ds = dMean2_ds - 2 * mean_pnl * dMean_ds
        dStd_ds = 0.5 * (var_pnl + 1e-6)**(-0.5) * dVar_ds

        dSharpe_ds = (dMean_ds * std_pnl - mean_pnl * dStd_ds) / (var_pnl + 1e-6)

        # We want to minimize Loss = -Sharpe
        # So grad = -dSharpe/dpreds

        grad = -dSharpe_ds * gs

        # Add L2 regularization on predictions to prevent explosion
        grad += lambda_reg * preds

        # Hessian approximation
        hess = np.abs(grad) + lambda_reg

        return grad, hess

    return sharpe_loss

def get_tanh_objective(scale: float = 1.0) -> Callable:
    """
    Returns a Tanh objective function for LightGBM.
    Loss = log(cosh(scale * (p - y)))
    This is Log-Cosh loss, robust to outliers.

    Args:
        scale: Scaling factor for the error.
    """
    def tanh_loss(preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label()
        residual = preds - y_true
        x = residual * scale

        # Gradient
        grad = np.tanh(x) * scale

        # Hessian
        hess = (scale ** 2) * (1 - np.tanh(x)**2)

        return grad, hess

    return tanh_loss

def get_huber_objective(delta: float = 1.0, alpha_asym: float = 0.5) -> Callable:
    """
    Returns a Huber objective function for LightGBM, optionally asymmetric.

    Args:
        delta: The threshold at which loss transitions from quadratic to linear.
        alpha_asym: Asymmetry parameter (quantile). 0.5 = symmetric.
    """
    def huber_loss(preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label()
        res = preds - y_true
        abs_res = np.abs(res)

        # Quadratic region
        quad_mask = abs_res <= delta
        lin_mask = ~quad_mask

        grad = np.zeros_like(res)
        hess = np.zeros_like(res)

        # Quadratic gradients
        grad[quad_mask] = res[quad_mask]
        hess[quad_mask] = 1.0

        # Linear gradients
        grad[lin_mask] = delta * np.sign(res[lin_mask])
        hess[lin_mask] = 0.0  # Actually 0

        # Asymmetry
        if alpha_asym != 0.5:
            # Standard quantile weighting
            # If res > 0 (overprediction), weight 1-alpha
            # If res < 0 (underprediction), weight alpha
            weights = np.where(res > 0, 1.0 - alpha_asym, alpha_asym)
            grad *= 2 * weights
            hess *= 2 * weights

        return grad, hess

    return huber_loss

def get_binary_brier_objective() -> Callable:
    """
    Returns Brier score objective (MSE on probabilities) for classification.
    L = (p - y)^2
    """
    def brier_loss(preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label()
        # Preds are log-odds (margin). Convert to prob.
        p = 1.0 / (1.0 + np.exp(-preds))

        # Loss = (p - y)^2
        # dL/dp = 2(p - y)
        # dp/dz = p(1-p)  (where z is log-odds)
        # dL/dz = 2(p-y) * p(1-p)

        grad = 2 * (p - y_true) * p * (1 - p)
        hess = p * (1 - p)

        return grad, hess

    return brier_loss

def get_fair_objective(c: float = 1.0) -> Callable:
    """
    Returns a Fair Loss objective function for LightGBM.
    Fair Loss: c^2 * ( |x|/c - log(1 + |x|/c) )
    Gradient: x / (1 + |x|/c)
    Hessian: 1 / (1 + |x|/c)^2

    Robust to outliers, smooth at 0.
    """
    def fair_loss(preds: np.ndarray, train_data: Any) -> Tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label()
        residual = preds - y_true
        abs_res = np.abs(residual)

        grad = residual / (1.0 + abs_res / c)
        hess = 1.0 / ((1.0 + abs_res / c) ** 2)

        return grad, hess

    return fair_loss
