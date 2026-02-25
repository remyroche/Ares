from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def safe_clip_proba(p, eps: float = 1e-6):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def logit(p, eps: float = 1e-6):
    p = safe_clip_proba(p, eps=eps)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def compute_prevalences(y, w=None):
    y = np.asarray(y, dtype=np.float64)
    p_unweighted = float(np.mean(y)) if y.size else 0.5
    if w is None:
        return p_unweighted, p_unweighted
    w = np.asarray(w, dtype=np.float64)
    den = float(np.sum(w))
    if den <= 1e-12:
        return p_unweighted, p_unweighted
    p_weighted = float(np.sum(w * y) / den)
    return p_unweighted, p_weighted


def compute_logit_shift(p_unweighted: float, p_weighted: float, eps: float = 1e-6):
    return float(logit(p_unweighted, eps=eps) - logit(p_weighted, eps=eps))


def apply_logit_shift(p_raw, delta_logit: float, eps: float = 1e-6):
    p_raw = safe_clip_proba(p_raw, eps=eps)
    z = logit(p_raw, eps=eps) + float(delta_logit)
    return safe_clip_proba(sigmoid(z), eps=eps)


class TemperatureScaling:
    """Post-hoc probability calibration via Temperature Scaling.

    Optimizes a single scalar T > 0 such that softmax(logits / T) minimizes NLL.
    Supports binary and multiclass.
    """
    def __init__(self, t_init: float = 1.0):
        self.temperature = float(t_init)

    def fit(self, probs: np.ndarray, y_true: np.ndarray):
        """Fit temperature T on validation set.

        Args:
            probs: (N, K) uncalibrated probabilities from base model.
            y_true: (N,) integer class labels or (N, K) one-hot.
        """
        probs = np.asarray(probs, dtype=np.float64)
        y_true = np.asarray(y_true)

        # Clip for numerical stability
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)

        # Recover logits
        # For binary (N,), treat as (N, 2)
        if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
            p1 = probs.reshape(-1)
            probs = np.column_stack([1.0 - p1, p1])

        logits = np.log(probs)

        # Handle 1D y_true
        if y_true.ndim == 1:
            # Ensure valid class indices
            valid_mask = (y_true >= 0) & (y_true < probs.shape[1])
            logits = logits[valid_mask]
            y_true = y_true[valid_mask]
            if len(y_true) < 10:
                return self

        def nll_fn(t):
            t_val = t[0]
            if t_val <= 0:
                return 1e9

            # Apply temperature
            scaled_logits = logits / t_val

            # Softmax
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            exp_l = np.exp(scaled_logits - max_l)
            sum_exp = np.sum(exp_l, axis=1, keepdims=True)
            log_sum_exp = np.log(sum_exp).flatten() + max_l.flatten()

            # Cross-Entropy Loss
            # For integer labels:
            if y_true.ndim == 1:
                # Select logit corresponding to true class
                true_logits = scaled_logits[np.arange(len(y_true)), y_true.astype(int)]
                loss = -np.mean(true_logits - log_sum_exp)
            else:
                # One-hot
                log_probs = scaled_logits - log_sum_exp[:, None]
                loss = -np.mean(np.sum(y_true * log_probs, axis=1))

            return loss

        res = minimize(
            nll_fn,
            x0=np.array([1.0]),
            bounds=[(0.1, 5.0)],
            method='L-BFGS-B'
        )

        if res.success:
            self.temperature = float(res.x[0])

        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling."""
        probs = np.asarray(probs, dtype=np.float64)
        eps = 1e-12
        probs = np.clip(probs, eps, 1.0 - eps)

        # Handle binary case
        flatten = False
        if probs.ndim == 1:
            p1 = probs
            probs = np.column_stack([1.0 - p1, p1])
            flatten = True
        elif probs.shape[1] == 1:
            p1 = probs.reshape(-1)
            probs = np.column_stack([1.0 - p1, p1])
            flatten = True

        logits = np.log(probs)
        scaled_logits = logits / self.temperature

        # Softmax
        max_l = np.max(scaled_logits, axis=1, keepdims=True)
        exp_l = np.exp(scaled_logits - max_l)
        calibrated = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        if flatten:
            return calibrated[:, 1]

        return calibrated
