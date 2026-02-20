from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

class HuberNNLS(BaseEstimator, RegressorMixin):
    """Custom Huber Regressor with Non-Negative Least Squares constraint.

    Optimizes:
        L = Mean(HuberLoss(y - y_pred)) + 0.5 * alpha * ||w||^2
        Subject to: w >= 0 (NNLS)
    """
    def __init__(self, alpha=1.0, delta=1.35, fit_intercept=True):
        self.alpha = alpha  # L2 regularization strength
        self.delta = delta  # Huber threshold
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            sample_weight = sample_weight / np.mean(sample_weight)  # Normalize

        n_samples, n_features = X.shape

        # Initial guess (Ridge) - warm start
        ridge = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        ridge.fit(X, y, sample_weight=sample_weight)

        if self.fit_intercept:
            w0 = np.concatenate([ridge.coef_, [ridge.intercept_]])
        else:
            w0 = ridge.coef_

        # Objective function
        def objective(w):
            if self.fit_intercept:
                coef = w[:-1]
                intercept = w[-1]
            else:
                coef = w
                intercept = 0.0

            y_pred = X @ coef + intercept
            residual = y - y_pred

            # Huber loss
            abs_r = np.abs(residual)
            mask = abs_r <= self.delta
            loss = np.empty_like(abs_r)
            loss[mask] = 0.5 * residual[mask]**2
            loss[~mask] = self.delta * (abs_r[~mask] - 0.5 * self.delta)

            # Weighted mean loss
            total_loss = np.sum(loss * sample_weight) / np.sum(sample_weight)

            # L2 Regularization (on coef only)
            reg = 0.5 * self.alpha * np.sum(coef**2)

            return total_loss + reg

        # Gradient
        def gradient(w):
            if self.fit_intercept:
                coef = w[:-1]
                intercept = w[-1]
            else:
                coef = w
                intercept = 0.0

            y_pred = X @ coef + intercept
            residual = y_pred - y  # Gradient w.r.t prediction (dLoss/dPred)

            # Derivative of Huber loss w.r.t prediction
            # L = 0.5*r^2 (r=y-p) -> dL/dp = p-y
            grad_pred = np.empty_like(residual)
            mask = np.abs(residual) <= self.delta # Note: using residual (p-y) magnitude is same as |y-p|
            grad_pred[mask] = residual[mask]
            grad_pred[~mask] = self.delta * np.sign(residual[~mask])

            # Weighted gradient
            grad_pred *= sample_weight
            grad_pred /= np.sum(sample_weight)

            # dL/dw = X.T @ dL/dp
            grad_w = X.T @ grad_pred

            # Add L2 reg gradient
            grad_w += self.alpha * coef

            if self.fit_intercept:
                grad_intercept = np.sum(grad_pred)
                return np.concatenate([grad_w, [grad_intercept]])
            else:
                return grad_w

        # Bounds: Non-negative for coefs, unconstrained for intercept
        bounds = [(0.0, None)] * n_features
        if self.fit_intercept:
            bounds.append((None, None))

        res = minimize(
            objective,
            w0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-6}
        )

        if self.fit_intercept:
            self.coef_ = res.x[:-1]
            self.intercept_ = res.x[-1]
        else:
            self.coef_ = res.x
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_
