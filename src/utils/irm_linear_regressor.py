import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class IRMLinearModel(BaseEstimator):
    """
    IRM-v1 implementation for Linear Models (Regression & Classification).
    Supports Ridge, Huber, ElasticNet, LogLoss, and Modified Huber.

    Parameters:
    loss_type : str, 'ridge', 'huber', 'elasticnet', 'logloss', or 'modified_huber'
    alpha : float, overall regularization strength
    l1_ratio : float, mix between L1 and L2 (only for 'elasticnet')
    irm_lambda : float, weight of the Invariant Risk penalty
    huber_epsilon : float, threshold for Huber loss
    max_iter : int, maximum iterations for optimizer
    """

    def __init__(
        self,
        loss_type: str = 'ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        irm_lambda: float = 1.0,
        huber_epsilon: float = 1.35,
        max_iter: int = 1000
    ):
        self.loss_type = loss_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.irm_lambda = irm_lambda
        self.huber_epsilon = huber_epsilon
        self.max_iter = max_iter

    def _huber_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Piecewise Huber loss and its gradient."""
        res = y - X @ w
        abs_res = np.abs(res)
        mask = abs_res <= self.huber_epsilon

        # Weighted loss
        loss_vec = np.where(
            mask,
            0.5 * res**2,
            self.huber_epsilon * (abs_res - 0.5 * self.huber_epsilon)
        )
        loss = np.average(loss_vec, weights=weights)

        # Weighted gradient
        grad_vec = np.where(mask, -res, -self.huber_epsilon * np.sign(res))
        grad = (X.T @ (weights * grad_vec)) / np.sum(weights)

        return float(loss), grad

    def _modified_huber_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Modified Huber loss for classification (robust to outliers)."""
        y_mapped = 2.0 * y - 1.0
        f = X @ w
        margin = y_mapped * f

        mask_quad = margin >= -1
        loss_vec = np.where(mask_quad, np.maximum(0, 1 - margin)**2, -4 * margin)
        loss = np.average(loss_vec, weights=weights)

        grad_f = np.where(mask_quad, np.where(margin < 1, -2 * y_mapped * (1 - margin), 0), -4 * y_mapped)
        grad = (X.T @ (weights * grad_f)) / np.sum(weights)

        return float(loss), grad

    def _mse_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Standard MSE loss and its gradient."""
        res = y - X @ w
        loss = np.average(res**2, weights=weights)
        grad = -2 * (X.T @ (weights * res)) / np.sum(weights)
        return float(loss), grad

    def _log_loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Log Loss (Logistic Regression) and its gradient."""
        logits = X @ w
        p = expit(logits)
        epsilon = 1e-15
        p_safe = np.clip(p, epsilon, 1 - epsilon)
        log_loss_vec = - (y * np.log(p_safe) + (1 - y) * np.log(1 - p_safe))
        loss = np.average(log_loss_vec, weights=weights)
        grad = (X.T @ (weights * (p - y))) / np.sum(weights)
        return float(loss), grad

    def _objective(self, w: np.ndarray, envs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
        total_erm_loss = 0.0
        irm_penalty = 0.0

        for X_e, y_e, w_e in envs:
            if self.loss_type == 'huber':
                loss_e, grad_e = self._huber_loss_and_grad(w, X_e, y_e, w_e)
            elif self.loss_type == 'modified_huber':
                loss_e, grad_e = self._modified_huber_loss_and_grad(w, X_e, y_e, w_e)
            elif self.loss_type == 'logloss':
                loss_e, grad_e = self._log_loss_and_grad(w, X_e, y_e, w_e)
            else:
                loss_e, grad_e = self._mse_loss_and_grad(w, X_e, y_e, w_e)

            total_erm_loss += loss_e
            irm_penalty += float(np.sum(grad_e**2))

        l2_penalty = 0.5 * np.sum(w**2)
        l1_penalty = np.sum(np.abs(w))
        reg = self.alpha * l2_penalty
        if self.loss_type == 'elasticnet':
             reg = self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty)

        return (total_erm_loss / len(envs)) + reg + (self.irm_lambda * irm_penalty)

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray] = None, sample_weight: np.ndarray = None) -> "IRMLinearModel":
        X, y = check_X_y(X, y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        
        # Prepare environment data
        if env_indices is None or len(env_indices) == 0:
            # Default to single environment (ERM)
            env_indices = [np.arange(len(y))]
            
        envs = [(X[idx], y[idx], sample_weight[idx]) for idx in env_indices]
        w0 = np.zeros(X.shape[1])
        res = minimize(self._objective, w0, args=(envs,), method='L-BFGS-B', options={'maxiter': self.max_iter})
        self.coef_ = res.x
        self.intercept_ = 0.0 # Standard for these models in this codebase
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

class IRMLinearRegressor(IRMLinearModel, RegressorMixin):
    """Regressor wrapper for IRMLinearModel."""
    pass

class IRMLinearClassifier(IRMLinearModel, ClassifierMixin):
    """Classifier wrapper for IRMLinearModel."""
    def __init__(self, loss_type: str = "modified_huber", alpha: float = 1.0, l1_ratio: float = 0.5, irm_lambda: float = 1.0, max_iter: int = 1000):
        super().__init__(loss_type=loss_type, alpha=alpha, l1_ratio=l1_ratio, irm_lambda=irm_lambda, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray] = None, sample_weight: np.ndarray = None) -> "IRMLinearClassifier":
        super().fit(X, y, env_indices, sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits."""
        return super().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.decision_function(X)
        if self.loss_type == 'logloss':
            pos = expit(logits)
        elif self.loss_type == 'modified_huber':
            pos = np.clip((1.0 + logits) / 2.0, 0.0, 1.0)
        else:
            pos = np.clip(logits, 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def get_vol_env_indices(sample_weight):
    if sample_weight is None:
        return None
    median_w = np.median(sample_weight)
    high_vol_mask = sample_weight < median_w
    low_vol_mask = sample_weight >= median_w
    if np.sum(high_vol_mask) > 10 and np.sum(low_vol_mask) > 10:
         return [np.where(high_vol_mask)[0], np.where(low_vol_mask)[0]]
    return None
