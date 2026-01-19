import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class IRMLinearRegressor(BaseEstimator, RegressorMixin):
    """
    IRM-v1 implementation for Ridge, Huber, and ElasticNet.

    Parameters:
    -----------
    loss_type : str, 'ridge', 'huber', or 'elasticnet'
    alpha : float, overall regularization strength (Ridge/Lasso component)
    l1_ratio : float, mix between L1 and L2 (only for 'elasticnet')
    irm_lambda : float, weight of the Invariant Risk penalty
    huber_epsilon : float, threshold for Huber loss
    max_iter : int, maximum number of iterations for the optimizer
    """
    def __init__(self, loss_type='ridge', alpha=1.0, l1_ratio=0.5,
                 irm_lambda=1.0, huber_epsilon=1.35, max_iter=1000):
        self.loss_type = loss_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.irm_lambda = irm_lambda
        self.huber_epsilon = huber_epsilon
        self.max_iter = max_iter

    def _huber_loss_and_grad(self, w, X, y):
        """Piecewise Huber loss and its gradient."""
        res = y - X @ w
        abs_res = np.abs(res)
        mask = abs_res <= self.huber_epsilon

        # Loss
        loss = np.where(mask, 0.5 * res**2,
                        self.huber_epsilon * (abs_res - 0.5 * self.huber_epsilon))

        # Gradient
        grad = np.where(mask, -res, -self.huber_epsilon * np.sign(res))
        return np.mean(loss), (X.T @ grad) / len(y)

    def _mse_loss_and_grad(self, w, X, y):
        """Standard MSE loss and its gradient."""
        res = y - X @ w
        loss = np.mean(res**2)
        grad = -2 * (X.T @ res) / len(y)
        return loss, grad

    def _objective(self, w, envs):
        total_erm_loss = 0
        irm_penalty = 0

        for X_e, y_e in envs:
            # 1. Calculate loss and gradient for this specific environment
            if self.loss_type == 'huber':
                loss_e, grad_e = self._huber_loss_and_grad(w, X_e, y_e)
            else:
                loss_e, grad_e = self._mse_loss_and_grad(w, X_e, y_e)

            total_erm_loss += loss_e
            # 2. IRM Penalty: Squared norm of the gradient per environment
            irm_penalty += np.sum(grad_e**2)

        # 3. Structural Regularization (Ridge/Lasso/ElasticNet)
        l2_penalty = 0.5 * np.sum(w**2)
        l1_penalty = np.sum(np.abs(w))
        # Note: L1 penalty with L-BFGS-B is not ideal as it's non-differentiable at 0.
        # This implementation follows the provided snippet but convergence to sparse solutions is not guaranteed.

        if self.loss_type == 'ridge':
            reg = self.alpha * l2_penalty
        elif self.loss_type == 'elasticnet':
            reg = self.alpha * (self.l1_ratio * l1_penalty + (1 - self.l1_ratio) * l2_penalty)
        else: # huber (usually L2 regularized)
            reg = self.alpha * l2_penalty

        return (total_erm_loss / len(envs)) + reg + (self.irm_lambda * irm_penalty)

    def fit(self, X, y, env_indices=None):
        """
        X: Training features
        y: Training target
        env_indices: list of lists/arrays, where each sub-list contains
                     the row indices for one 'environment' (e.g. [ [0..100], [101..200] ])
        """
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        # Prepare environment data
        if env_indices is None:
            # Default to single environment if not provided
            env_indices = [np.arange(len(y))]

        envs = [(X[idx], y[idx]) for idx in env_indices]

        # Initial guess (zeros)
        w0 = np.zeros(n_features)

        # Optimize using L-BFGS-B (handles high-dimensional crypto features well)
        res = minimize(self._objective, w0, args=(envs,),
                       method='L-BFGS-B', options={'maxiter': self.max_iter})

        self.coef_ = res.x
        self.intercept_ = 0.0
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

def get_vol_env_indices(sample_weight):
    if sample_weight is None:
        return None
    median_w = np.median(sample_weight)
    high_vol_mask = sample_weight < median_w
    low_vol_mask = sample_weight >= median_w
    if np.sum(high_vol_mask) > 10 and np.sum(low_vol_mask) > 10:
         return [np.where(high_vol_mask)[0], np.where(low_vol_mask)[0]]
    return None
