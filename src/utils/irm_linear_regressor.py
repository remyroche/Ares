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

        # Prevent explosion in penalty
        irm_penalty = np.clip(irm_penalty, 0.0, 1e12)
        
        obj_val = (total_erm_loss / len(envs)) + reg + (self.irm_lambda * irm_penalty)
        if not np.isfinite(obj_val):
             return 1e12 # Penalty for invalid weights
        return obj_val

    def fit(self, X: np.ndarray, y: np.ndarray, env_indices: list[np.ndarray] = None, sample_weight: np.ndarray = None) -> "IRMLinearModel":
        X, y = check_X_y(X, y)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        
        # Prepare environment data
        if env_indices is None or len(env_indices) == 0:
            env_indices = [np.arange(len(y))]
            
        # Convert to float64 for numerical stability in optimizer
        X = X.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        sample_weight = sample_weight.astype(np.float64, copy=False)

        envs = []
        for idx in env_indices:
            envs.append((X[idx], y[idx], sample_weight[idx]))
        
        w0 = np.zeros(X.shape[1], dtype=np.float64)
        
        # Log training start for debugging
        from src.utils.tprint import tprint_info, tprint_warning
        
        # Select JIT implementation
        if self.loss_type == 'huber':
            jit_func = _jit_irm_huber_loss_and_grad
        elif self.loss_type == 'modified_huber':
            jit_func = _jit_irm_mod_huber_loss_and_grad
        else:
             # Fallback or other implementations
             jit_func = None

        def objective_wrapper(w):
            if jit_func:
                return jit_func(w, envs, self.alpha, self.irm_lambda, self.huber_epsilon)
            else:
                return self._objective_python(w, envs)

        try:
            # Looser tolerance (1e-4) for speed, jac=True for analytic gradient
            res = minimize(
                objective_wrapper,
                w0,
                method='L-BFGS-B',
                jac=True if jit_func else False,
                options={'maxiter': self.max_iter, 'ftol': 1e-4, 'gtol': 1e-4}
            )
            self.coef_ = res.x
            if not np.isfinite(self.coef_).all():
                tprint_warning(f"      ⚠️ IRM {self.loss_type} produced non-finite coefficients! Sanitizing.")
                self.coef_ = np.nan_to_num(self.coef_)
        except Exception as e:
            tprint_warning(f"      ❌ IRM {self.loss_type} optimization failed: {e}")
            self.coef_ = w0

        self.intercept_ = 0.0  # Standard for these models in this codebase
        self.is_fitted_ = True
        return self

    def _objective_python(self, w: np.ndarray, envs: list) -> float:
        # Legacy Python implementation for non-JIT types
        total_erm_loss = 0.0
        irm_penalty = 0.0

        for X_e, y_e, w_e in envs:
             # Use the existing python methods
             if self.loss_type == 'logloss':
                 loss_e, grad_e = self._log_loss_and_grad(w, X_e, y_e, w_e)
             else:
                 loss_e, grad_e = self._mse_loss_and_grad(w, X_e, y_e, w_e)
             
             total_erm_loss += loss_e
             irm_penalty += float(np.sum(grad_e**2))

        l2_penalty = 0.5 * np.sum(w**2)
        reg = self.alpha * l2_penalty
        
        irm_penalty = np.clip(irm_penalty, 0.0, 1e12)
        obj_val = (total_erm_loss / len(envs)) + reg + (self.irm_lambda * irm_penalty)
        return obj_val 

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

# --- JIT Compiled Helpers ---
try:
    from numba import jit
    
    @jit(nopython=True, cache=True)
    def _jit_huber_loss_grad_hessian_product(w, X, y, weights, epsilon):
        """
        Computes (Loss, Gradient, Hessian*Vector) for Huber wrt w.
        The Hessian-Vector product is computed efficiently without building H.
        HV_product(v) -> returns H @ v
        """
        res = y - X @ w
        abs_res = np.abs(res)
        mask = abs_res <= epsilon
        
        # Loss
        loss_vec = np.empty_like(res)
        for i in range(len(res)):
            if mask[i]:
                loss_vec[i] = 0.5 * res[i]**2
            else:
                loss_vec[i] = epsilon * (abs_res[i] - 0.5 * epsilon)
        loss = np.sum(loss_vec * weights) / np.sum(weights)
        
        # Gradient
        grad_vec = np.empty_like(res)
        for i in range(len(res)):
            if mask[i]:
                grad_vec[i] = -res[i]
            else:
                grad_vec[i] = -epsilon * np.sign(res[i])
        
        # weighted gradient
        w_grad_vec = weights * grad_vec
        grad = X.T @ w_grad_vec / np.sum(weights)
        
        # For IRM gradient, we need H @ grad.
        # H for Huber is X.T @ diag(weights * mask) @ X / sum(weights)
        # So H @ grad = X.T @ (weights * mask * (X @ grad)) / sum(weights)
        
        X_grad = X @ grad
        w_mask_X_grad = weights * mask.astype(np.float64) * X_grad
        h_grad = X.T @ w_mask_X_grad / np.sum(weights)
        
        return loss, grad, h_grad

    @jit(nopython=True, cache=True)
    def _jit_irm_huber_loss_and_grad(w, envs, alpha, irm_lambda, epsilon):
        total_loss = 0.0
        total_grad = np.zeros_like(w)
        n_envs = len(envs)
        
        # Regularization (L2)
        reg_loss = 0.5 * alpha * np.sum(w**2)
        reg_grad = alpha * w
        
        total_loss += reg_loss
        total_grad += reg_grad
        
        for i in range(n_envs):
            # Tuples in numba list are tricky, assumes homogeneous list of tuples if passed correctly
            # But list of tuples is not fully supported in nopython mode unless typed.
            # We will use envs[i][0] etc.
            X_e = envs[i][0]
            y_e = envs[i][1]
            w_e = envs[i][2]
            
            loss_e, grad_e, h_grad_e = _jit_huber_loss_grad_hessian_product(w, X_e, y_e, w_e, epsilon)
            
            # ERM part
            total_loss += loss_e / n_envs
            total_grad += grad_e / n_envs
            
            # IRM part: lambda * ||grad_e||^2
            # Gradient of IRM part: 2 * lambda * H_e @ grad_e
            sq_norm_grad = np.sum(grad_e**2)
            
            total_loss += irm_lambda * sq_norm_grad
            total_grad += irm_lambda * 2.0 * h_grad_e
            
        return total_loss, total_grad

    @jit(nopython=True, cache=True)
    def _jit_mod_huber_loss_grad_hessian_product(w, X, y, weights):
        # Modified Huber for classification
        # y is 0/1, mapped to -1/1
        y_mapped = 2.0 * y - 1.0
        f = X @ w
        margin = y_mapped * f
        
        loss_vec = np.zeros_like(margin)
        grad_vec = np.zeros_like(margin) # Derivative of loss wrt f
        hess_vec = np.zeros_like(margin) # Second derivative of loss wrt f
        
        # Vectorized logic
        for i in range(len(margin)):
            m = margin[i]
            if m >= -1:
                if m < 1:
                    # Quadratic region
                    # Loss = (1-m)^2 -> max(0, 1-m)^2
                    loss_vec[i] = (1 - m)**2
                    # dLoss/df = d/df (1 - yf)^2 = 2(1-yf)(-y) = -2y(1-m)
                    grad_vec[i] = -2 * y_mapped[i] * (1 - m)
                    # d2Loss/df2 = -2y(-y) = 2y^2 = 2
                    hess_vec[i] = 2.0
                else:
                     # m >= 1 -> Loss = 0
                     loss_vec[i] = 0.0
                     grad_vec[i] = 0.0
                     hess_vec[i] = 0.0
            else:
                # Linear region
                # Loss = -4m
                loss_vec[i] = -4 * m
                # dLoss/df = -4y
                grad_vec[i] = -4 * y_mapped[i]
                # d2Loss/df2 = 0
                hess_vec[i] = 0.0
                
        sum_w = np.sum(weights)
        loss = np.sum(loss_vec * weights) / sum_w
        
        # Grad = X.T @ (weights * grad_vec) / sum_w
        w_grad_vec = weights * grad_vec
        grad = X.T @ w_grad_vec / sum_w
        
        # Hessian-Vector Product: H @ v
        # H = X.T @ diag(weights * hess_vec) @ X / sum_w
        # H @ grad = X.T @ (weights * hess_vec * (X @ grad)) / sum_w
        
        X_grad = X @ grad
        w_h_X_grad = weights * hess_vec * X_grad
        h_grad = X.T @ w_h_X_grad / sum_w
        
        return loss, grad, h_grad

    @jit(nopython=True, cache=True)
    def _jit_irm_mod_huber_loss_and_grad(w, envs, alpha, irm_lambda, epsilon):
        # Epsilon unused for modified huber, kept for signature comp
        total_loss = 0.0
        total_grad = np.zeros_like(w)
        n_envs = len(envs)
        
        # Regularization (L2)
        reg_loss = 0.5 * alpha * np.sum(w**2)
        reg_grad = alpha * w
        
        total_loss += reg_loss
        total_grad += reg_grad
        
        for i in range(n_envs):
            X_e = envs[i][0]
            y_e = envs[i][1]
            w_e = envs[i][2]
            
            loss_e, grad_e, h_grad_e = _jit_mod_huber_loss_grad_hessian_product(w, X_e, y_e, w_e)
            
            total_loss += loss_e / n_envs
            total_grad += grad_e / n_envs
            
            # IRM
            sq_norm_grad = np.sum(grad_e**2)
            total_loss += irm_lambda * sq_norm_grad
            total_grad += irm_lambda * 2.0 * h_grad_e
            
        return total_loss, total_grad

except ImportError:
    # Fallback if numba not present
    _jit_irm_huber_loss_and_grad = None
    _jit_irm_mod_huber_loss_and_grad = None

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
