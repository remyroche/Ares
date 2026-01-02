import numpy as np

# In orthogonal_label_generation.py

def get_focal_loss_lgbm(alpha, gamma):
    """
    Factory to return a custom Focal Loss objective function for LightGBM.
    Args:
        alpha (float): Balancing factor. alpha for class 1, (1-alpha) for class 0.
        gamma (float): Focusing parameter.
    """
    def focal_loss_objective(y_pred, train_data):
        y_true = train_data.get_label()
        # Robust sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Gradients
        # For y=1: alpha * (1-p)^gamma * (gamma * p * log(p) + p - 1)
        # For y=0: (1-alpha) * p^gamma * (-gamma * (1-p) * log(1-p) + p)
        
        # Add epsilon to logs
        epsilon = 1e-9
        log_p = np.log(p + epsilon)
        log_1_p = np.log(1.0 - p + epsilon)
        
        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(y_pred)
        
        # Vectorized calculation
        # Term 1 (Positive samples)
        pos_mask = (y_true == 1)
        if pos_mask.any():
            p_pos = p[pos_mask]
            term_1 = alpha * np.power(1 - p_pos, gamma) * (gamma * p_pos * log_p[pos_mask] + p_pos - 1)
            grad[pos_mask] = term_1
            # Hessian approximation
            hess[pos_mask] = p_pos * (1 - p_pos) * alpha * np.power(1 - p_pos, gamma) # scaled
            
        # Term 0 (Negative samples)
        neg_mask = (y_true == 0)
        if neg_mask.any():
            p_neg = p[neg_mask]
            term_0 = (1 - alpha) * np.power(p_neg, gamma) * (-gamma * (1 - p_neg) * log_1_p[neg_mask] + p_neg)
            grad[neg_mask] = term_0
            hess[neg_mask] = p_neg * (1 - p_neg) * (1 - alpha) * np.power(p_neg, gamma)

        return grad, hess
        
    return focal_loss_objective

def get_focal_loss_xgb(alpha, gamma):
    """
    Factory to return a custom Focal Loss objective function for XGBoost.
    Args:
        alpha (float): Balancing factor. alpha for class 1, (1-alpha) for class 0.
        gamma (float): Focusing parameter.
    """
    def focal_loss_objective(y_pred, dtrain):
        y_true = dtrain.get_label()
        # XGBoost output is margin (logit)
        # Robust sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Add epsilon to logs
        epsilon = 1e-9
        log_p = np.log(p + epsilon)
        log_1_p = np.log(1.0 - p + epsilon)
        
        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(y_pred)
        
        # Vectorized calculation
        # Term 1 (Positive samples)
        pos_mask = (y_true == 1)
        if pos_mask.any():
            p_pos = p[pos_mask]
            term_1 = alpha * np.power(1 - p_pos, gamma) * (gamma * p_pos * log_p[pos_mask] + p_pos - 1)
            grad[pos_mask] = term_1
            hess[pos_mask] = p_pos * (1 - p_pos) * alpha * np.power(1 - p_pos, gamma)
            
        # Term 0 (Negative samples)
        neg_mask = (y_true == 0)
        if neg_mask.any():
            p_neg = p[neg_mask]
            term_0 = (1 - alpha) * np.power(p_neg, gamma) * (-gamma * (1 - p_neg) * log_1_p[neg_mask] + p_neg)
            grad[neg_mask] = term_0
            hess[neg_mask] = p_neg * (1 - p_neg) * (1 - alpha) * np.power(p_neg, gamma)

        return grad, hess
        
    return focal_loss_objective

class RobustFocalLoss:
    """
    Advanced Robust Focal Loss for LightGBM with adaptive alpha and asymmetric gamma.
    
    Features:
    1. Adaptive Alpha: Automatically balances classes based on batch statistics if alpha is None.
    2. Asymmetric Gamma: Different focusing parameters for positives (gamma_pos) and negatives (gamma_neg).
    3. Label Smoothing: Softens hard labels to prevent overfitting.
    4. Mix Ratio: Blends Focal Loss with standard Log Loss for robustness (mix * LogLoss + (1-mix) * FocalLoss).
    """
    def __init__(self, gamma_pos=1.0, gamma_neg=1.0, alpha=None, mix=0.0, label_smoothing=0.0, verbose=False):
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.mix = mix
        self.label_smoothing = label_smoothing
        self.verbose = verbose
        self.name = "robust_focal_loss"
        
    def __call__(self, y_pred, dataset):
        y_true = dataset.get_label()
        
        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        # Robust sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Adaptive Alpha Calculation
        # If alpha is None, calculate it based on current batch ratio
        if self.alpha is None:
            pos_ratio = np.mean(y_true)
            # Clip for safety
            pos_ratio = np.clip(pos_ratio, 0.001, 0.999)
            
            # Logic: If positives are rare (<0.5), we want high alpha to boost them.
            # If negatives are rare (>0.5), we want low alpha (high 1-alpha) to boost them.
            if pos_ratio < 0.5:
                # Rare positives: boost alpha
                # e.g. ratio=0.1 -> dist=0.4 -> alpha=0.5 + 0.4*0.8 = 0.82
                current_alpha = 0.5 + (0.5 - pos_ratio) * 0.8
            else:
                # Rare negatives: reduce alpha
                # e.g. ratio=0.9 -> dist=0.4 -> alpha=0.5 - 0.4*0.8 = 0.18
                current_alpha = 0.5 - (pos_ratio - 0.5) * 0.8
        else:
            current_alpha = self.alpha
            
        # Epsilon for numerical stability
        epsilon = 1e-9
        log_p = np.log(p + epsilon)
        log_1_p = np.log(1.0 - p + epsilon)
        
        grad = np.zeros_like(y_pred)
        hess = np.zeros_like(y_pred)
        
        # --- Focal Loss Components ---
        # Positives (y=1)
        pos_mask = (y_true > 0.5) # Handle soft labels
        if pos_mask.any():
            p_pos = p[pos_mask]
            # pt = p, (1-pt) = 1-p
            term_pos = current_alpha * np.power(1 - p_pos, self.gamma_pos)
            
            # Gradient: dL/dp * dp/dx
            # dL/dp = -alpha * (1-p)^gamma / p + alpha * gamma * (1-p)^(gamma-1) * log(p)
            # but simpler formula for logits: alpha * (1-p)^gamma * (gamma * p * log(p) + p - 1)
            grad_focal_pos = term_pos * (self.gamma_pos * p_pos * log_p[pos_mask] + p_pos - 1)
            
            # Hessian approx
            hess_focal_pos = p_pos * (1 - p_pos) * term_pos # Simplified scaling
            
            grad[pos_mask] += (1 - self.mix) * grad_focal_pos
            hess[pos_mask] += (1 - self.mix) * hess_focal_pos

        # Negatives (y=0)
        neg_mask = (y_true <= 0.5)
        if neg_mask.any():
            p_neg = p[neg_mask]
            # pt = 1-p, (1-pt) = p
            term_neg = (1 - current_alpha) * np.power(p_neg, self.gamma_neg)
            
            # Gradient for negatives
            grad_focal_neg = term_neg * (-self.gamma_neg * (1 - p_neg) * log_1_p[neg_mask] + p_neg)
            
            # Hessian approx
            hess_focal_neg = p_neg * (1 - p_neg) * term_neg
            
            grad[neg_mask] += (1 - self.mix) * grad_focal_neg
            hess[neg_mask] += (1 - self.mix) * hess_focal_neg
            
        # --- Log Loss Components (Mix) ---
        if self.mix > 0:
            # Standard LogLoss Gradient = p - y
            grad_log = p - y_true
            hess_log = p * (1 - p)
            
            grad += self.mix * grad_log
            hess += self.mix * hess_log
            
        return grad, hess


# ==========================================
# REGRESSION LOSS FUNCTIONS (NEW)
# ==========================================

def get_focal_regression_loss_lgbm(alpha=1.0, gamma=2.0):
    '''
    Focal loss adapted for regression - focuses on large prediction errors.
    
    Args:
        alpha (float): Base weighting factor
        gamma (float): Focusing parameter (higher = more focus on hard examples)
    '''
    def focal_regression_objective(y_pred, train_data):
        y_true = train_data.get_label()
        residual = y_pred - y_true
        abs_residual = np.abs(residual)
        
        # Normalize residuals for weighting (0-1 range)
        max_residual = np.max(abs_residual)
        if max_residual > 0:
            normalized_residual = abs_residual / max_residual
        else:
            normalized_residual = abs_residual
        
        # Focal weighting: emphasize hard examples (large residuals)
        # (1 - normalized_residual)^gamma where normalized_residual is close to 1 for large errors
        focal_weight = np.power(1.0 - normalized_residual, gamma)
        
        # Weighted MSE gradient and hessian
        grad = 2.0 * residual * focal_weight * alpha
        hess = 2.0 * focal_weight * alpha
        
        return grad, hess
    
    return focal_regression_objective

def get_quantile_loss_lgbm(quantile=0.5):
    '''
    Quantile loss (Pinball loss) for regression.
    
    Args:
        quantile (float): Quantile to predict (0.0-1.0)
                        0.5 = median, 0.9 = 90th percentile, etc.
    '''
    def quantile_loss_objective(y_pred, train_data):
        y_true = train_data.get_label()
        residual = y_pred - y_true
        
        # Pinball loss gradient
        grad = np.where(residual > 0, quantile, quantile - 1.0)
        hess = np.ones_like(y_pred)  # Constant hessian for quantile loss
        
        return grad, hess
    
    return quantile_loss_objective

def get_huber_loss_lgbm(delta=1.0):
    '''
    Huber loss for regression - quadratic near zero, linear far from zero.
    
    Args:
        delta (float): Transition point between quadratic and linear regions
    '''
    def huber_loss_objective(y_pred, train_data):
        y_true = train_data.get_label()
        residual = y_pred - y_true
        abs_residual = np.abs(residual)
        
        # Quadratic for small errors, linear for large errors
        grad = np.where(abs_residual <= delta, residual, delta * np.sign(residual))
        hess = np.where(abs_residual <= delta, np.ones_like(y_true), np.zeros_like(y_true))
        
        return grad, hess
    
    return huber_loss_objective

def get_adaptive_mse_loss_lgbm(focusing_factor=2.0):
    '''
    MSE with adaptive weighting based on error magnitude.
    
    Args:
        focusing_factor (float): How much to focus on hard examples
    '''
    def adaptive_mse_objective(y_pred, train_data):
        y_true = train_data.get_label()
        residual = y_pred - y_true
        abs_residual = np.abs(residual)
        
        # Focus on hard examples (large residuals)
        median_abs_residual = np.median(abs_residual)
        if median_abs_residual > 0:
            weights = np.power(abs_residual / median_abs_residual, focusing_factor)
        else:
            weights = np.ones_like(abs_residual)
        
        # Weighted MSE
        grad = 2.0 * residual * weights
        hess = 2.0 * weights
        
        return grad, hess
    
    return adaptive_mse_objective


def get_asymmetric_regression_loss_lgbm(alpha_pos=0.7, alpha_neg=0.3, huber_delta=0.1):
    """
    Factory to return an asymmetric regression loss function for LightGBM.
    Penalizes underestimation and overestimation differently.
    
    Args:
        alpha_pos (float): Weight for positive errors (underestimation)
        alpha_neg (float): Weight for negative errors (overestimation)  
        huber_delta (float): Delta parameter for Huber loss transition
    """
    def asymmetric_regression_objective(y_pred, train_data):
        y_true = train_data.get_label()
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Asymmetric Huber-like loss
        grad = np.zeros_like(residuals)
        hess = np.zeros_like(residuals)
        
        # Positive residuals (underestimation) - penalize more
        pos_mask = residuals > 0
        if pos_mask.any():
            r_pos = residuals[pos_mask]
            if r_pos.any() > huber_delta:
                # Quadratic region
                grad[pos_mask] = alpha_pos * r_pos
                hess[pos_mask] = alpha_pos
            else:
                # Linear region  
                grad[pos_mask] = alpha_pos * huber_delta * np.sign(r_pos)
                hess[pos_mask] = alpha_pos * huber_delta * 0.1  # Small hessian in linear region
        
        # Negative residuals (overestimation) - penalize less
        neg_mask = residuals <= 0
        if neg_mask.any():
            r_neg = residuals[neg_mask]
            if np.abs(r_neg) > huber_delta:
                # Quadratic region
                grad[neg_mask] = alpha_neg * r_neg
                hess[neg_mask] = alpha_neg
            else:
                # Linear region
                grad[neg_mask] = alpha_neg * huber_delta * np.sign(r_neg)
                hess[neg_mask] = alpha_neg * huber_delta * 0.1

        return grad, hess
        
    return asymmetric_regression_objective

def get_asymmetric_regression_loss_xgb(alpha_pos=0.7, alpha_neg=0.3, huber_delta=0.1):
    """
    Factory to return an asymmetric regression loss function for XGBoost.
    """
    def asymmetric_regression_objective(y_pred, y_true):
        y_true = y_true.ravel()
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Asymmetric Huber-like loss
        grad = np.zeros_like(residuals)
        hess = np.zeros_like(residuals)
        
        # Positive residuals (underestimation) - penalize more
        pos_mask = residuals > 0
        if pos_mask.any():
            r_pos = residuals[pos_mask]
            if np.abs(r_pos) > huber_delta:
                grad[pos_mask] = alpha_pos * r_pos
                hess[pos_mask] = alpha_pos
            else:
                grad[pos_mask] = alpha_pos * huber_delta * np.sign(r_pos)
                hess[pos_mask] = alpha_pos * huber_delta * 0.1
        
        # Negative residuals (overestimation) - penalize less
        neg_mask = residuals <= 0
        if neg_mask.any():
            r_neg = residuals[neg_mask]
            if np.abs(r_neg) > huber_delta:
                grad[neg_mask] = alpha_neg * r_neg
                hess[neg_mask] = alpha_neg
            else:
                grad[neg_mask] = alpha_neg * huber_delta * np.sign(r_neg)
                hess[neg_mask] = alpha_neg * huber_delta * 0.1

        return grad, hess
        
    return asymmetric_regression_objective

class AsymmetricRegressionLoss:
    """
    Production-grade Asymmetric Regression Loss for LightGBM in Financial ML.
    """
    
    def __init__(self, alpha_pos=0.7, alpha_neg=0.3, huber_delta=0.1, verbose=True):
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg  
        self.huber_delta = huber_delta
        self.verbose = verbose
        self._is_init = False

    def __call__(self, preds, train_data):
        if hasattr(train_data, 'get_label'):
            y_true = train_data.get_label()
        else:
            y_true = train_data

        # Calculate residuals
        residuals = y_true - preds
        
        # Asymmetric Huber-like loss
        grad = np.zeros_like(residuals)
        hess = np.zeros_like(residuals)
        
        # Positive residuals (underestimation) - penalize more
        pos_mask = residuals > 0
        if pos_mask.any():
            r_pos = residuals[pos_mask]
            if np.abs(r_pos) > self.huber_delta:
                grad[pos_mask] = self.alpha_pos * r_pos
                hess[pos_mask] = self.alpha_pos
            else:
                grad[pos_mask] = self.alpha_pos * self.huber_delta * np.sign(r_pos)
                hess[pos_mask] = self.alpha_pos * self.huber_delta * 0.1
        
        # Negative residuals (overestimation) - penalize less
        neg_mask = residuals <= 0
        if neg_mask.any():
            r_neg = residuals[neg_mask]
            if np.abs(r_neg) > self.huber_delta:
                grad[neg_mask] = self.alpha_neg * r_neg
                hess[neg_mask] = self.alpha_neg
            else:
                grad[neg_mask] = self.alpha_neg * self.huber_delta * np.sign(r_neg)
                hess[neg_mask] = self.alpha_neg * self.huber_delta * 0.1

        # Ensure positive hessian for numerical stability
        hess = np.maximum(hess, 1e-6)

        return grad, hess
