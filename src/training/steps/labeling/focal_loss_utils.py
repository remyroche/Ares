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

