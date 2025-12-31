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
            # Hessian approximation: p * (1-p) * weight? Or just constant?
            # Using p*(1-p) is standard for log-loss based objectives, but let's be more precise or safe
            # A simplified hessian often works well: sigmoid derivative * loss second derivative factor
            # For simplicity and stability, often just H ~ abs(grad)? No.
            # Let's use constant 1.0 or p*(1-p) * alpha...
            # Actually, standard Focal Loss implementation in libraries often uses Autograd.
            # Let's use a robust approximation: p * (1-p) * alpha * (1-p)^gamma (roughly)
            # Or just p*(1-p)
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
