import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
import logging

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Copy-paste of the XGBFocalLoss class from label_based_layer_2.py
class XGBFocalLoss:
    """
    Focal Loss for XGBoost (custom objective function).
    Matches RobustFocalLoss behavior for consistency across LGBM and XGB.
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
    
    def __call__(self, preds, dtrain):
        if hasattr(dtrain, 'get_label'):
            labels = dtrain.get_label()
        else:
            labels = dtrain
        
        # Convert logits to probabilities (standardized clipping)
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        
        # Safe log operations
        log_p = np.log(p + 1e-12)
        log_1mp = np.log(1 - p + 1e-12)
        
        # Focal loss terms
        term_pos = np.power(1 - p, self.gamma)
        term_neg = np.power(p, self.gamma)
        

        # Gradient
        grad = np.where(
            labels == 1,
            -self.alpha * term_pos * (1 - p - self.gamma * p * log_p),
            (1 - self.alpha) * term_neg * (p - self.gamma * (1 - p) * log_1mp)
        )
        grad = -grad # FLIP SIGN TEST

        
        
        # Hessian approximation (Binary Cross Entropy Hessian)
        # This guarantees positive curvature and stability.
        # h = p * (1 - p)
        # We can also scale it by alpha/gamma terms roughly, but p(1-p) is usually sufficient for Newton direction directionality.
        # Let's try simple robust approximation first:
        hess = p * (1.0 - p)
        
        # Hessian stability
        hess = np.maximum(hess, 1e-6)

        
        return grad, hess

def run_reproduction():
    print("Generating synthetic data...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # Instantiate custom objective
    focal_xgb = XGBFocalLoss(gamma=2.0, alpha=0.25)
    
    # Instantiate Classifier
    print("Initializing XGBClassifier...")
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        objective=focal_xgb, # Pass custom objective here
        eval_metric='auc',
        verbosity=1,
        random_state=42,
        n_jobs=1,
    )
    
    # Fit
    print("Fitting model...")
    try:
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )
    except Exception as e:
        print(f"Fit failed with error: {e}")
        return

    # Predict
    print("Predicting...")
    # predict_proba returns (N, 2), we want probability of class 1
    preds = model_xgb.predict_proba(X_val)[:, 1]
    
    # Check for constant predictions
    print(f"Predictions min: {preds.min()}, max: {preds.max()}, mean: {preds.mean()}")
    if preds.min() == preds.max():
        print("WARNING: Constant predictions detected!")
    
    # Score
    score = roc_auc_score(y_val, preds)
    print(f"Validation AUC: {score:.4f}")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def focal_loss_scalar(z, y, gamma, alpha):
    p = sigmoid(z)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    if y == 1:
        return -alpha * np.power(1 - p, gamma) * np.log(p)
    else:
        return -(1 - alpha) * np.power(p, gamma) * np.log(1 - p)

def check_gradients_numerically():
    print("Checking gradients numerically...")
    gamma = 2.0
    alpha = 0.25
    
    obj = XGBFocalLoss(gamma=gamma, alpha=alpha)
    
    logits = np.array([-2.0, 0.0, 2.0])
    labels = np.array([0, 1, 0])
    

    
    grad_custom, hess_custom = obj(logits, labels)
    
    eps = 1e-4
    for i, z in enumerate(logits):
        y = labels[i]
        
        # Numerical
        l_plus = focal_loss_scalar(z + eps, y, gamma, alpha)
        l_minus = focal_loss_scalar(z - eps, y, gamma, alpha)
        grad_num = (l_plus - l_minus) / (2 * eps)
        
        l_curr = focal_loss_scalar(z, y, gamma, alpha)
        hess_num = (l_plus - 2 * l_curr + l_minus) / (eps ** 2)
        
        # New Analytical v2
        p = sigmoid(z)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        log_p = np.log(p)
        log_1mp = np.log(1 - p)
        
        if y == 1:
            # Derived y=1 Hessian
            # h = alpha * p * (1-p)^gamma * ( gamma * [1 - p - gamma * p * log p] + (1-p) * [1 + gamma + gamma log p] )
            common = alpha * p * np.power(1 - p, gamma)
            term_A = gamma * (1 - p - gamma * p * log_p)
            term_B = (1 - p) * (1 + gamma + gamma * log_p)
            hess_v2 = common * (term_A + term_B)
        else:
            # Derived y=0 Hessian
            # h = (1-alpha) * p^gamma * (1-p) * ( gamma * [ p - gamma * (1-p) * log(1-p) ] + p * [ 1 - gamma + gamma log(1-p) ] )
            common = (1 - alpha) * np.power(p, gamma) * (1 - p)
            term_A = gamma * (p - gamma * (1 - p) * log_1mp)
            term_B = p * (1 - gamma + gamma * log_1mp)
            hess_v2 = common * (term_A + term_B)
            
        print(f"Logit: {z}, Label: {y}")
        print(f"  Grad: Custom={grad_custom[i]:.6f}, Num={grad_num:.6f}")
        print(f"  Hess: Custom={hess_custom[i]:.6f}, Num={hess_num:.6f}, NewV2={hess_v2:.6f}")
        print(f"  V2 Diff: {abs(hess_v2 - hess_num):.6f}")

if __name__ == "__main__":
    # check_gradients_numerically()
    run_reproduction()



