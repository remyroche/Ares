
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from typing import Dict, List, Tuple, Optional, Any
import logging

# Configure logger
logger = logging.getLogger(__name__)

def compute_ridge_monotonic_constraints(
    X: pd.DataFrame, 
    y: pd.Series, 
    alpha: float = 1.0, 
    n_bootstrap: int = 30, 
    threshold: float = 2.0,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Compute monotonic constraints using Bootstrapped Ridge Regression.
    
    Logic:
    1. Scale features (StandardScaler) to make coefficients comparable.
    2. Run n_bootstrap Ridge regressions on subsamples.
    3. Compute t-stats = Mean(Coef) / Std(Coef).
    4. If |t| > threshold:
        - t > threshold -> +1 (Increasing interaction)
        - t < -threshold -> -1 (Decreasing interaction)
        - else -> 0 (No constraint)
        
    This approach "balances" itself: in high-noise regimes, the standard error rises,
    naturally making the threshold harder to reach. In clear trends, it lowers.
    
    Args:
        X: Feature DataFrame
        y: Target variable (Series) - can be binary or continuous
        alpha: Ridge regularization strength
        n_bootstrap: Number of bootstrap iterations
        threshold: t-statistic threshold for significance (default 2.0 approx 95% CI)
        verbose: Print debug info
        
    Returns:
        Dictionary of {feature_name: constraint_value} where value is -1, 0, or 1.
    """
    try:
        if X.empty or len(y) == 0:
            return {col: 0 for col in X.columns}
        
        # Ensure numeric inputs
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
             return {col: 0 for col in X.columns}
             
        feature_names = X_numeric.columns.tolist()
        n_samples = len(X_numeric)
        
        # 1. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        # Align y
        y_arr = y.values
        if len(y_arr) != n_samples:
             # Try to align by index if possible
             y_aligned = y.reindex(X_numeric.index)
             y_arr = y_aligned.values
        
        # Store coefficients
        coef_list = []
        
        # 2. Bootstrap Ridge
        for i in range(n_bootstrap):
            # Resample (bootstrap with replacement)
            X_res, y_res = resample(X_scaled, y_arr, random_state=42+i)
            
            # Fit Ridge
            model = Ridge(alpha=alpha)
            model.fit(X_res, y_res)
            
            coef_list.append(model.coef_)
            
        # 3. Compute Statistics
        coef_matrix = np.vstack(coef_list)
        coef_mean = np.mean(coef_matrix, axis=0)
        coef_std = np.std(coef_matrix, axis=0) + 1e-9 # Avoid division by zero
        
        t_stats = coef_mean / coef_std
        
        # 4. Determine Constraints
        constraints = {}
        counts = {1: 0, -1: 0, 0: 0}
        
        for idx, feat in enumerate(feature_names):
            t_val = t_stats[idx]
            
            if t_val > threshold:
                constraint = 1
            elif t_val < -threshold:
                constraint = -1
            else:
                constraint = 0
                
            constraints[feat] = constraint
            counts[constraint] += 1
            
        if verbose:
            logger.info(f"Ridge Constraints Summary: Increasing={counts[1]}, Decreasing={counts[-1]}, None={counts[0]}")
            
        return constraints
        
    except Exception as e:
        logger.warning(f"Error computing ridge constraints: {e}")
        # Return no constraints on failure
        return {col: 0 for col in X.columns}
