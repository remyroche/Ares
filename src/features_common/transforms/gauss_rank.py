
import numpy as np
import pandas as pd
from scipy.special import erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, List
from src.utils.tprint import tprint_warning

class GaussRankScaler(BaseEstimator, TransformerMixin):
    """
    Transform features to a Gaussian distribution using rank transformation.
    
    This is effectively a robust quantile normalization that maps the distribution
    of values to a standard normal distribution (Process is also known as 
    'Inverse Normal Transformation' or 'Van der Waerden scores').
    
    Advantages:
    - Highly robust to outliers
    - Preserves rank order
    - Enforces normality (crucial for GMM/PCA/Correlation)
    - Bounded 'magnitude' relationships
    """
    
    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon
        self.lower_bound = -1 + epsilon
        self.upper_bound = 1 - epsilon

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Apply GaussRank transformation.
        """
        try:
            # Convert to pandas structure for rank method
            if isinstance(X, np.ndarray):
                X_pd = pd.DataFrame(X)
                return_numpy = True
            else:
                X_pd = X
                return_numpy = False

            # Compute ranks (normalized to -1 to 1 range for erfinv)
            # Use 'average' tie handling to preserve information
            ranks = X_pd.rank(axis=0, method='average', pct=True)
            
            # Map [0, 1] to [-1+eps, 1-eps] to avoid infinity at extremes
            # Simplified: Scale to (-1+eps, 1-eps)
            ranks_scaled = ranks * (self.upper_bound - self.lower_bound) + self.lower_bound
            
            # Clip rigorously
            ranks_scaled = ranks_scaled.clip(self.lower_bound, self.upper_bound)
            
            # Apply inverse error function
            # erfinv range is (-1, 1) -> (-inf, inf)
            # result is multiplied by sqrt(2) to get std dev = 1 (Standard Normal)
            transformed = erfinv(ranks_scaled) * np.sqrt(2)
            
            if return_numpy:
                return transformed.values
            return transformed

        except Exception as e:
            tprint_warning(f"⚠️ GaussRank transformation failed: {e}")
            return X
